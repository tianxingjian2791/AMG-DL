#ifndef DIFFUSIONMODEL_HPP
#define DIFFUSIONMODEL_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_control.h> 
#include <deal.II/lac/petsc_sparse_matrix.h> 
#include <deal.II/lac/petsc_vector.h> 
#include <deal.II/lac/petsc_solver.h> 
#include <deal.II/lac/petsc_precondition.h> 
#include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/petsc_sparse_matrix.templates.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <array>
#include <chrono>
#include <memory>

#include "AMGOperators.hpp"


namespace AMGDiffusion
{
  using namespace dealii;

  // Paper-aligned coefficient patterns:
  // (a) left/right split
  // (b) 2x2 checkerboard
  // (c) four vertical stripes
  // (d) 4x4 checkerboard
  enum class DiffusionPattern
  {
    VERTICAL_STRIPES,
    CHECKERBOARD_2X2,
    VERTICAL_STRIPES_4,
    CHECKERBOARD_4X4
  };

  inline const char *pattern_to_string(DiffusionPattern pattern)
  {
    switch (pattern)
    {
    case DiffusionPattern::VERTICAL_STRIPES:
      return "vertical_stripes";
    case DiffusionPattern::CHECKERBOARD_2X2:
      return "checkerboard_2x2";
    case DiffusionPattern::VERTICAL_STRIPES_4:
      return "vertical_stripes_4";
    case DiffusionPattern::CHECKERBOARD_4X4:
      return "checkerboard_4x4";
    default:
      return "unknown";
    }
  }

  // Manufactured solution matched to the paper's pattern families.
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(DiffusionPattern pattern)
      : Function<dim>(1)
      , pattern(pattern)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const double x = p[0];
      const double y = p[1];
      switch (pattern)
      {
      case DiffusionPattern::VERTICAL_STRIPES:
      case DiffusionPattern::CHECKERBOARD_2X2:
        return std::cos(M_PI * x) * std::cos(M_PI * y);
      case DiffusionPattern::VERTICAL_STRIPES_4:
      case DiffusionPattern::CHECKERBOARD_4X4:
        return std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y);
      default:
        AssertThrow(false, ExcNotImplemented());
        return 0.0;
      }
    }

  private:
    DiffusionPattern pattern;
  };

  // Right-hand side corresponding to the chosen manufactured solution.
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(DiffusionPattern pattern)
      : Function<dim>(1)
      , pattern(pattern)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const double x = p[0];
      const double y = p[1];
      switch (pattern)
      {
      case DiffusionPattern::VERTICAL_STRIPES:
      case DiffusionPattern::CHECKERBOARD_2X2:
      {
        const double u_value = std::cos(M_PI * x) * std::cos(M_PI * y);
        return 2 * M_PI * M_PI * u_value; // -Δu = 2π²u
      }
      case DiffusionPattern::VERTICAL_STRIPES_4:
      case DiffusionPattern::CHECKERBOARD_4X4:
      {
        const double u_value = std::cos(2 * M_PI * x) * std::cos(2 * M_PI * y);
        return 8 * M_PI * M_PI * u_value; // -Δu = 8π²u
      }
      default:
        AssertThrow(false, ExcNotImplemented());
        return 0.0;
      }
    }

  private:
    DiffusionPattern pattern;
  };

  // Piecewise-constant diffusion coefficient aligned with the mesh.
  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    DiffusionCoefficient(DiffusionPattern pattern, double epsilon)
      : Function<dim>(1)
      , pattern(pattern)
      , epsilon(epsilon)
    {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      const double x = p[0];
      const double y = p[1];
      const double high = std::pow(10.0, epsilon);
      const double tol = 1e-12;

      switch (pattern)
      {
      case DiffusionPattern::VERTICAL_STRIPES:
        return (x < 0.0 + tol) ? 1.0 : high;
      case DiffusionPattern::CHECKERBOARD_2X2:
      {
        const int i = std::min(static_cast<int>(std::floor(x + 1.0)), 1);
        const int j = std::min(static_cast<int>(std::floor(y + 1.0)), 1);
        return ((i + j) % 2 == 0) ? 1.0 : high;
      }
      case DiffusionPattern::VERTICAL_STRIPES_4:
        if (x < -0.5 + tol) return 1.0;
        if (x < 0.0 + tol) return high;
        if (x < 0.5 + tol) return 1.0;
        return high;
      case DiffusionPattern::CHECKERBOARD_4X4:
      {
        int i = static_cast<int>(std::floor((x + 1.0) / 0.5));
        int j = static_cast<int>(std::floor((y + 1.0) / 0.5));
        i = std::min(i, 3);
        j = std::min(j, 3);
        return ((i + j) % 2 == 0) ? 1.0 : high;
      }
      default:
        AssertThrow(false, ExcNotImplemented());
        return 0.0;
      }
    }

  private:
    DiffusionPattern pattern;
    double epsilon;
  };

  // Output format enum
  enum class OutputFormat
  {
    THETA_CNN,   // theta prediction - CNN format (pooled)
    THETA_GNN,   // theta prediction - GNN format (graph)
    P_VALUE      // P-value prediction (with C/F splitting and P, S matrices)
  };

  // Solver
  template <int dim>
  class Solver
  {
  public:
    Solver(DiffusionPattern pattern, double epsilon, unsigned int refinement, bool use_simplex = false);
    Solver(double epsilon, unsigned int refinement, bool use_simplex = false);
    void set_pattern(DiffusionPattern pattern);
    void set_theta(double theta);
    void set_epsilon(double epsilon);
    void set_refinement(unsigned int refinement);
    void set_output_format(OutputFormat format);
    void run(std::ofstream &file);
    void run(std::ofstream &file, OutputFormat format);

    // Getter methods for unified interface
    AMGOperators::CSRMatrix get_system_matrix_csr();
    double get_mesh_size() const;
    double get_convergence_factor() const;
    double get_linear_solve_elapsed_sec() const;
    unsigned int get_solver_iterations() const;
    unsigned int get_amg_hierarchy_levels() const;

    // support static function
    static std::vector<double> linspace(double start, double end, size_t num_points)
    {
      std::vector<double> result;
      if (num_points == 0) return result;
      if (num_points == 1) {
          result.push_back(start);
          return result;
      }

      double step = (end - start) / (num_points - 1);
      for (size_t i = 0; i < num_points; i++) {
          result.push_back(start + i * step);
      }
      return result;
    }

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve(std::ofstream &file);
    void solve_with_format(std::ofstream &file, OutputFormat format);
    void write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix, std::ofstream &file, double rho, double h);
    void write_pvalue_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix, std::ofstream &file, double rho, double h);
    AMGOperators::CSRMatrix petsc_to_csr(const PETScWrappers::MPI::SparseMatrix &matrix);
    void update_amg_hierarchy_levels(
      const dealii::PETScWrappers::PreconditionBoomerAMG &preconditioner);


    // mode parameter
    DiffusionPattern pattern;
    double theta;
    double epsilon;
    unsigned int refinement;
    bool use_simplex;  // false = quad cells (default), true = triangle cells
    OutputFormat output_format;

    // Convergence metrics (to be stored after solve)
    double convergence_factor;
    double linear_solve_elapsed_sec;
    double mesh_size_h;
    unsigned int solver_iterations;
    unsigned int amg_hierarchy_levels;

    // Grids and finite elements
    dealii::Triangulation<dim> triangulation;
    std::unique_ptr<dealii::FiniteElement<dim>> fe;
    std::unique_ptr<dealii::Mapping<dim>> mapping;
    dealii::DoFHandler<dim> dof_handler;
    dealii::SolverControl solver_control; // solver controller

    // Constraint and system matrix
    dealii::AffineConstraints<double> constraints;
    dealii::PETScWrappers::MPI::SparseMatrix system_matrix;
    dealii::PETScWrappers::MPI::Vector solution;
    // dealii::PETScWrappers::MPI::Vector init_solution;
    dealii::PETScWrappers::MPI::Vector system_rhs;

    // Exact solution and right hand
    ExactSolution<dim> exact_solution;
    RightHandSide<dim> right_hand_side;
    DiffusionCoefficient<dim> diffusion_coefficient;
  };

  template <int dim>
  Solver<dim>::Solver(DiffusionPattern pattern, double epsilon, unsigned int refinement, bool use_simplex)
    : pattern(pattern)
    , epsilon(epsilon)
    , refinement(refinement)
    , use_simplex(use_simplex)
    , output_format(OutputFormat::THETA_GNN) // default format
    , convergence_factor(1.0)
    , mesh_size_h(0.0)
    , solver_iterations(0)
    , amg_hierarchy_levels(0)
    , dof_handler(triangulation)
    , solver_control(1000, 1e-12) // max iterations is 1000，tolerance is 1e-12
    , exact_solution(pattern)
    , right_hand_side(pattern)
    , diffusion_coefficient(pattern, epsilon)
  {
    if (use_simplex)
      {
        // Triangle cells (P1 Lagrange), with an isoparametric mapping built
        // from the same element, as in deal.II's step_3_simplex tutorial.
        fe = std::make_unique<dealii::FE_SimplexP<dim>>(1);
        mapping = std::make_unique<dealii::MappingFE<dim>>(dealii::FE_SimplexP<dim>(1));
      }
    else
      {
        // Quad cells (Q1 Lagrange) -- unchanged default behavior.
        fe = std::make_unique<dealii::FE_Q<dim>>(1);
        mapping = std::make_unique<dealii::MappingQ1<dim>>();
      }
  }

  template <int dim>
  Solver<dim>::Solver(double epsilon, unsigned int refinement, bool use_simplex)
    : Solver(DiffusionPattern::VERTICAL_STRIPES, epsilon, refinement, use_simplex)
  {}

  template <int dim>
  void Solver<dim>::make_grid()
  {
    if (use_simplex)
      {
        // Triangle mesh: subdivide the square into `2^refinement` cells per
        // side (each further split into simplices), matching the number of
        // subdivisions the quad path gets from refine_global(refinement).
        GridGenerator::subdivided_hyper_cube_with_simplices(
          triangulation, 1u << refinement, -1.0, 1.0);
      }
    else
      {
        // Generate square grids
        GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
        triangulation.refine_global(refinement);
      }

    // Store the cell side length used as h in the paper tables.
    mesh_size_h = GridTools::maximal_cell_diameter(triangulation) / std::sqrt(dim);
    // std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
  }

  template <int dim>
  void Solver<dim>::set_theta(double theta)
  {
    this->theta = theta;
    // std::cout<<this->theta<<std::endl;
  }

  template <int dim>
  void Solver<dim>::set_pattern(DiffusionPattern pattern)
  {
    this->pattern = pattern;
    exact_solution = ExactSolution<dim>(pattern);
    right_hand_side = RightHandSide<dim>(pattern);
    diffusion_coefficient = DiffusionCoefficient<dim>(pattern, epsilon);
  }

  template <int dim>
  void Solver<dim>::set_epsilon(double epsilon)
  {
    this->epsilon = epsilon;
  }

  template <int dim>
  void Solver<dim>::set_refinement(unsigned int refinement)
  {
    this->refinement = refinement;
  }

  template <int dim>
  void Solver<dim>::set_output_format(OutputFormat format)
  {
    this->output_format = format;
  }

  template <int dim>
  void Solver<dim>::setup_system()
  {
    // Setup dofs
    dof_handler.distribute_dofs(*fe);
    // std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    // Initialize MPI variables
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Create sparsity pattern
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();

    // Initialize matrices and vectors
    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    // init_solution = solution;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // Setup boundary conditions
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(*mapping,
                                            dof_handler,
                                            0,
                                            exact_solution,
                                            constraints);
    constraints.close();
  }

  template <int dim>
  void Solver<dim>::assemble_system()
  {
    std::unique_ptr<Quadrature<dim>> quadrature_formula;
    if (use_simplex)
      quadrature_formula = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
    else
      quadrature_formula = std::make_unique<QGauss<dim>>(fe->degree + 1);

    FEValues<dim> fe_values(*mapping,
                           *fe,
                           *quadrature_formula,
                           update_values | update_gradients |
                           update_JxW_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula->size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Traverse all the cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);

      // Get the diffusion coefficient of the current cell
      const double mu = diffusion_coefficient.value(fe_values.quadrature_point(0));

      // Assemble the matrix and right handside for the cell
      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
      {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            cell_matrix(i, j) += mu *
                                 fe_values.shape_grad(i, q_index) *
                                 fe_values.shape_grad(j, q_index) *
                                 fe_values.JxW(q_index);
          }

          // right handside：f * phi_i
          cell_rhs(i) += (right_hand_side.value(fe_values.quadrature_point(q_index)) *
                          fe_values.shape_value(i, q_index) *
                          fe_values.JxW(q_index));
        }
      }

      // Add the cell contribution to the global system
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                            cell_rhs,
                                            local_dof_indices,
                                            system_matrix,
                                            system_rhs);
    }

    // apply the constraint to and comress the matrix
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void Solver<dim>::solve(std::ofstream &file)
  {
    // solution = init_solution;
    // setup the parameters of the solver
    dealii::PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
    dealii::PETScWrappers::PreconditionBoomerAMG preconditioner;

    // configure the parameters of BoomerAMG
    dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.strong_threshold = theta;  // setup the strong threshold θ(adjustable)
    // std::cout<<"strong threshold: "<<data.strong_threshold<<std::endl;
    data.symmetric_operator = true; // setup the symmetric operator

    // Initialize the preconditioner of AMG
    preconditioner.initialize(system_matrix, data);
    preconditioner.setup();

    PETScWrappers::MPI::Vector residual(system_rhs);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double init_r_norm = residual.l2_norm();
    // std::cout<<init_r_norm<<" ";

    // Time only the preconditioned CG solve, excluding grid/setup/assembly and reporting.
    const auto linear_solve_start = std::chrono::steady_clock::now();
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    const auto linear_solve_end = std::chrono::steady_clock::now();
    linear_solve_elapsed_sec = std::chrono::duration<double>(linear_solve_end - linear_solve_start).count();

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double final_r_norm = residual.l2_norm();  
    // std::cout<<final_r_norm<<std::endl;

    update_amg_hierarchy_levels(preconditioner);

    // Print the iterative information
    // std::cout << "   Solver converged in " << solver_control.last_step()
    //           << " iterations." << std::endl;

    const unsigned int k = solver_control.last_step();
    solver_iterations = k;
    if (k < 1) {
      std::cerr << "Warning: Insufficient residuals recorded (" 
                << k << "). Returning rho=0." << std::endl;
      return;
  }

    // ρ = (||r_k|| / ||r_0||)^{1/k}
    const double rho = (k > 0) ? std::pow(final_r_norm / init_r_norm, 1.0 / k) : 0.0;
    double h = mesh_size_h; // Cell side length, matching the paper's h values.

    // Store for getter methods
    convergence_factor = rho;
    // mesh_size_h already set in make_grid()

    write_matrix_to_csv(system_matrix, file, rho, h);


    // Apply the constraints
    constraints.distribute(solution);

    // return rho;
  }
    
  template <int dim>
  void Solver<dim>::write_matrix_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix,
    std::ofstream &file,
    double rho,
    double h)
  {  
    const unsigned int m = matrix.m();
    const unsigned int n = matrix.n();
  
    PetscInt start, end;
    MatGetOwnershipRange(matrix, &start, &end);
    
    std::vector<PetscInt> row_ptr;
    std::vector<PetscInt> col_ind;
    std::vector<PetscScalar> values;
    
    PetscInt idx = 0;
    for (PetscInt i = start; i < end; i++) {
      PetscInt ncols;
      const PetscInt* cols;
      const PetscScalar* vals;
      MatGetRow(matrix, i, &ncols, &cols, &vals);
      
      row_ptr.push_back(idx);
      double zero_tol = 1e-12;
      for (PetscInt j = 0; j < ncols; j++) {
        // Filter the zeros explicitly
        if (std::abs(vals[j]) > zero_tol){
          col_ind.push_back(cols[j]);
          values.push_back(vals[j]);
          idx++;
        }
        
      }
      MatRestoreRow(matrix, i, &ncols, &cols, &vals);
    }
    row_ptr.push_back(idx);
  
    // write m(rows), n(cols), rho, h, nnz
    file << m << "," << n << "," << theta << "," << rho << "," << h << "," << values.size();
  
    // write non-zero value
    for (const auto &val : values)
    file << "," << val;
  
    // write row ptrs
    for (const auto &r : row_ptr)
    file << "," << r;
  
    // write col indices
    for (const auto &c : col_ind)
    file << "," << c;
  
    file << "\n";
  }

  /**
   * Convert PETSc matrix to CSRMatrix format for AMG operations
   */
  template <int dim>
  AMGOperators::CSRMatrix Solver<dim>::petsc_to_csr(const PETScWrappers::MPI::SparseMatrix &matrix)
  {
    const unsigned int n = matrix.m();
    AMGOperators::CSRMatrix csr(n, n);

    PetscInt start, end;
    MatGetOwnershipRange(matrix, &start, &end);

    std::vector<std::vector<int>> temp_cols(n);
    std::vector<std::vector<double>> temp_vals(n);

    // Extract matrix data
    for (PetscInt i = start; i < end; i++)
    {
      PetscInt ncols;
      const PetscInt* cols;
      const PetscScalar* vals;
      MatGetRow(matrix, i, &ncols, &cols, &vals);

      for (PetscInt j = 0; j < ncols; j++)
      {
        if (std::abs(vals[j]) > 1e-12)  // Filter near-zeros
        {
          temp_cols[i].push_back(cols[j]);
          temp_vals[i].push_back(vals[j]);
        }
      }

      MatRestoreRow(matrix, i, &ncols, &cols, &vals);
    }

    // Build CSR format
    csr.row_ptr[0] = 0;
    for (unsigned int i = 0; i < n; i++)
    {
      csr.row_ptr[i + 1] = csr.row_ptr[i] + temp_cols[i].size();
      csr.col_indices.insert(csr.col_indices.end(), temp_cols[i].begin(), temp_cols[i].end());
      csr.values.insert(csr.values.end(), temp_vals[i].begin(), temp_vals[i].end());
    }

    return csr;
  }

  // Write P-value format CSV with C/F splitting and P, S matrices
  // Format: n, n, theta, rho, h, nnz, [A_values], [A_row_ptrs], [A_col_indices], num_coarse, 
  // [coarse_indices], nnz_P, [P_values], [P_row_ptrs], [P_col_indices], nnz_S, [S_values], [S_row_ptrs], [S_col_indices]
  template <int dim>
  void Solver<dim>::write_pvalue_to_csv(const PETScWrappers::MPI::SparseMatrix &matrix,
                                         std::ofstream &file,
                                         double rho,
                                         double h)
  {
    // Convert to CSR format
    AMGOperators::CSRMatrix A = petsc_to_csr(matrix);

    const unsigned int n = A.n_rows;

    // Perform C/F splitting using theta
    std::vector<int> cf_markers = AMGOperators::classical_cf_splitting(A, theta);
    std::vector<int> coarse_nodes = AMGOperators::extract_coarse_nodes(cf_markers);

    // Compute prolongation matrix P
    AMGOperators::CSRMatrix P = AMGOperators::compute_baseline_prolongation(A, cf_markers);

    // Compute strength matrix S
    AMGOperators::CSRMatrix S = AMGOperators::compute_strength_matrix(A, theta);

    // Write to CSV
    // Header: n, n, theta, rho, h, nnz_A
    file << n << "," << n << "," << theta << "," << rho << "," << h << "," << A.nnz();

    // Write A matrix values
    for (const auto &val : A.values)
      file << "," << val;

    // Write A row pointers
    for (const auto &r : A.row_ptr)
      file << "," << r;

    // Write A column indices
    for (const auto &c : A.col_indices)
      file << "," << c;

    // Write number of coarse nodes
    file << "," << coarse_nodes.size();

    // Write coarse node indices
    for (const auto &c : coarse_nodes)
      file << "," << c;

    // Write P matrix: nnz_P, values, row_ptr, col_indices
    file << "," << P.nnz();
    for (const auto &val : P.values)
      file << "," << val;
    for (const auto &r : P.row_ptr)
      file << "," << r;
    for (const auto &c : P.col_indices)
      file << "," << c;

    // Write S matrix: nnz_S, values, row_ptr, col_indices
    file << "," << S.nnz();
    for (const auto &val : S.values)
      file << "," << val;
    for (const auto &r : S.row_ptr)
      file << "," << r;
    for (const auto &c : S.col_indices)
      file << "," << c;

    file << "\n";
  }

  // Solve with specified output format
  template <int dim>
  void Solver<dim>::solve_with_format(std::ofstream &file, OutputFormat format)
  {
    // Solve system (same as before)
    dealii::PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
    dealii::PETScWrappers::PreconditionBoomerAMG preconditioner;

    dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.strong_threshold = theta;
    data.symmetric_operator = true;

    preconditioner.initialize(system_matrix, data);
    preconditioner.setup();

    PETScWrappers::MPI::Vector residual(system_rhs);

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double init_r_norm = residual.l2_norm();

    const auto linear_solve_start = std::chrono::steady_clock::now();
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    const auto linear_solve_end = std::chrono::steady_clock::now();
    linear_solve_elapsed_sec = std::chrono::duration<double>(linear_solve_end - linear_solve_start).count();

    system_matrix.vmult(residual, solution);
    residual -= system_rhs;
    double final_r_norm = residual.l2_norm();

    update_amg_hierarchy_levels(preconditioner);

    const unsigned int k = solver_control.last_step();
    solver_iterations = k;
    if (k < 1)
    {
      std::cerr << "Warning: Insufficient residuals recorded ("
                << k << "). Returning rho=0." << std::endl;
      return;
    }

    const double rho = (k > 0) ? std::pow(final_r_norm / init_r_norm, 1.0 / k) : 0.0;
    double h = mesh_size_h;
    convergence_factor = rho;

    // Write based on format
    if (format == OutputFormat::P_VALUE)
    {
      write_pvalue_to_csv(system_matrix, file, rho, h);
    }
    else
    {
      // THETA_CNN and THETA_GNN both use same format (for now)
      write_matrix_to_csv(system_matrix, file, rho, h);
    }

    constraints.distribute(solution);
  }

  template <int dim>
  void Solver<dim>::run(std::ofstream &file)
  {
    amg_hierarchy_levels = 0;
    solver_iterations = 0;
    convergence_factor = 1.0;
    linear_solve_elapsed_sec = 0.0;

    make_grid();
    setup_system();
    assemble_system();

    solve(file);
  }

  template <int dim>
  void Solver<dim>::run(std::ofstream &file, OutputFormat format)
  {
    amg_hierarchy_levels = 0;
    solver_iterations = 0;
    convergence_factor = 1.0;
    linear_solve_elapsed_sec = 0.0;

    make_grid();
    setup_system();
    assemble_system();

    solve_with_format(file, format);

  }

  // Getter methods for unified interface
  template <int dim>
  AMGOperators::CSRMatrix Solver<dim>::get_system_matrix_csr()
  {
    return petsc_to_csr(system_matrix);
  }

  template <int dim>
  double Solver<dim>::get_mesh_size() const
  {
    return mesh_size_h;
  }

  template <int dim>
  double Solver<dim>::get_convergence_factor() const
  {
    return convergence_factor;
  }

  template <int dim>
  double Solver<dim>::get_linear_solve_elapsed_sec() const
  {
    return linear_solve_elapsed_sec;
  }

  template <int dim>
  unsigned int Solver<dim>::get_solver_iterations() const
  {
    return solver_iterations;
  }

  template <int dim>
  unsigned int Solver<dim>::get_amg_hierarchy_levels() const
  {
    return amg_hierarchy_levels;
  }

  template <int dim>
  void Solver<dim>::update_amg_hierarchy_levels(
    const dealii::PETScWrappers::PreconditionBoomerAMG &preconditioner)
  {
    PetscInt num_levels = 0;
    Mat *coarse_operators = nullptr;
    const PetscErrorCode ierr = PCGetCoarseOperators(preconditioner.get_pc(),
                                                     &num_levels,
                                                     &coarse_operators);
    if (ierr != 0)
      throw std::runtime_error("Failed to read BoomerAMG hierarchy levels from PETSc.");
    amg_hierarchy_levels = static_cast<unsigned int>(num_levels);

    for (PetscInt level = 0; level < num_levels - 1; ++level)
      MatDestroy(&coarse_operators[level]);
    PetscFree(coarse_operators);
  }

  
  // Generate complete dataset (LEGACY - deprecated, will be removed)
  void generate_dataset(std::ofstream &file, std::string train_flag)
  {
    std::cerr << "ERROR: generate_dataset() is deprecated." << std::endl;
    std::cerr << "This legacy dataset path has been replaced by the unified generator." << std::endl;
    std::cerr << "Please use the unified generator: ./generate_amg_data" << std::endl;
    throw std::runtime_error("Deprecated function called");
  }

} // namespace AMGTest

#endif // DIFUSSIONMODEL_HPP 
