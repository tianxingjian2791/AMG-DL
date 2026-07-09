/* polydg_amg_solve.cc
 *
 * Standalone HYPRE BoomerAMG-preconditioned CG solver for the PolyDG diffusion
 * matrices exported by polydg_diffusion_hetero.cc. Replicates the exact solve
 * from include/DiffusionModel.hpp so that the polygonal convergence factor rho
 * lands on the SAME scale as the original FEM training data:
 *
 *   solver   : PETScWrappers::SolverCG
 *   control  : SolverControl(1000, 1e-12)          (max_iter=1000, tol=1e-12)
 *   precond  : PreconditionBoomerAMG, symmetric_operator=true,
 *              strong_threshold = theta
 *   x0       : 0
 *   rho      : (||A x_k - b|| / ||A*0 - b||)^(1/k),  k = last_step()
 *   n_levels : PCGetCoarseOperators(pc, &num_levels, ...)
 *   timing   : std::chrono::steady_clock around solve() only
 *
 * Usage:
 *   polydg_amg_solve <matrix.mtx> <rhs.mtx> <theta1,theta2,...>
 *
 * Output (stdout), one CSV row per theta (no header):
 *   theta,rho,iterations,elapsed_sec,n_levels
 */

#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <petscpc.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dealii;

// PLACEHOLDER_READERS

// Skip MatrixMarket comment/banner lines (those beginning with '%').
static std::string next_data_line(std::istream &in)
{
  std::string line;
  while (std::getline(in, line))
    {
      if (line.empty())
        continue;
      if (line[0] == '%')
        continue;
      return line;
    }
  return std::string();
}

// Read a MatrixMarket coordinate matrix into a PETSc serial sparse matrix.
// Handles 'general' and 'symmetric' (mirrors the lower/upper entry). Matches
// the output of PolyUtils::write_to_matrix_market_format.
static void read_mtx_matrix(const std::string          &path,
                            PETScWrappers::MPI::SparseMatrix &A)
{
  std::ifstream in(path);
  AssertThrow(in, ExcMessage("cannot open matrix file: " + path));

  std::string banner;
  std::getline(in, banner);
  const bool symmetric = banner.find("symmetric") != std::string::npos;

  const std::string dims = next_data_line(in);
  std::istringstream ds(dims);
  unsigned int nrows, ncols, nnz;
  ds >> nrows >> ncols >> nnz;

  // First pass: build sparsity pattern.
  std::vector<std::vector<unsigned int>> cols(nrows);
  std::vector<std::vector<double>>       vals(nrows);
  for (unsigned int e = 0; e < nnz; ++e)
    {
      const std::string l = next_data_line(in);
      std::istringstream ls(l);
      unsigned int r, c;
      double v;
      ls >> r >> c >> v;
      --r; --c; // MatrixMarket is 1-indexed
      cols[r].push_back(c);
      vals[r].push_back(v);
      if (symmetric && r != c)
        {
          cols[c].push_back(r);
          vals[c].push_back(v);
        }
    }

  DynamicSparsityPattern dsp(nrows, ncols);
  for (unsigned int r = 0; r < nrows; ++r)
    for (const auto c : cols[r])
      dsp.add(r, c);

  const IndexSet all = complete_index_set(nrows);
  A.reinit(all, all, dsp, MPI_COMM_WORLD);
  for (unsigned int r = 0; r < nrows; ++r)
    for (std::size_t k = 0; k < cols[r].size(); ++k)
      A.set(r, cols[r][k], vals[r][k]);
  A.compress(VectorOperation::insert);
}

// Read a MatrixMarket array (dense) vector into a PETSc vector.
static void read_mtx_vector(const std::string        &path,
                            PETScWrappers::MPI::Vector &b)
{
  std::ifstream in(path);
  AssertThrow(in, ExcMessage("cannot open rhs file: " + path));

  std::string banner;
  std::getline(in, banner);

  const std::string dims = next_data_line(in);
  std::istringstream ds(dims);
  unsigned int n, ncols;
  ds >> n >> ncols;

  b.reinit(complete_index_set(n), MPI_COMM_WORLD);
  for (unsigned int i = 0; i < n; ++i)
    {
      const std::string l = next_data_line(in);
      std::istringstream ls(l);
      double v;
      ls >> v;
      b(i) = v;
    }
  b.compress(VectorOperation::insert);
}

// PLACEHOLDER_SOLVE

// Number of AMG hierarchy levels (matches DiffusionModel::update_amg_hierarchy_levels).
static unsigned int amg_num_levels(const PETScWrappers::PreconditionBoomerAMG &prec)
{
  PetscInt num_levels = 0;
  Mat     *coarse_ops = nullptr;
  const PetscErrorCode ierr =
    PCGetCoarseOperators(prec.get_pc(), &num_levels, &coarse_ops);
  if (ierr != 0)
    return 0;
  for (PetscInt lvl = 0; lvl < num_levels - 1; ++lvl)
    MatDestroy(&coarse_ops[lvl]);
  PetscFree(coarse_ops);
  return static_cast<unsigned int>(num_levels);
}

struct SolveResult
{
  double       rho;
  unsigned int iterations;
  double       elapsed_sec;
  unsigned int n_levels;
};

// One BoomerAMG-preconditioned CG solve at a given strong threshold theta.
static SolveResult solve_one(const PETScWrappers::MPI::SparseMatrix &A,
                             const PETScWrappers::MPI::Vector       &b,
                             double                                  theta)
{
  SolverControl solver_control(1000, 1e-12); // same as DiffusionModel.hpp
  PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);

  PETScWrappers::PreconditionBoomerAMG           preconditioner;
  PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
  data.strong_threshold  = theta;
  data.symmetric_operator = true;
  preconditioner.initialize(A, data);
  preconditioner.setup();

  PETScWrappers::MPI::Vector solution(b);
  solution = 0.0;

  PETScWrappers::MPI::Vector residual(b);
  A.vmult(residual, solution);
  residual -= b;
  const double init_r_norm = residual.l2_norm(); // = ||b|| since x0=0

  const auto t0 = std::chrono::steady_clock::now();
  solver.solve(A, solution, b, preconditioner);
  const auto t1 = std::chrono::steady_clock::now();

  A.vmult(residual, solution);
  residual -= b;
  const double final_r_norm = residual.l2_norm();

  const unsigned int k = solver_control.last_step();
  const double rho =
    (k > 0 && init_r_norm > 0.0)
      ? std::pow(final_r_norm / init_r_norm, 1.0 / k)
      : 0.0;

  SolveResult r;
  r.rho         = rho;
  r.iterations  = k;
  r.elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
  r.n_levels    = amg_num_levels(preconditioner);
  return r;
}

static std::vector<double> parse_thetas(const std::string &csv)
{
  std::vector<double> out;
  std::istringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty())
      out.push_back(std::stod(tok));
  return out;
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  if (argc < 4)
    {
      std::cerr << "usage: " << argv[0]
                << " <matrix.mtx> <rhs.mtx> <theta1,theta2,...>\n";
      return 1;
    }

  const std::string matrix_path = argv[1];
  const std::string rhs_path    = argv[2];
  const std::vector<double> thetas = parse_thetas(argv[3]);

  PETScWrappers::MPI::SparseMatrix A;
  PETScWrappers::MPI::Vector       b;
  read_mtx_matrix(matrix_path, A);
  read_mtx_vector(rhs_path, b);

  AssertThrow(A.m() == b.size(),
              ExcMessage("matrix/rhs size mismatch"));

  std::cout.precision(10);
  // CSV rows: theta,rho,iterations,elapsed_sec,n_levels
  for (const double theta : thetas)
    {
      const SolveResult r = solve_one(A, b, theta);
      std::cout << theta << "," << r.rho << "," << r.iterations << ","
                << r.elapsed_sec << "," << r.n_levels << "\n";
    }
  return 0;
}



