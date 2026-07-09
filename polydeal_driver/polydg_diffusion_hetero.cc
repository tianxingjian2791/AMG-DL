/* ------------------------------------------------------------------
 * Heterogeneous PolyDG diffusion driver (Option A, dataset generation).
 *
 * Assembles the SIP-DG matrix for  -div(mu(x) grad u) = 0  (matrix only)
 * on an agglomerated polytopic mesh, with mu(x) a PIECEWISE-CONSTANT
 * coefficient following the four paper-aligned patterns (2111.01629v2,
 * replicated from AMG-DL/include/DiffusionModel.hpp) on the domain [-1,1]^2:
 *   0 VERTICAL_STRIPES   : left/right split  (x<0 -> 1, else high)
 *   1 CHECKERBOARD_2X2   : 2x2 checkerboard
 *   2 VERTICAL_STRIPES_4 : four alternating vertical stripes
 *   3 CHECKERBOARD_4X4   : 4x4 checkerboard
 * with high = 10^epsilon, low = 1.
 *
 * mu is evaluated PER QUADRATURE POINT (mu is a spatial function). Polydeal's
 * agglomerated quadrature is gathered from the fine background cells, and the
 * pattern jumps (multiples of 0.5) align with background-cell boundaries at
 * refinement >= 2, so intra-polytope jumps are integrated exactly.
 *
 * Manual coefficient-aware SIP-DG assembly adapted from examples/minimal_SIP.cc
 * (PolyUtils::assemble_dg_matrix is coefficient-free and cannot be used).
 *
 * Exports:
 *   <prefix>_matrix.mtx     Trilinos SIP-DG matrix (MatrixMarket)
 *   <prefix>_geometry.csv   per-polytope center, diameter, volume, ..., mu
 *                            (mu = representative value at the bbox center)
 *
 * CLI: ./polydg_diffusion_hetero [n_sub] [degree] [refine] [pattern] [epsilon] [prefix]
 * ------------------------------------------------------------------ */

#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace dealii;

// Paper-aligned coefficient patterns (see file header). Values match
// AMG-DL/include/DiffusionModel.hpp exactly.
enum class DiffusionPattern
{
  VERTICAL_STRIPES   = 0,
  CHECKERBOARD_2X2   = 1,
  VERTICAL_STRIPES_4 = 2,
  CHECKERBOARD_4X4   = 3
};

inline const char *pattern_name(DiffusionPattern p)
{
  switch (p)
    {
      case DiffusionPattern::VERTICAL_STRIPES:   return "vertical_stripes";
      case DiffusionPattern::CHECKERBOARD_2X2:   return "checkerboard_2x2";
      case DiffusionPattern::VERTICAL_STRIPES_4: return "vertical_stripes_4";
      case DiffusionPattern::CHECKERBOARD_4X4:   return "checkerboard_4x4";
      default:                                   return "unknown";
    }
}

// mu(x) for the given pattern on [-1,1]^2; high = 10^epsilon, low = 1.
template <int dim>
double
pattern_mu(const Point<dim> &pt, DiffusionPattern pattern, double epsilon)
{
  const double x    = pt[0];
  const double y    = (dim > 1) ? pt[1] : 0.0;
  const double high = std::pow(10.0, epsilon);
  const double tol  = 1e-12;

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
        if (x <  0.0 + tol) return high;
        if (x <  0.5 + tol) return 1.0;
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
        return 1.0;
    }
}

// Frequency factor a of the manufactured solution: a=1 for patterns 0,1
// (cos(pi x)cos(pi y)); a=2 for patterns 2,3 (cos(2pi x)cos(2pi y)). Chosen so
// grad(u).n = 0 at every mu-jump line, making cos the exact solution of the
// heterogeneous problem. Matches include/DiffusionModel.hpp ExactSolution.
inline double pattern_freq(DiffusionPattern pattern)
{
  switch (pattern)
    {
      case DiffusionPattern::VERTICAL_STRIPES:
      case DiffusionPattern::CHECKERBOARD_2X2:
        return 1.0;
      case DiffusionPattern::VERTICAL_STRIPES_4:
      case DiffusionPattern::CHECKERBOARD_4X4:
        return 2.0;
      default:
        return 1.0;
    }
}

// Manufactured exact solution u(x) = cos(a*pi*x) cos(a*pi*y); used as the
// non-homogeneous Dirichlet boundary data g on the whole boundary.
template <int dim>
double exact_solution(const Point<dim> &pt, DiffusionPattern pattern)
{
  const double a = pattern_freq(pattern);
  const double x = pt[0];
  const double y = (dim > 1) ? pt[1] : 0.0;
  return std::cos(a * numbers::PI * x) * std::cos(a * numbers::PI * y);
}

// Forcing f = -Laplacian(u) = 2(a*pi)^2 cos(a*pi*x) cos(a*pi*y). NOTE: no mu
// factor -- replicates DiffusionModel.hpp RightHandSide exactly so rho lands on
// the same scale as the FEM training data.
template <int dim>
double rhs_forcing(const Point<dim> &pt, DiffusionPattern pattern)
{
  const double a = pattern_freq(pattern);
  const double c = 2.0 * (a * numbers::PI) * (a * numbers::PI);
  return c * exact_solution(pt, pattern);
}

// Coefficient-aware WEIGHTED SIP-DG assembly of -div(mu(x) grad u), robust to
// high coefficient contrast (Ern-Stephansen-Zunino). mu is PER-POLYTOPE
// constant mu_I = pattern_mu(centroid) -- matching the original FEM code (one
// mu per element) -- used in BOTH the volume term and the face coefficients.
// Interior faces use harmonic weighting: the consistency coefficient is
// mu_w = mu_I*mu_J/(mu_I+mu_J) for both sides' gradients (since w_I*mu_I =
// w_J*mu_J = mu_w), and the penalty uses the harmonic mean 2*mu_w. Because
// mu_w <= min(mu_I,mu_J), the consistency term is controlled by the weaker
// side's energy -> provably coercive (SPD). Reduces to the standard scheme
// (0.5 averaging, penalty ~ mu) when mu_I = mu_J. Block formulas adapted from
// examples/minimal_SIP.cc.
template <int dim>
void
assemble_hetero(const AgglomerationHandler<dim> &ah,
                const FE_DGQ<dim>               &fe_dg,
                DiffusionPattern                 pattern,
                double                           epsilon,
                TrilinosWrappers::SparseMatrix  &system_matrix,
                Vector<double>                  &system_rhs)
{
  AffineConstraints<double> constraints;
  constraints.close();

  const unsigned int dpp = fe_dg.n_dofs_per_cell();
  const double penalty_constant = 10.0 * (fe_dg.degree + dim) * (fe_dg.degree + 1);

  FullMatrix<double> cell_matrix(dpp, dpp);
  FullMatrix<double> M11(dpp, dpp), M12(dpp, dpp), M21(dpp, dpp), M22(dpp, dpp);
  Vector<double>     cell_rhs(dpp), b11(dpp);
  std::vector<types::global_dof_index> ldi(dpp), ldi_neighbor(dpp);

  // Per-polytope constant coefficient mu_I = pattern_mu at the bbox center
  // (same value stored in geometry.csv, so the GNN feature is consistent).
  std::vector<double> mu_poly;
  for (const auto &polytope : ah.polytope_iterators())
    {
      const unsigned int idx = polytope->index();
      if (idx >= mu_poly.size())
        mu_poly.resize(idx + 1, 1.0);
      mu_poly[idx] =
        pattern_mu(polytope->get_bounding_box().center(), pattern, epsilon);
    }

  for (const auto &polytope : ah.polytope_iterators())
    {
      const double mu_I = mu_poly[polytope->index()];

      // ---- volume term: mu_I * grad(i).grad(j) * JxW ----
      const auto        &fev = ah.reinit(polytope);
      const unsigned int nq  = fev.n_quadrature_points;

      cell_matrix = 0.;
      cell_rhs    = 0.;
      polytope->get_dof_indices(ldi);
      for (unsigned int q = 0; q < nq; ++q)
        {
          const double fq = rhs_forcing(fev.quadrature_point(q), pattern);
          for (unsigned int i = 0; i < dpp; ++i)
            {
              for (unsigned int j = 0; j < dpp; ++j)
                cell_matrix(i, j) +=
                  mu_I * fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
              // volume RHS: (f, v)
              cell_rhs(i) += fq * fev.shape_value(i, q) * fev.JxW(q);
            }
        }
      constraints.distribute_local_to_global(cell_matrix, ldi, system_matrix);
      constraints.distribute_local_to_global(cell_rhs, ldi, system_rhs);

      // ---- face terms ----
      const double h_I = std::fabs(polytope->diameter());
      for (unsigned int f = 0; f < polytope->n_faces(); ++f)
        {
          if (polytope->at_boundary(f))
            {
              // Homogeneous Dirichlet SIP-DG boundary terms. No averaging on
              // the boundary: consistency factor is 1 (not 0.5), jump [v]=v.
              // These terms make the operator SPD (remove the constant null
              // mode); skipping them leaves A singular/indefinite.
              const auto  &feb     = ah.reinit(polytope, f);
              const auto  &bnormals = feb.get_normal_vectors();
              const unsigned int nqb = feb.n_quadrature_points;
              const double pen = penalty_constant * mu_I / h_I;
              M11 = 0.;
              b11 = 0.;
              for (unsigned int q = 0; q < nqb; ++q)
                {
                  const auto   n    = bnormals[q];
                  const double JxW  = feb.JxW(q);
                  // g = exact solution (non-homogeneous Dirichlet BC value)
                  const double g    =
                    exact_solution(feb.quadrature_point(q), pattern);
                  for (unsigned int i = 0; i < dpp; ++i)
                    {
                      for (unsigned int j = 0; j < dpp; ++j)
                        M11(i, j) +=
                          (-mu_I * (feb.shape_grad(i, q) * n) * feb.shape_value(j, q)
                           - mu_I * (feb.shape_grad(j, q) * n) * feb.shape_value(i, q)
                           + pen * feb.shape_value(i, q) * feb.shape_value(j, q)) * JxW;
                      // Nitsche RHS: -mu_I (grad v . n) g + pen * v * g
                      b11(i) += (-mu_I * (feb.shape_grad(i, q) * n) * g
                                 + pen * feb.shape_value(i, q) * g) * JxW;
                    }
                }
              constraints.distribute_local_to_global(M11, ldi, system_matrix);
              constraints.distribute_local_to_global(b11, ldi, system_rhs);
              continue;
            }

          const auto neigh = polytope->neighbor(f);
          if (polytope->index() >= neigh->index())
            continue; // visit each interior face once

          const double h_J = std::fabs(neigh->diameter());
          const double h_factor = std::max(1.0 / h_I, 1.0 / h_J);

          // Harmonic (Ern-Stephansen-Zunino) weighting for the coefficient.
          // With w_I = mu_J/(mu_I+mu_J), w_J = mu_I/(mu_I+mu_J) we get
          // w_I*mu_I = w_J*mu_J = mu_w, so both sides' fluxes use mu_w and the
          // penalty scales with the harmonic mean 2*mu_w. This restores
          // coercivity at high contrast (mu_w <= min(mu_I, mu_J)).
          const double mu_J = mu_poly[neigh->index()];
          const double mu_w = (mu_I * mu_J) / (mu_I + mu_J);
          const double pen  = penalty_constant * (2.0 * mu_w) * h_factor;

          const unsigned int nofn =
            polytope->neighbor_of_agglomerated_neighbor(f);
          const auto fe_faces = ah.reinit_interface(polytope, neigh, f, nofn);
          const auto &fe0 = fe_faces.first;   // this side
          const auto &fe1 = fe_faces.second;  // neighbor side
          const auto &normals = fe0.get_normal_vectors();

          M11 = 0.; M12 = 0.; M21 = 0.; M22 = 0.;
          const unsigned int nqf = fe0.n_quadrature_points;
          for (unsigned int q = 0; q < nqf; ++q)
            {
              const auto   n    = normals[q];
              const double JxW  = fe0.JxW(q);
              for (unsigned int i = 0; i < dpp; ++i)
                for (unsigned int j = 0; j < dpp; ++j)
                  {
                    M11(i, j) += (-mu_w * (fe0.shape_grad(i, q) * n) * fe0.shape_value(j, q)
                                  - mu_w * (fe0.shape_grad(j, q) * n) * fe0.shape_value(i, q)
                                  + pen * fe0.shape_value(i, q) * fe0.shape_value(j, q)) * JxW;
                    M12(i, j) += (+mu_w * (fe0.shape_grad(i, q) * n) * fe1.shape_value(j, q)
                                  - mu_w * (fe1.shape_grad(j, q) * n) * fe0.shape_value(i, q)
                                  - pen * fe0.shape_value(i, q) * fe1.shape_value(j, q)) * JxW;
                    M21(i, j) += (-mu_w * (fe1.shape_grad(i, q) * n) * fe0.shape_value(j, q)
                                  + mu_w * (fe0.shape_grad(j, q) * n) * fe1.shape_value(i, q)
                                  - pen * fe1.shape_value(i, q) * fe0.shape_value(j, q)) * JxW;
                    M22(i, j) += (+mu_w * (fe1.shape_grad(i, q) * n) * fe1.shape_value(j, q)
                                  + mu_w * (fe1.shape_grad(j, q) * n) * fe1.shape_value(i, q)
                                  + pen * fe1.shape_value(i, q) * fe1.shape_value(j, q)) * JxW;
                  }
            }
          neigh->get_dof_indices(ldi_neighbor);
          constraints.distribute_local_to_global(M11, ldi, system_matrix);
          constraints.distribute_local_to_global(M12, ldi, ldi_neighbor, system_matrix);
          constraints.distribute_local_to_global(M21, ldi_neighbor, ldi, system_matrix);
          constraints.distribute_local_to_global(M22, ldi_neighbor, system_matrix);
        }
    }
  system_matrix.compress(VectorOperation::add);
}
template <int dim>
void
generate(unsigned int n_subdomains, unsigned int fe_degree,
         unsigned int n_global_refinements, DiffusionPattern pattern,
         double epsilon, const std::string &out_prefix)
{
  // Domain [-1,1]^2 to match the paper's pattern boundaries.
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1.0, 1.0);
  tria.refine_global(n_global_refinements);
  std::cout << "[step 1] fine mesh: " << tria.n_active_cells() << " cells\n";
  if (tria.n_active_cells() < 4 * n_subdomains)
    {
      std::cerr << "[error] too few cells for " << n_subdomains
                << " agglomerates\n";
      return;
    }

  MappingQ<dim>         mapping(1);
  GridTools::Cache<dim> cached_tria(tria, mapping);
  GridTools::partition_triangulation(n_subdomains, tria,
                                     SparsityTools::Partitioner::metis);

  AgglomerationHandler<dim> ah(cached_tria);
  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain(n_subdomains);
  for (const auto &cell : tria.active_cell_iterators())
    cells_per_subdomain[cell->subdomain_id()].push_back(cell);
  unsigned int n_defined = 0;
  for (unsigned int i = 0; i < n_subdomains; ++i)
    if (!cells_per_subdomain[i].empty())
      { ah.define_agglomerate(cells_per_subdomain[i]); ++n_defined; }
  std::cout << "[step 3] defined " << n_defined << " agglomerates\n";

  FE_DGQ<dim> fe_dg(fe_degree);
  ah.distribute_agglomerated_dofs(fe_dg);

  const QGauss<dim>     cell_quad(fe_degree + 1);
  const QGauss<dim - 1> face_quad(fe_degree + 1);
  const UpdateFlags cell_flags =
    update_values | update_gradients | update_JxW_values | update_quadrature_points;
  const UpdateFlags face_flags = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points |
                                 update_normal_vectors;
  ah.initialize_fe_values(cell_quad, cell_flags, face_quad, face_flags);
  std::cout << "[step 4] n_dofs=" << ah.n_dofs()
            << " pattern=" << pattern_name(pattern)
            << " epsilon=" << epsilon << "\n";

  // Sparsity pattern + Trilinos matrix (assemble_hetero fills it).
  DynamicSparsityPattern dsp;
  ah.create_agglomeration_sparsity_pattern(dsp);
  const IndexSet all_dofs = complete_index_set(ah.n_dofs());
  TrilinosWrappers::SparseMatrix system_matrix;
  system_matrix.reinit(all_dofs, all_dofs, dsp, MPI_COMM_WORLD);
  Vector<double> system_rhs(ah.n_dofs());

  assemble_hetero(ah, fe_dg, pattern, epsilon, system_matrix, system_rhs);
  std::cout << "[step 5] assembled " << system_matrix.m() << "x"
            << system_matrix.n() << " nnz=" << system_matrix.n_nonzero_elements()
            << " |rhs|=" << system_rhs.l2_norm() << "\n";

  PolyUtils::write_to_matrix_market_format(out_prefix + "_matrix.mtx",
                                           "polydg_diffusion_hetero",
                                           system_matrix);
  // RHS as a MatrixMarket dense array (1-column general), matching what a
  // PETSc/Trilinos reader expects for the paired b vector.
  {
    std::ofstream rhs_out(out_prefix + "_rhs.mtx");
    rhs_out.precision(16);
    rhs_out << "%%MatrixMarket matrix array real general\n";
    rhs_out << system_rhs.size() << " 1\n";
    for (unsigned int i = 0; i < system_rhs.size(); ++i)
      rhs_out << system_rhs[i] << "\n";
  }
  std::cout << "[step 6] wrote " << out_prefix << "_matrix.mtx and _rhs.mtx\n";

  // Geometry + representative mu (evaluated at each polytope bbox center).
  std::ofstream g(out_prefix + "_geometry.csv");
  g << "polytope_index,center_x,center_y,diameter,volume,n_faces,"
       "n_background_cells,mu\n";
  g.precision(16);
  for (const auto &polytope : ah.polytope_iterators())
    {
      const auto       bbox = polytope->get_bounding_box();
      const Point<dim> c    = bbox.center();
      g << polytope->index() << ",";
      for (unsigned int d = 0; d < dim; ++d)
        g << c[d] << ",";
      g << polytope->diameter() << "," << polytope->volume() << ","
        << polytope->n_faces() << "," << polytope->n_background_cells() << ","
        << pattern_mu(c, pattern, epsilon) << "\n";
    }
  std::cout << "[done] n_sub=" << n_subdomains << " p=" << fe_degree
            << " pattern=" << pattern_name(pattern) << " eps=" << epsilon << "\n";
}
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int n_sub   = (argc > 1) ? std::stoul(argv[1]) : 100;
  const unsigned int degree  = (argc > 2) ? std::stoul(argv[2]) : 1;
  const unsigned int refine  = (argc > 3) ? std::stoul(argv[3]) : 6;
  const unsigned int pat_id  = (argc > 4) ? std::stoul(argv[4]) : 0;
  const double       epsilon = (argc > 5) ? std::stod(argv[5]) : 4.0;
  const std::string  prefix  = (argc > 6) ? argv[6] : "polydg_Dhet";

  const DiffusionPattern pattern = static_cast<DiffusionPattern>(pat_id);
  generate<2>(n_sub, degree, refine, pattern, epsilon, prefix);
  return 0;
}
