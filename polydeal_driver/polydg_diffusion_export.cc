/* ------------------------------------------------------------------
 * Phase 0d minimal driver: PolyDG diffusion matrix -> Matrix Market.
 *
 * Builds a fine Cartesian background mesh, agglomerates it into
 * `n_subdomains` polygons via METIS, assembles the SIP-DG Laplace
 * (diffusion) matrix at polynomial degree p on the agglomerated
 * polytopic mesh using Polydeal's PolyUtils::assemble_dg_matrix,
 * and exports:
 *    <prefix>_matrix.mtx      the raw system matrix (MatrixMarket, 1-based)
 *    <prefix>_geometry.csv    per-polytope geometry (for later GNN features)
 *
 * Uses TrilinosWrappers::SparseMatrix: PolyUtils::assemble_dg_matrix builds
 * a Trilinos sparsity pattern and reinits the matrix internally, so the
 * matrix type must be Trilinos (a serial SparseMatrix<double> does not
 * compile). No C/F splitting yet (export path 2).
 * Drop this file into Polydeal/examples/ and rebuild to compile it.
 * ------------------------------------------------------------------ */

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <agglomeration_handler.h>
#include <poly_utils.h>

#include <fstream>
#include <string>

using namespace dealii;


template <int dim>
void
generate(const unsigned int n_subdomains,
         const unsigned int fe_degree,
         const unsigned int n_global_refinements,
         const std::string &out_prefix)
{
  // 1. Fine Cartesian background mesh on the unit square/cube.
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1.);
  tria.refine_global(n_global_refinements);
  std::cout << "[step 1] fine mesh: " << tria.n_active_cells()
            << " active cells" << std::endl;

  // Guard against over-partitioning: METIS leaves empty subdomains (and
  // define_agglomerate then crashes) when n_subdomains is close to the
  // cell count. Require a comfortable margin.
  if (tria.n_active_cells() < 4 * n_subdomains)
    {
      std::cerr << "[error] too few cells (" << tria.n_active_cells()
                << ") for " << n_subdomains
                << " agglomerates; increase refinements or reduce "
                   "n_subdomains (want >= 4x cells)."
                << std::endl;
      return;
    }

  MappingQ<dim>         mapping(1);
  GridTools::Cache<dim> cached_tria(tria, mapping);

  // 2. METIS partition of the fine cells -> one agglomerate per subdomain.
  GridTools::partition_triangulation(n_subdomains,
                                     tria,
                                     SparsityTools::Partitioner::metis);
  std::cout << "[step 2] METIS partition into " << n_subdomains
            << " subdomains" << std::endl;

  // 3. Build the agglomerated polytopic mesh. Skip any empty subdomain id
  //    (METIS does not guarantee every id is used).
  AgglomerationHandler<dim> ah(cached_tria);

  std::vector<std::vector<typename Triangulation<dim>::active_cell_iterator>>
    cells_per_subdomain(n_subdomains);
  for (const auto &cell : tria.active_cell_iterators())
    cells_per_subdomain[cell->subdomain_id()].push_back(cell);

  unsigned int n_defined = 0;
  for (unsigned int i = 0; i < n_subdomains; ++i)
    if (!cells_per_subdomain[i].empty())
      {
        ah.define_agglomerate(cells_per_subdomain[i]);
        ++n_defined;
      }
  std::cout << "[step 3] defined " << n_defined << " agglomerates (of "
            << n_subdomains << " requested)" << std::endl;

  // 4. Distribute DG(p) dofs over the polytopes.
  FE_DGQ<dim> fe_dg(fe_degree);
  ah.distribute_agglomerated_dofs(fe_dg);
  std::cout << "[step 4] distributed dofs: n_dofs=" << ah.n_dofs()
            << std::endl;

  // 4b. Initialize the handler's FEValues. assemble_dg_matrix calls
  //     ah.reinit(...) internally but does NOT initialize FEValues itself,
  //     so this MUST be done first or reinit segfaults. Volume needs
  //     gradients + JxW; faces need values, gradients, normals, and
  //     quadrature points for the SIP jump/average and penalty terms.
  const QGauss<dim>     cell_quad(fe_degree + 1);
  const QGauss<dim - 1> face_quad(fe_degree + 1);
  const UpdateFlags     cell_flags =
    update_gradients | update_JxW_values | update_quadrature_points;
  const UpdateFlags face_flags = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points |
                                 update_normal_vectors;
  ah.initialize_fe_values(cell_quad, cell_flags, face_quad, face_flags);
  std::cout << "[step 4b] initialized FEValues" << std::endl;

  // 5. Assemble the SIP-DG diffusion matrix. assemble_dg_matrix builds the
  //    Trilinos sparsity pattern and reinits system_matrix internally, and
  //    computes the penalty itself; no manual setup needed here.
  TrilinosWrappers::SparseMatrix system_matrix;
  PolyUtils::assemble_dg_matrix(system_matrix, fe_dg, ah);
  std::cout << "[step 5] assembled matrix: " << system_matrix.m() << "x"
            << system_matrix.n() << ", nnz=" << system_matrix.n_nonzero_elements()
            << std::endl;

  // 6. Export the raw matrix (MatrixMarket, readable via scipy.io.mmread).
  PolyUtils::write_to_matrix_market_format(out_prefix + "_matrix.mtx",
                                           "polydg_diffusion",
                                           system_matrix);
  std::cout << "[step 6] wrote " << out_prefix << "_matrix.mtx" << std::endl;

  // 7. Export per-polytope geometry (bbox center, diameter, volume, ...).
  std::ofstream g(out_prefix + "_geometry.csv");
  g << "polytope_index,center_x,center_y,diameter,volume,n_faces,"
       "n_background_cells\n";
  g.precision(16);
  for (const auto &polytope : ah.polytope_iterators())
    {
      const auto  bbox = polytope->get_bounding_box();
      const Point<dim> c = bbox.center();
      g << polytope->index() << ",";
      for (unsigned int d = 0; d < dim; ++d)
        g << c[d] << ",";
      g << polytope->diameter() << "," << polytope->volume() << ","
        << polytope->n_faces() << "," << polytope->n_background_cells()
        << "\n";
    }

  std::cout << "[polydg] n_subdomains=" << n_subdomains
            << " p=" << fe_degree << " n_dofs=" << ah.n_dofs()
            << " -> " << out_prefix << "_matrix.mtx\n";
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // Defaults; override on the command line:
  //   ./polydg_diffusion_export [n_subdomains] [degree] [refinements] [prefix]
  const unsigned int n_subdomains         = (argc > 1) ? std::stoul(argv[1]) : 100;
  const unsigned int fe_degree            = (argc > 2) ? std::stoul(argv[2]) : 1;
  const unsigned int n_global_refinements = (argc > 3) ? std::stoul(argv[3]) : 4;
  const std::string  out_prefix           = (argc > 4) ? argv[4] : "polydg_D";

  generate<2>(n_subdomains, fe_degree, n_global_refinements, out_prefix);

  return 0;
}
