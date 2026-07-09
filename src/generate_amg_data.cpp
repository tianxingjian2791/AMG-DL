/* ---------------------------------------------------------------------
 * generate_amg_data.cpp
 *
 * Unified AMG dataset generator
 *
 * Supports all problem types:
 * - D: Diffusion equations (FEM)
 * - E: Elastic equations (FEM)
 * - S: Stokes equations (FEM)
 * - GL: Graph Laplacian (random graphs)
 * - SC: Spectral Clustering (k-NN graphs)
 *
 * Output formats:
 * - theta-cnn: Pooled 50×50 matrices for CNN
 * - theta-gnn: Sparse CSR graphs for GNN
 * - p-value: Full AMG operators (C/F, P, S matrices)
 * - all: Generate all three formats
 *
 * Usage:
 *   ./generate_amg_data -p <type> -s <split> -f <format> -c <scale> [options]
 * ---------------------------------------------------------------------
 */

#include "../include/DiffusionModel.hpp"
#include "../include/ElasticModel.hpp"
#ifdef DEAL_II_WITH_P4EST
#include "../include/StokesModel.hpp"
#endif
#include "../include/GraphLaplacianModelEigen.hpp"
#include "../include/AMGOperators.hpp"
#include "../include/UnifiedDataGenerator.hpp"
#include "../include/NPZWriter.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>
#include <omp.h>

// Configuration Enums and Structures
enum class ProblemType {
    DIFFUSION,
    ELASTIC,
    STOKES,
    GRAPH_LAPLACIAN,
    SPECTRAL_CLUSTERING
};

enum class DatasetSplit {
    TRAIN,
    TEST,
    ALL
};

enum class OutputFormat {
    THETA_CNN,
    THETA_GNN,
    P_VALUE,
    ALL
};

enum class DatasetScale {
    SMALL = 0,
    LARGE = 1,
    PAPER = 2,
};

// Mesh cell type for FEM problems (currently only honored by Diffusion).
enum class CellType {
    QUAD,
    SIMPLEX
};

struct CommandLineArgs {
    ProblemType problem;
    DatasetSplit split = DatasetSplit::ALL;
    OutputFormat format;
    DatasetScale scale;
    std::string output_dir = "./datasets";
    int num_threads = 0;  // 0 = auto (OpenMP default)
    int seed = 42;
    bool verbose = false;
    bool use_npy = false;  // Use CSV as default, can switch to NPZ for better performance
    CellType cell_type = CellType::QUAD;  // Diffusion-only: quad (default) or simplex
};


// Scale Configuration Structures
struct FEMScaleConfig {
    std::vector<double> param1_values;  // epsilon for D, E for Elastic, viscosity for S
    std::vector<double> param2_values;  // (optional) nu for E, velocity_degree for S
    std::vector<double> theta_values;
    std::vector<unsigned int> refinements;

    int total_samples() const {
        int count = param1_values.size() * theta_values.size() * refinements.size();
        if (!param2_values.empty()) {
            count *= param2_values.size();
        }
        return count;
    }
};

struct DiffusionExperimentRecord {
    std::string scale_name;
    int scale_id;
    std::string pattern_name;
    int pattern_id;
    double epsilon;
    unsigned int refinement;
    double h;
    double theta;
    double rho;
    unsigned int iterations;
    double elapsed_sec;
    int n_levels;
    int n;
    int nnz;
};

struct GraphScaleConfig {
    int num_samples;
    int num_points;  // Nodes per graph
};

// Utility Functions
std::vector<double> linspace(double start, double end, size_t num_points) {
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

void create_directories(const std::string& path) {
    std::string command = "mkdir -p " + path;
    system(command.c_str());
}

std::string problem_type_to_string(ProblemType type) {
    switch (type) {
        case ProblemType::DIFFUSION: return "D";
        case ProblemType::ELASTIC: return "E";
        case ProblemType::STOKES: return "S";
        case ProblemType::GRAPH_LAPLACIAN: return "GL";
        case ProblemType::SPECTRAL_CLUSTERING: return "SC";
        default: return "UNKNOWN";
    }
}

std::string scale_to_string(DatasetScale scale) {
    switch (scale) {
        case DatasetScale::SMALL: return "small";
        case DatasetScale::LARGE: return "large";
        case DatasetScale::PAPER: return "paper";
        default: return "unknown";
    }
}

std::string split_to_string(DatasetSplit split) {
    switch (split) {
        case DatasetSplit::TRAIN: return "train";
        case DatasetSplit::TEST: return "test";
        case DatasetSplit::ALL: return "all";
        default: return "unknown";
    }
}

std::string cell_type_to_string(CellType cell_type) {
    switch (cell_type) {
        case CellType::QUAD: return "quad";
        case CellType::SIMPLEX: return "simplex";
        default: return "unknown";
    }
}

// Configuration Factory
class ConfigFactory {
public:
    static FEMScaleConfig get_diffusion_config(DatasetScale scale, DatasetSplit split) {
        FEMScaleConfig config;
        
        const std::vector<double> three_theta_values = {0.24, 0.48, 0.72};
        const std::vector<double> ten_theta_values = linspace(0.02, 0.9, 10);
        const std::vector<double> paper_theta_values = linspace(0.02, 0.9, 25);
        const std::vector<double> paper_epsilon_values = {0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.5, 5.0, 7.0, 9.5};

        switch (scale) {
            case DatasetScale::SMALL:
                config.param1_values = {0.0, 0.8, 1.6, 2.4, 5.0, 9.5}; // epsilon subset from paper grid
                config.theta_values = three_theta_values;
                config.refinements = {4, 5, 6};                     // h = 1.25e-01 ... 3.12e-02
                // Base total per pattern: 6 × 3 × 3 = 54 samples
                break;

            case DatasetScale::LARGE:
                config.param1_values = paper_epsilon_values;
                config.theta_values = ten_theta_values;             // 10 theta values for denser scatter data
                config.refinements = {4, 5, 6, 7, 8, 9, 10, 11};    // h = 1.25e-01 ... 9.77e-04
                // Base total per pattern: 12 × 10 × 8 = 960 samples
                break;

            case DatasetScale::PAPER:
                config.param1_values = paper_epsilon_values;        // full paper epsilon grid
                config.theta_values = paper_theta_values;           // full paper theta grid
                config.refinements = {4, 5, 6, 7, 8, 9, 10, 11};    // h = 1.25e-01 ... 9.77e-04
                // Base total per pattern: 12 × 25 × 8 = 2,400 samples
                break;
        }

        return config;
    }

    static FEMScaleConfig get_elastic_config(DatasetScale scale, DatasetSplit split) {
        FEMScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.param1_values = {2.5e2, 2.5e4, 2.5e6};      // E: 3 values
                config.param2_values = {0.25, 0.35};                // nu: 2 values
                config.theta_values = linspace(0.1, 0.8, 5);        // theta: 5 values
                config.refinements = {3, 4};                         // 2 levels
                // Total: 3 × 2 × 5 × 2 = 60 samples
                break;

            case DatasetScale::LARGE:
                config.param1_values = {2.5, 2.5e2, 2.5e4}; // 3 values
                config.param2_values = {0.20, 0.25, 0.30, 0.35, 0.40};    // 5 values
                config.theta_values = linspace(0.02, 0.9, 50);      // 50 values
                config.refinements = {3, 4};                   // 2 levels
                // Total: 3 × 5 × 50 × 2 = 1,500 samples (1000 for training, 500 for testing)
                break;

            case DatasetScale::PAPER:
                config.param1_values = {2.5, 2.5e2, 2.5e4, 2.5e6, 2.5e8}; // 5 values
                config.param2_values = linspace(0.15, 0.45, 7);     // nu: 7 values
                config.theta_values = linspace(0.02, 0.9, 32);      // theta: 32 values
                config.refinements = {3, 4, 5, 6};                   // 4 levels
                // Total: 5 × 7 × 32 × 4 = 4,480 samples
                break;
        }

        return config;
    }

    static FEMScaleConfig get_stokes_config(DatasetScale scale, DatasetSplit split) {
        FEMScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.param1_values = linspace(0.1, 6.1, 6);       // viscosity: 6 values
                config.param2_values = {2};                          // velocity_degree: 1 value
                config.theta_values = linspace(0.1, 0.8, 5);         // theta: 5 values
                config.refinements = {3, 4};                          // 2 levels
                // Total: 6 × 1 × 5 × 2 = 60 samples
                break;

            case DatasetScale::LARGE:
                config.param1_values = linspace(0.1, 6.1, 18);      // viscosity: 18 values
                config.param2_values = {2};                       // velocity_degree: 1 values
                config.theta_values = linspace(0.02, 0.9, 50);       // theta: 50 values
                config.refinements = {3, 4};                    // 2 levels
                // Total: 18 × 1 × 50 × 2 = 1,800 samples (1200 for training, 600 for testing)
                break;

            case DatasetScale::PAPER:
                config.param1_values = linspace(0.1, 6.1, 20);      // viscosity: 20 values
                config.param2_values = {2, 3, 4};                    // velocity_degree: 3 values
                config.theta_values = linspace(0.02, 0.9, 32);       // theta: 32 values
                config.refinements = {3, 4, 5, 6};                    // 4 levels
                // Total: 20 × 3 × 32 × 4 = 7,680 samples
                break;
        }

        return config;
    }

    static GraphScaleConfig get_graph_laplacian_config(DatasetScale scale, DatasetSplit split) {
        GraphScaleConfig config;

        switch (scale) {
            case DatasetScale::SMALL:
                config.num_samples = 50;
                config.num_points = 64;
                break;

            case DatasetScale::LARGE:
                config.num_samples = 5000;
                config.num_points = 64;  // Can be set to 256 for having rich computing resources
                break;

            case DatasetScale::PAPER:
                config.num_samples = 10000;
                config.num_points = 512;
                break;
        }

        return config;
    }

    static GraphScaleConfig get_spectral_clustering_config(DatasetScale scale, DatasetSplit split) {
        return get_graph_laplacian_config(scale, split);
    }
};

// Unified AMG Data Generator
class UnifiedAMGDataGenerator {
public:
    explicit UnifiedAMGDataGenerator(const CommandLineArgs& args)
        : args_(args) {
        output_base_ = args_.output_dir + "/" + split_to_string(args_.split) + "/";
    }

    void generate() {
        switch (args_.problem) {
            case ProblemType::DIFFUSION:
                generate_diffusion();
                break;
            case ProblemType::ELASTIC:
                generate_elastic();
                break;
            case ProblemType::STOKES:
#ifdef DEAL_II_WITH_P4EST
                generate_stokes();
#else
                throw std::runtime_error(
                    "Stokes generation requires a deal.II build with p4est support "
                    "(DEAL_II_WITH_P4EST). Use a p4est-enabled deal.II install or choose another problem type."
                );
#endif
                break;
            case ProblemType::GRAPH_LAPLACIAN:
                generate_graph_laplacian();
                break;
            case ProblemType::SPECTRAL_CLUSTERING:
                generate_spectral_clustering();
                break;
            default:
                throw std::runtime_error("Unknown problem type");
        }
    }

private:
    void generate_diffusion();
    void generate_elastic();
    void generate_stokes();
    void generate_graph_laplacian();
    void generate_spectral_clustering();

    void open_output_files(std::ofstream& theta_cnn_file,
                          std::ofstream& theta_gnn_file,
                          std::ofstream& p_value_file);

    void close_output_files(std::ofstream& theta_cnn_file,
                           std::ofstream& theta_gnn_file,
                           std::ofstream& p_value_file);

    void write_sample_theta_cnn(const AMGOperators::CSRMatrix& A,
                               double h, double theta, double rho,
                               std::ofstream& file);

    void write_sample_theta_gnn(const AMGOperators::CSRMatrix& A,
                               double h, double theta, double rho,
                               std::ofstream& file);

    void write_sample_p_value(const AMGOperators::CSRMatrix& A,
                             double h, double theta, double rho,
                             std::ofstream& file);

    void write_sample(const AMGOperators::CSRMatrix& A,
                     double h, double theta, double rho,
                     std::ofstream& theta_cnn_file,
                     std::ofstream& theta_gnn_file,
                     std::ofstream& p_value_file);

    // NPZ writing functions (high performance binary format)
    void write_sample_npz_theta_gnn(const AMGOperators::CSRMatrix& A,
                                    double h, double theta, double rho,
                                    const std::string& output_dir,
                                    int sample_id,
                                    int pattern_id = -1,
                                    double epsilon = 0.0,
                                    unsigned int refinement = 0,
                                    unsigned int iterations = 0);

    void write_sample_npz_p_value(const AMGOperators::CSRMatrix& A,
                                 double h, double theta, double rho,
                                 const std::string& output_dir,
                                 int sample_id,
                                 int pattern_id = -1,
                                 double epsilon = 0.0,
                                 unsigned int refinement = 0,
                                 unsigned int iterations = 0);

    void write_sample_npz_theta_cnn(const AMGOperators::CSRMatrix& A,
                                   double h, double theta, double rho,
                                   const std::string& output_dir,
                                   int sample_id,
                                   int pattern_id = -1,
                                   double epsilon = 0.0,
                                   unsigned int refinement = 0,
                                   unsigned int iterations = 0);

    void append_diffusion_report(const std::string& report_path,
                                 const DiffusionExperimentRecord& record,
                                 bool write_header);

    void print_progress(int current, int total,
                       std::chrono::steady_clock::time_point start);

    void print_final_summary(int total,
                            std::chrono::steady_clock::time_point start);

    CommandLineArgs args_;
    std::string output_base_;
};

void UnifiedAMGDataGenerator::open_output_files(
    std::ofstream& theta_cnn_file,
    std::ofstream& theta_gnn_file,
    std::ofstream& p_value_file)
{
    std::string problem_suffix = problem_type_to_string(args_.problem);
    std::string split_prefix = split_to_string(args_.split);

    if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
        std::string path = output_base_ + "theta_cnn/";
        create_directories(path);
        path += split_prefix + "_" + problem_suffix + ".csv";
        theta_cnn_file.open(path);
        if (!theta_cnn_file) {
            throw std::runtime_error("Failed to open theta_cnn output file: " + path);
        }
    }

    if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
        std::string path = output_base_ + "theta_gnn/";
        create_directories(path);
        path += split_prefix + "_" + problem_suffix + ".csv";
        theta_gnn_file.open(path);
        if (!theta_gnn_file) {
            throw std::runtime_error("Failed to open theta_gnn output file: " + path);
        }
    }

    if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
        std::string path = output_base_ + "p_value/";
        create_directories(path);
        path += split_prefix + "_" + problem_suffix + ".csv";
        p_value_file.open(path);
        if (!p_value_file) {
            throw std::runtime_error("Failed to open p_value output file: " + path);
        }
    }
}

void UnifiedAMGDataGenerator::close_output_files(
    std::ofstream& theta_cnn_file,
    std::ofstream& theta_gnn_file,
    std::ofstream& p_value_file)
{
    if (theta_cnn_file.is_open()) theta_cnn_file.close();
    if (theta_gnn_file.is_open()) theta_gnn_file.close();
    if (p_value_file.is_open()) p_value_file.close();
}

void UnifiedAMGDataGenerator::append_diffusion_report(
    const std::string& report_path,
    const DiffusionExperimentRecord& record,
    bool write_header)
{
    std::ofstream report(report_path, std::ios::out | std::ios::app);
    if (!report) {
        throw std::runtime_error("Failed to open diffusion report file: " + report_path);
    }

    if (write_header) {
        report << "scale,scale_id,pattern,pattern_id,epsilon,refinement,h,theta,rho,iterations,elapsed_sec,n_levels,n,nnz\n";
    }

    report << record.scale_name << ","
           << record.scale_id << ","
           << record.pattern_name << ","
           << record.pattern_id << ","
           << record.epsilon << ","
           << record.refinement << ","
           << record.h << ","
           << record.theta << ","
           << record.rho << ","
           << record.iterations << ","
           << record.elapsed_sec << ","
           << record.n_levels << ","
           << record.n << ","
           << record.nnz << "\n";
}

void UnifiedAMGDataGenerator::write_sample_theta_cnn(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& file)
{
    // Pool matrix to 50x50 using pooling function from Pooling.hpp
    std::vector<std::vector<double>> V;
    std::vector<std::vector<int>> C;
    int pool_size = 50;

    parallel_pooling_csr(A.values, A.col_indices, A.row_ptr,
                        A.n_rows, pool_size, PoolingOp::SUM, V, C);

    // Standardize
    std_normalize(V);

    // Write: n, rho, h, theta, pooled_values (2500 values)
    file << A.n_rows << "," << rho << "," << h << "," << theta;

    for (int i = 0; i < pool_size; ++i) {
        for (int j = 0; j < pool_size; ++j) {
            file << "," << V[i][j];
        }
    }

    file << "\n";
}

void UnifiedAMGDataGenerator::write_sample_theta_gnn(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& file)
{
    int n = A.n_rows;
    int nnz = A.values.size();

    // Write: n, rho, h, theta, nnz, values, row_ptr, col_indices
    file << n << "," << rho << "," << h << "," << theta << "," << nnz;

    // Write values
    for (const auto& val : A.values) {
        file << "," << val;
    }

    // Write row pointers
    for (const auto& r : A.row_ptr) {
        file << "," << r;
    }

    // Write column indices
    for (const auto& c : A.col_indices) {
        file << "," << c;
    }

    file << "\n";
}

void UnifiedAMGDataGenerator::write_sample_p_value(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& file)
{
    // Compute AMG operators
    std::vector<int> cf_splitting = AMGOperators::classical_cf_splitting(A, theta);
    AMGOperators::CSRMatrix S = AMGOperators::compute_strength_matrix(A, theta);
    AMGOperators::CSRMatrix P = AMGOperators::compute_baseline_prolongation(A, cf_splitting);

    int n = A.n_rows;
    int nnz_A = A.values.size();
    int nnz_S = S.values.size();
    int nnz_P = P.values.size();

    // Write header: n, rho, h, theta, nnz_A, nnz_S, nnz_P
    file << n << "," << rho << "," << h << "," << theta << ","
         << nnz_A << "," << nnz_S << "," << nnz_P;

    // Write A matrix (values, row_ptr, col_indices)
    for (const auto& val : A.values) file << "," << val;
    for (const auto& r : A.row_ptr) file << "," << r;
    for (const auto& c : A.col_indices) file << "," << c;

    // Write S matrix
    for (const auto& val : S.values) file << "," << val;
    for (const auto& r : S.row_ptr) file << "," << r;
    for (const auto& c : S.col_indices) file << "," << c;

    // Write P matrix
    for (const auto& val : P.values) file << "," << val;
    for (const auto& r : P.row_ptr) file << "," << r;
    for (const auto& c : P.col_indices) file << "," << c;

    // Write C/F splitting
    for (const auto& cf : cf_splitting) file << "," << cf;

    file << "\n";
}

void UnifiedAMGDataGenerator::write_sample(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    std::ofstream& theta_cnn_file,
    std::ofstream& theta_gnn_file,
    std::ofstream& p_value_file)
{
    if (theta_cnn_file.is_open()) {
        write_sample_theta_cnn(A, h, theta, rho, theta_cnn_file);
    }

    if (theta_gnn_file.is_open()) {
        write_sample_theta_gnn(A, h, theta, rho, theta_gnn_file);
    }

    if (p_value_file.is_open()) {
        write_sample_p_value(A, h, theta, rho, p_value_file);
    }
}

// NPZ Writing Functions (Binary Format)
void UnifiedAMGDataGenerator::write_sample_npz_theta_gnn(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    const std::string& output_dir,
    int sample_id,
    int pattern_id,
    double epsilon,
    unsigned int refinement,
    unsigned int iterations)
{
    // Create filename: sample_00000.npz
    std::ostringstream filename;
    filename << output_dir << "/sample_" << std::setfill('0') << std::setw(5) << sample_id << ".npz";

    // Convert CSR to COO (edge_index format for PyTorch Geometric)
    std::vector<int> edge_src, edge_dst;
    std::vector<double> edge_attr;

    for (int row = 0; row < (int)A.n_rows; ++row) {
        for (int idx = A.row_ptr[row]; idx < A.row_ptr[row + 1]; ++idx) {
            int col = A.col_indices[idx];
            double val = A.values[idx];

            if (row != col) {  // Skip diagonal for GNN (treat as undirected graph)
                edge_src.push_back(row);
                edge_dst.push_back(col);
                edge_attr.push_back(std::abs(val));
            }
        }
    }

    // Create edge_index as 2D array (2, num_edges) - convert to double for NPZ
    std::vector<double> edge_index_flat;
    edge_index_flat.reserve(edge_src.size() * 2);
    for (size_t i = 0; i < edge_src.size(); ++i) {
        edge_index_flat.push_back(static_cast<double>(edge_src[i]));
    }
    for (size_t i = 0; i < edge_dst.size(); ++i) {
        edge_index_flat.push_back(static_cast<double>(edge_dst[i]));
    }

    // Create metadata array: [n, rho, h, epsilon, pattern_id, refinement, iterations]
    std::vector<double> metadata = {
        static_cast<double>(A.n_rows),
        rho,
        h,
        epsilon,
        static_cast<double>(pattern_id),
        static_cast<double>(refinement),
        static_cast<double>(iterations)
    };

    std::vector<double> theta_arr = {theta};
    std::vector<double> y_arr = {rho};  // y = rho for regression

    // Write NPZ file
    NPZWriter::begin(filename.str());

    if (!edge_src.empty()) {
        NPZWriter::add_array_2d("edge_index", edge_index_flat, 2, edge_src.size());
    } else {
        // Empty graph - write empty arrays
        std::vector<double> empty_double;
        NPZWriter::add_array_2d("edge_index", empty_double, 2, 0);
    }

    NPZWriter::add_array("edge_attr", edge_attr);
    NPZWriter::add_array("theta", theta_arr);
    NPZWriter::add_array("y", y_arr);
    NPZWriter::add_array("metadata", metadata);
    NPZWriter::finalize();
}

void UnifiedAMGDataGenerator::write_sample_npz_p_value(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    const std::string& output_dir,
    int sample_id,
    int pattern_id,
    double epsilon,
    unsigned int refinement,
    unsigned int iterations)
{
    // Compute AMG operators
    std::vector<int> cf_splitting = AMGOperators::classical_cf_splitting(A, theta);
    AMGOperators::CSRMatrix S = AMGOperators::compute_strength_matrix(A, theta);
    AMGOperators::CSRMatrix P = AMGOperators::compute_baseline_prolongation(A, cf_splitting);

    // Extract coarse nodes
    std::vector<int> coarse_nodes;
    for (int i = 0; i < (int)cf_splitting.size(); ++i) {
        if (cf_splitting[i] == 1) {  // 1 = coarse point
            coarse_nodes.push_back(i);
        }
    }

    // Create metadata: [n, theta, rho, h, pattern_id, epsilon, refinement, iterations]
    std::vector<double> metadata = {
        static_cast<double>(A.n_rows),
        theta,
        rho,
        h,
        static_cast<double>(pattern_id),
        epsilon,
        static_cast<double>(refinement),
        static_cast<double>(iterations)
    };

    // Create filename
    std::ostringstream filename;
    filename << output_dir << "/sample_" << std::setfill('0') << std::setw(5) << sample_id << ".npz";

    // Write NPZ file
    NPZWriter::begin(filename.str());

    // A matrix
    NPZWriter::add_array("A_values", A.values);
    NPZWriter::add_array("A_row_ptr", reinterpret_cast<const std::vector<int>&>(A.row_ptr));
    NPZWriter::add_array("A_col_idx", reinterpret_cast<const std::vector<int>&>(A.col_indices));

    // P matrix
    NPZWriter::add_array("P_values", P.values);
    NPZWriter::add_array("P_row_ptr", reinterpret_cast<const std::vector<int>&>(P.row_ptr));
    NPZWriter::add_array("P_col_idx", reinterpret_cast<const std::vector<int>&>(P.col_indices));

    // S matrix
    NPZWriter::add_array("S_values", S.values);
    NPZWriter::add_array("S_row_ptr", reinterpret_cast<const std::vector<int>&>(S.row_ptr));
    NPZWriter::add_array("S_col_idx", reinterpret_cast<const std::vector<int>&>(S.col_indices));

    // Coarse nodes and metadata
    NPZWriter::add_array("coarse_nodes", coarse_nodes);
    NPZWriter::add_array("metadata", metadata);

    NPZWriter::finalize();
}

void UnifiedAMGDataGenerator::write_sample_npz_theta_cnn(
    const AMGOperators::CSRMatrix& A,
    double h, double theta, double rho,
    const std::string& output_dir,
    int sample_id,
    int pattern_id,
    double epsilon,
    unsigned int refinement,
    unsigned int iterations)
{
    // Pool matrix to 50x50
    std::vector<std::vector<double>> V;
    std::vector<std::vector<int>> C;
    int pool_size = 50;

    parallel_pooling_csr(A.values, A.col_indices, A.row_ptr,
                        A.n_rows, pool_size, PoolingOp::SUM, V, C);

    // Standardize
    std_normalize(V);

    // Flatten pooled matrix for NPY
    std::vector<double> pooled_flat;
    pooled_flat.reserve(pool_size * pool_size);
    for (int i = 0; i < pool_size; ++i) {
        for (int j = 0; j < pool_size; ++j) {
            pooled_flat.push_back(V[i][j]);
        }
    }

    // Metadata: [n, rho, h, theta, pattern_id, epsilon, refinement, iterations]
    std::vector<double> metadata = {
        static_cast<double>(A.n_rows),
        rho,
        h,
        theta,
        static_cast<double>(pattern_id),
        epsilon,
        static_cast<double>(refinement),
        static_cast<double>(iterations)
    };

    std::vector<double> y_arr = {rho};  // y = rho (convergence factor to predict)

    // Create filename
    std::ostringstream filename;
    filename << output_dir << "/sample_" << std::setfill('0') << std::setw(5) << sample_id << ".npz";

    // Write NPZ file
    NPZWriter::begin(filename.str());
    NPZWriter::add_array_2d("pooled_matrix", pooled_flat, pool_size, pool_size);
    NPZWriter::add_array("y", y_arr);
    NPZWriter::add_array("metadata", metadata);
    NPZWriter::finalize();
}

void UnifiedAMGDataGenerator::print_progress(
    int current, int total,
    std::chrono::steady_clock::time_point start)
{
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start).count();
    double rate = current / elapsed;
    double eta = (total - current) / rate;

    // Save current cout format state
    std::ios_base::fmtflags old_flags = std::cout.flags();
    std::streamsize old_precision = std::cout.precision();

    std::cout << "[" << current << "/" << total << "] "
              << std::fixed << std::setprecision(1)
              << (100.0 * current / total) << "% | "
              << "Rate: " << std::setprecision(1) << rate << " samples/s | "
              << "ETA: " << std::setprecision(0) << eta << "s"
              << std::endl;

    // Restore original cout format state
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

void UnifiedAMGDataGenerator::print_final_summary(
    int total,
    std::chrono::steady_clock::time_point start)
{
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Save current cout format state
    std::ios_base::fmtflags old_flags = std::cout.flags();
    std::streamsize old_precision = std::cout.precision();

    std::cout << "Generation complete!" << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << elapsed << "s" << std::endl;
    std::cout << "Average rate: " << std::setprecision(2)
              << (total / elapsed) << " samples/s" << std::endl;

    // Restore original cout format state
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

void UnifiedAMGDataGenerator::generate_diffusion() {
    FEMScaleConfig config = ConfigFactory::get_diffusion_config(args_.scale, args_.split);
    const std::array<AMGDiffusion::DiffusionPattern, 4> patterns = {
        AMGDiffusion::DiffusionPattern::VERTICAL_STRIPES,
        AMGDiffusion::DiffusionPattern::CHECKERBOARD_2X2,
        AMGDiffusion::DiffusionPattern::VERTICAL_STRIPES_4,
        AMGDiffusion::DiffusionPattern::CHECKERBOARD_4X4
    };
    const std::string diffusion_scale = scale_to_string(args_.scale);
    const bool use_simplex = (args_.cell_type == CellType::SIMPLEX);
    // Empty for QUAD keeps output paths byte-identical to pre-existing runs;
    // SIMPLEX gets its own non-colliding subtree.
    const std::string cell_type_segment = use_simplex ? "simplex/" : "";
    const std::string diffusion_base = args_.output_dir + "/diffusion/" + diffusion_scale + "/" + cell_type_segment;
    const std::vector<double> param1_values_vec = config.param1_values;

    int total = static_cast<int>(param1_values_vec.size() * config.refinements.size() * config.theta_values.size() * patterns.size());
    int current = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Diffusion dataset" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "Format: " << (args_.use_npy ? "NPZ (binary)" : "CSV (text)") << std::endl;
    std::cout << "\n" << std::endl;

    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    std::string npy_theta_cnn_dir, npy_theta_gnn_dir, npy_p_value_dir;

    if (args_.use_npy) {
        std::string problem_suffix = problem_type_to_string(args_.problem);

        if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
            npy_theta_cnn_dir = diffusion_base + "theta_cnn_npy/" + problem_suffix;
            create_directories(npy_theta_cnn_dir);
        }
        if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
            npy_theta_gnn_dir = diffusion_base + "theta_gnn_npy/" + problem_suffix;
            create_directories(npy_theta_gnn_dir);
        }
        if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
            npy_p_value_dir = diffusion_base + "p_value_npy/" + problem_suffix;
            create_directories(npy_p_value_dir);
        }
    } else {
        std::string problem_suffix = problem_type_to_string(args_.problem);

        if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
            std::string path = diffusion_base + "theta_cnn/";
            create_directories(path);
            path += problem_suffix + ".csv";
            theta_cnn_file.open(path);
            if (!theta_cnn_file) {
                throw std::runtime_error("Failed to open diffusion theta_cnn output file: " + path);
            }
        }

        if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
            std::string path = diffusion_base + "theta_gnn/";
            create_directories(path);
            path += problem_suffix + ".csv";
            theta_gnn_file.open(path);
            if (!theta_gnn_file) {
                throw std::runtime_error("Failed to open diffusion theta_gnn output file: " + path);
            }
        }

        if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
            std::string path = diffusion_base + "p_value/";
            create_directories(path);
            path += problem_suffix + ".csv";
            p_value_file.open(path);
            if (!p_value_file) {
                throw std::runtime_error("Failed to open diffusion p_value output file: " + path);
            }
        }
    }

    std::string diffusion_report_dir = diffusion_base + "diffusion_reports/";
    create_directories(diffusion_report_dir);
    std::string diffusion_report_path = diffusion_report_dir +
        "D.csv";
    {
        std::ofstream report(diffusion_report_path, std::ios::out | std::ios::trunc);
        if (!report) {
            throw std::runtime_error("Failed to initialize diffusion report file: " + diffusion_report_path);
        }
        report << "scale,scale_id,pattern,pattern_id,epsilon,refinement,h,theta,rho,iterations,elapsed_sec,n_levels,n,nnz\n";
    }

    // Nested loops over all parameters and the four coefficient patterns
    for (auto pattern : patterns) {
        for (double epsilon : param1_values_vec) {
            for (unsigned int refinement : config.refinements) {
                for (double theta : config.theta_values) {
                    AMGDiffusion::Solver<2> solver(pattern, epsilon, refinement, use_simplex);
                    solver.set_theta(theta);

                    // Solve (this stores convergence metrics internally)
                    std::ofstream dummy_file;
                    solver.run(dummy_file);

                    // Extract results using getter methods
                    AMGOperators::CSRMatrix A = solver.get_system_matrix_csr();
                    double h = solver.get_mesh_size();
                    double rho = solver.get_convergence_factor();
                    double elapsed_sec = solver.get_linear_solve_elapsed_sec();
                    unsigned int iterations = solver.get_solver_iterations();
                    unsigned int n_levels = solver.get_amg_hierarchy_levels();
                    int pattern_id = static_cast<int>(pattern);
                    std::string pattern_name = AMGDiffusion::pattern_to_string(pattern);

                    DiffusionExperimentRecord record{
                        diffusion_scale,
                        static_cast<int>(args_.scale),
                        pattern_name,
                        pattern_id,
                        epsilon,
                        refinement,
                        h,
                        theta,
                        rho,
                        iterations,
                        elapsed_sec,
                        static_cast<int>(n_levels),
                        A.n_rows,
                        A.nnz()
                    };
                    append_diffusion_report(diffusion_report_path, record, false);

                    // Write sample
                    if (args_.use_npy) {
                        if (!npy_theta_cnn_dir.empty()) {
                            write_sample_npz_theta_cnn(A, h, theta, rho, npy_theta_cnn_dir, current,
                                                       pattern_id, epsilon, refinement, iterations);
                        }
                        if (!npy_theta_gnn_dir.empty()) {
                            write_sample_npz_theta_gnn(A, h, theta, rho, npy_theta_gnn_dir, current,
                                                       pattern_id, epsilon, refinement, iterations);
                        }
                        if (!npy_p_value_dir.empty()) {
                            write_sample_npz_p_value(A, h, theta, rho, npy_p_value_dir, current,
                                                     pattern_id, epsilon, refinement, iterations);
                        }
                    } else {
                        write_sample(A, h, theta, rho,
                                    theta_cnn_file, theta_gnn_file, p_value_file);
                    }

                    current++;
                    if (args_.verbose && current % 100 == 0) {
                        print_progress(current, total, start_time);
                    }
                }
            }
        }
    }

    if (!args_.use_npy) {
        close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }
    print_final_summary(current, start_time);
}

void UnifiedAMGDataGenerator::generate_elastic() {
    FEMScaleConfig config = ConfigFactory::get_elastic_config(args_.scale, args_.split);

    int total = config.total_samples();
    int current = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Elastic dataset" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "Format: " << (args_.use_npy ? "NPZ (binary)" : "CSV (text)") << std::endl;
    std::cout << "\n" << std::endl;

    // Setup output (CSV or NPZ)
    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    std::string npy_theta_cnn_dir, npy_theta_gnn_dir, npy_p_value_dir;

    if (args_.use_npy) {
        std::string problem_suffix = problem_type_to_string(args_.problem);
        std::string split_prefix = (args_.split == DatasetSplit::TRAIN) ? "train" : "test";

        if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
            npy_theta_cnn_dir = output_base_ + "theta_cnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_cnn_dir);
        }
        if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
            npy_theta_gnn_dir = output_base_ + "theta_gnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_gnn_dir);
        }
        if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
            npy_p_value_dir = output_base_ + "p_value_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_p_value_dir);
        }
    } else {
        open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }

    std::vector<double> param1_values_vec;
    // We can create a train/test split by using only a subset of epsilon values for training
    if (args_.split == DatasetSplit::TRAIN) {
        size_t train_size = static_cast<size_t>(config.param1_values.size() * (2/3.0));  // Use first 2/3 of epsilon values for training
        param1_values_vec = std::vector<double>(config.param1_values.begin(), config.param1_values.begin() + train_size);
    } else {
        size_t test_size = static_cast<size_t>(config.param1_values.size() * (1/3.0));  // Use last 1/3 of epsilon values for testing
        param1_values_vec = std::vector<double>(config.param1_values.end() - test_size, config.param1_values.end());
    }    

    // Nested loops over parameters
    for (double E : param1_values_vec) {
        for (double nu : config.param2_values) {
            // Compute Lamé parameters
            AMGElastic::MaterialProperties material(E, nu);

            for (unsigned int refinement : config.refinements) {
                for (double theta : config.theta_values) {
                    // Create solver
                    AMGElastic::ElasticProblem<2> solver(material.get_lambda(), material.get_mu());
                    solver.set_theta(theta);

                    // Solve
                    std::ofstream dummy_file;
                    solver.run(dummy_file);

                    // Extract results
                    AMGOperators::CSRMatrix A = solver.get_system_matrix_csr();
                    double h = solver.get_mesh_size();
                    double rho = solver.get_convergence_factor();

                    // Write sample
                    if (args_.use_npy) {
                        if (!npy_theta_cnn_dir.empty()) {
                            write_sample_npz_theta_cnn(A, h, theta, rho, npy_theta_cnn_dir, current);
                        }
                        if (!npy_theta_gnn_dir.empty()) {
                            write_sample_npz_theta_gnn(A, h, theta, rho, npy_theta_gnn_dir, current);
                        }
                        if (!npy_p_value_dir.empty()) {
                            write_sample_npz_p_value(A, h, theta, rho, npy_p_value_dir, current);
                        }
                    } else {
                        write_sample(A, h, theta, rho,
                                    theta_cnn_file, theta_gnn_file, p_value_file);
                    }

                    current++;
                    if (args_.verbose && current % 100 == 0) {
                        print_progress(current, total, start_time);
                    }
                }
            }
        }
    }

    if (!args_.use_npy) {
        close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }
    print_final_summary(current, start_time);
}

void UnifiedAMGDataGenerator::generate_stokes() {
#ifdef DEAL_II_WITH_P4EST
    FEMScaleConfig config = ConfigFactory::get_stokes_config(args_.scale, args_.split);

    int total = config.total_samples();
    int current = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Stokes dataset" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << total << std::endl;
    std::cout << "Format: " << (args_.use_npy ? "NPZ (binary)" : "CSV (text)") << std::endl;
    std::cout << "\n" << std::endl;

    // Setup output (CSV or NPZ)
    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    std::string npy_theta_cnn_dir, npy_theta_gnn_dir, npy_p_value_dir;

    if (args_.use_npy) {
        std::string problem_suffix = problem_type_to_string(args_.problem);
        std::string split_prefix = (args_.split == DatasetSplit::TRAIN) ? "train" : "test";

        if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
            npy_theta_cnn_dir = output_base_ + "theta_cnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_cnn_dir);
        }
        if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
            npy_theta_gnn_dir = output_base_ + "theta_gnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_gnn_dir);
        }
        if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
            npy_p_value_dir = output_base_ + "p_value_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_p_value_dir);
        }
    } else {
        open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }

    unsigned int boundary_choice = 1;  // Fixed boundary condition

    std::vector<double> param1_values_vec;
    // For diffusion, we can create a train/test split by using only a subset of epsilon values for training
    if (args_.split == DatasetSplit::TRAIN) {
        size_t train_size = static_cast<size_t>(config.param1_values.size() * (2/3.0));  // Use first 2/3 of epsilon values for training
        param1_values_vec = std::vector<double>(config.param1_values.begin(), config.param1_values.begin() + train_size);
    } else {
        size_t test_size = static_cast<size_t>(config.param1_values.size() * (1/3.0));  // Use last 1/3 of epsilon values for testing
        param1_values_vec = std::vector<double>(config.param1_values.end() - test_size, config.param1_values.end());
    }

    for (double viscosity : param1_values_vec) {
        for (double velocity_degree_double : config.param2_values) {
            unsigned int velocity_degree = static_cast<unsigned int>(velocity_degree_double);

            for (unsigned int refinement : config.refinements) {
                for (double theta : config.theta_values) {
                    // Create solver
                    AMGStokes::StokesProblem<2> solver(velocity_degree, viscosity, boundary_choice);
                    solver.set_theta(theta);
                    solver.set_init_refinement(refinement);
                    solver.set_n_cycle(1);  // Single cycle for dataset generation

                    // Solve
                    std::ofstream dummy_file;
                    solver.run(dummy_file);

                    // Extract results
                    AMGOperators::CSRMatrix A = solver.get_system_matrix_csr();
                    double h = solver.get_mesh_size();
                    double rho = solver.get_convergence_factor();

                    // Write sample
                    if (args_.use_npy) {
                        if (!npy_theta_cnn_dir.empty()) {
                            write_sample_npz_theta_cnn(A, h, theta, rho, npy_theta_cnn_dir, current);
                        }
                        if (!npy_theta_gnn_dir.empty()) {
                            write_sample_npz_theta_gnn(A, h, theta, rho, npy_theta_gnn_dir, current);
                        }
                        if (!npy_p_value_dir.empty()) {
                            write_sample_npz_p_value(A, h, theta, rho, npy_p_value_dir, current);
                        }
                    } else {
                        write_sample(A, h, theta, rho,
                                    theta_cnn_file, theta_gnn_file, p_value_file);
                    }

                    current++;
                    if (args_.verbose && current % 100 == 0) {
                        print_progress(current, total, start_time);
                    }
                }
            }
        }
    }

    if (!args_.use_npy) {
        close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }
    print_final_summary(current, start_time);
#else
    throw std::runtime_error(
        "Stokes generation requires a deal.II build with p4est support "
        "(DEAL_II_WITH_P4EST). Use a p4est-enabled deal.II install or choose another problem type."
    );
#endif
}

void UnifiedAMGDataGenerator::generate_graph_laplacian() {
    GraphScaleConfig config = ConfigFactory::get_graph_laplacian_config(args_.scale, args_.split);

    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Graph Laplacian dataset" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << config.num_samples << std::endl;
    std::cout << "Nodes per graph: " << config.num_points << std::endl;
    std::cout << "Format: " << (args_.use_npy ? "NPZ (binary)" : "CSV (text)") << std::endl;
    std::cout << "\n" << std::endl;

    // Setup output (CSV or NPZ)
    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    std::string npy_theta_cnn_dir, npy_theta_gnn_dir, npy_p_value_dir;

    if (args_.use_npy) {
        // Create NPZ output directories
        std::string problem_suffix = problem_type_to_string(args_.problem);
        std::string split_prefix = (args_.split == DatasetSplit::TRAIN) ? "train" : "test";

        if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
            npy_theta_cnn_dir = output_base_ + "theta_cnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_cnn_dir);
        }
        if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
            npy_theta_gnn_dir = output_base_ + "theta_gnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_gnn_dir);
        }
        if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
            npy_p_value_dir = output_base_ + "p_value_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_p_value_dir);
        }
    } else {
        open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }

    if (args_.split == DatasetSplit::TEST) {
        config.num_points = int(0.2 * config.num_points);  // Smaller graphs for testing
    }

    // Configure graph generator
    GraphLaplacian::GraphConfig graph_config;
    graph_config.type = GraphLaplacian::GraphType::LOGNORMAL_LAPLACIAN;
    graph_config.num_points = config.num_points;
    graph_config.log_std = 1.0;
    graph_config.seed = args_.seed;

    // Pre-generate all samples in parallel
    std::vector<AMGOperators::CSRMatrix> matrices(config.num_samples);
    std::vector<double> thetas(config.num_samples);
    std::vector<double> rhos(config.num_samples);
    std::vector<double> hs(config.num_samples);

    #pragma omp parallel for schedule(dynamic) if (args_.num_threads != 1)
    for (int i = 0; i < config.num_samples; ++i) {
        int thread_seed = args_.seed + i;

        GraphLaplacian::GraphConfig local_config = graph_config;
        GraphLaplacian::GraphLaplacianGenerator generator(local_config);
        generator.set_seed(thread_seed);

        // Generate matrix and convert to CSR
        auto eigen_mat = generator.generate();
        matrices[i] = GraphLaplacian::eigenToCSR(eigen_mat);

        // Compute metrics (using Eigen matrix)
        hs[i] = GraphLaplacian::compute_mesh_size(eigen_mat, config.num_points);
        thetas[i] = GraphLaplacian::find_optimal_theta_for_graph(eigen_mat, rhos[i]);

        #pragma omp critical
        {
            if (args_.verbose && (i + 1) % 100 == 0) {
                print_progress(i + 1, config.num_samples, start_time);
            }
        }
    }

    // Write samples (CSV or NPZ)
    if (args_.use_npy) {
        // Write NPZ files
        for (int i = 0; i < config.num_samples; ++i) {
            if (!npy_theta_cnn_dir.empty()) {
                write_sample_npz_theta_cnn(matrices[i], hs[i], thetas[i], rhos[i], npy_theta_cnn_dir, i);
            }
            if (!npy_theta_gnn_dir.empty()) {
                write_sample_npz_theta_gnn(matrices[i], hs[i], thetas[i], rhos[i], npy_theta_gnn_dir, i);
            }
            if (!npy_p_value_dir.empty()) {
                write_sample_npz_p_value(matrices[i], hs[i], thetas[i], rhos[i], npy_p_value_dir, i);
            }
        }
    } else {
        // Write CSV files
        for (int i = 0; i < config.num_samples; ++i) {
            write_sample(matrices[i], hs[i], thetas[i], rhos[i],
                        theta_cnn_file, theta_gnn_file, p_value_file);
        }
        close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }

    print_final_summary(config.num_samples, start_time);
}

void UnifiedAMGDataGenerator::generate_spectral_clustering() {
    GraphScaleConfig config = ConfigFactory::get_spectral_clustering_config(args_.scale, args_.split);

    auto start_time = std::chrono::steady_clock::now();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Generating Spectral Clustering dataset" << std::endl;
    std::cout << "Scale: " << scale_to_string(args_.scale) << std::endl;
    std::cout << "Total samples: " << config.num_samples << std::endl;
    std::cout << "Nodes per graph: " << config.num_points << std::endl;
    std::cout << "Format: " << (args_.use_npy ? "NPZ (binary)" : "CSV (text)") << std::endl;
    std::cout << "\n" << std::endl;

    // Setup output (CSV or NPZ)
    std::ofstream theta_cnn_file, theta_gnn_file, p_value_file;
    std::string npy_theta_cnn_dir, npy_theta_gnn_dir, npy_p_value_dir;

    if (args_.use_npy) {
        std::string problem_suffix = problem_type_to_string(args_.problem);
        std::string split_prefix = (args_.split == DatasetSplit::TRAIN) ? "train" : "test";

        if (args_.format == OutputFormat::THETA_CNN || args_.format == OutputFormat::ALL) {
            npy_theta_cnn_dir = output_base_ + "theta_cnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_cnn_dir);
        }
        if (args_.format == OutputFormat::THETA_GNN || args_.format == OutputFormat::ALL) {
            npy_theta_gnn_dir = output_base_ + "theta_gnn_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_theta_gnn_dir);
        }
        if (args_.format == OutputFormat::P_VALUE || args_.format == OutputFormat::ALL) {
            npy_p_value_dir = output_base_ + "p_value_npy/" + split_prefix + "_" + problem_suffix;
            create_directories(npy_p_value_dir);
        }
    } else {
        open_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }

    if (args_.split == DatasetSplit::TEST) {
        config.num_points = int(0.2 * config.num_points);  // Smaller graphs for testing
    }

    // Configure graph generator for spectral clustering
    GraphLaplacian::GraphConfig graph_config;
    graph_config.type = GraphLaplacian::GraphType::SPECTRAL_CLUSTERING;
    graph_config.num_points = config.num_points;
    graph_config.k_neighbors = 10;
    graph_config.sigma = 0.1;
    graph_config.seed = args_.seed;

    // Pre-generate all samples in parallel
    std::vector<AMGOperators::CSRMatrix> matrices(config.num_samples);
    std::vector<double> thetas(config.num_samples);
    std::vector<double> rhos(config.num_samples);
    std::vector<double> hs(config.num_samples);

    #pragma omp parallel for schedule(dynamic) if (args_.num_threads != 1)
    for (int i = 0; i < config.num_samples; ++i) {
        int thread_seed = args_.seed + i;

        GraphLaplacian::GraphConfig local_config = graph_config;
        GraphLaplacian::GraphLaplacianGenerator generator(local_config);
        generator.set_seed(thread_seed);

        // Generate matrix and convert to CSR
        auto eigen_mat = generator.generate();
        matrices[i] = GraphLaplacian::eigenToCSR(eigen_mat);

        // Compute metrics (using Eigen matrix)
        hs[i] = GraphLaplacian::compute_mesh_size(eigen_mat, config.num_points);
        thetas[i] = GraphLaplacian::find_optimal_theta_for_graph(eigen_mat, rhos[i]);

        #pragma omp critical
        {
            if (args_.verbose && (i + 1) % 100 == 0) {
                print_progress(i + 1, config.num_samples, start_time);
            }
        }
    }

    // Write samples (CSV or NPZ)
    if (args_.use_npy) {
        for (int i = 0; i < config.num_samples; ++i) {
            if (!npy_theta_cnn_dir.empty()) {
                write_sample_npz_theta_cnn(matrices[i], hs[i], thetas[i], rhos[i], npy_theta_cnn_dir, i);
            }
            if (!npy_theta_gnn_dir.empty()) {
                write_sample_npz_theta_gnn(matrices[i], hs[i], thetas[i], rhos[i], npy_theta_gnn_dir, i);
            }
            if (!npy_p_value_dir.empty()) {
                write_sample_npz_p_value(matrices[i], hs[i], thetas[i], rhos[i], npy_p_value_dir, i);
            }
        }
    } else {
        for (int i = 0; i < config.num_samples; ++i) {
            write_sample(matrices[i], hs[i], thetas[i], rhos[i],
                        theta_cnn_file, theta_gnn_file, p_value_file);
        }
        close_output_files(theta_cnn_file, theta_gnn_file, p_value_file);
    }

    print_final_summary(config.num_samples, start_time);
}

// Command-Line Argument Parsing
void print_help() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Unified AMG Data Generator\n";

    std::cout << "Usage:\n";
    std::cout << "  ./generate_amg_data [OPTIONS]\n\n";

    std::cout << "Required Arguments:\n";
    std::cout << "  -p, --problem TYPE        Problem type: D|E|S|GL|SC\n";
    std::cout << "  -f, --format FORMAT       Output format: theta-cnn|theta-gnn|p-value|all\n";
    std::cout << "  -c, --scale SCALE         Dataset scale: small|large|paper\n\n";

    std::cout << "Optional Arguments:\n";
    std::cout << "  -s, --split SPLIT         Legacy split for non-diffusion data: train|test\n";
    std::cout << "  -o, --output-dir DIR      Output directory (default: ./datasets/unified)\n";
    std::cout << "  -t NUM         OpenMP threads (default: auto)\n";
    std::cout << "  --seed SEED               Random seed (default: 42)\n";
    std::cout << "  -v, --verbose             Verbose progress output\n";
    std::cout << "  --use-npy                 Use NPZ (binary) format instead of CSV (text)\n";
    std::cout << "  -e, --cell-type TYPE      Mesh cell type, Diffusion only: quad|simplex (default: quad)\n";
    std::cout << "  -h, --help                Show this help message\n\n";

    std::cout << "Problem Types:\n";
    std::cout << "  D  - Diffusion equations (piecewise-constant coefficient patterns)\n";
    std::cout << "  E  - Elastic equations (linear elasticity)\n";
    std::cout << "  S  - Stokes equations (incompressible flow)\n";
    std::cout << "  GL - Graph Laplacian (Delaunay with lognormal weights)\n";
    std::cout << "  SC - Spectral Clustering (k-NN graphs)\n\n";

    std::cout << "Examples:\n";
    std::cout << "  # Generate small diffusion dataset in theta-cnn format\n";
    std::cout << "  ./generate_amg_data -p D -f theta-cnn -c small\n\n";

    std::cout << "  # Generate paper graph Laplacian test set in all formats\n";
    std::cout << "  ./generate_amg_data -p GL -s test -f all -c paper -t 16\n\n";

    std::cout << "  # Generate large elastic training with custom output\n";
    std::cout << "  ./generate_amg_data -p E -s train -f p-value -c large -o custom_data\n\n";

    std::cout << "========================================\n\n";
}

bool parse_arguments(int argc, char* argv[], CommandLineArgs& args) {
    bool has_problem = false, has_split = false, has_format = false, has_scale = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help();
            return false;
        }
        else if (arg == "-p" || arg == "--problem") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string type = argv[i];
            if (type == "D") args.problem = ProblemType::DIFFUSION;
            else if (type == "E") args.problem = ProblemType::ELASTIC;
            else if (type == "S") args.problem = ProblemType::STOKES;
            else if (type == "GL") args.problem = ProblemType::GRAPH_LAPLACIAN;
            else if (type == "SC") args.problem = ProblemType::SPECTRAL_CLUSTERING;
            else {
                std::cerr << "Error: Invalid problem type: " << type << std::endl;
                return false;
            }
            has_problem = true;
        }
        else if (arg == "-s" || arg == "--split") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string split = argv[i];
            if (split == "train") args.split = DatasetSplit::TRAIN;
            else if (split == "test") args.split = DatasetSplit::TEST;
            else if (split == "all") args.split = DatasetSplit::ALL;
            else {
                std::cerr << "Error: Invalid split: " << split << std::endl;
                return false;
            }
            has_split = true;
        }
        else if (arg == "-f" || arg == "--format") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string format = argv[i];
            if (format == "theta-cnn") args.format = OutputFormat::THETA_CNN;
            else if (format == "theta-gnn") args.format = OutputFormat::THETA_GNN;
            else if (format == "p-value") args.format = OutputFormat::P_VALUE;
            else if (format == "all") args.format = OutputFormat::ALL;
            else {
                std::cerr << "Error: Invalid format: " << format << std::endl;
                return false;
            }
            has_format = true;
        }
        else if (arg == "-c" || arg == "--scale") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string scale = argv[i];
            if (scale == "small") args.scale = DatasetScale::SMALL;
            else if (scale == "large") args.scale = DatasetScale::LARGE;
            else if (scale == "paper") args.scale = DatasetScale::PAPER;
            else {
                std::cerr << "Error: Invalid scale: " << scale << std::endl;
                std::cerr << "Valid scales: small|large|paper" << std::endl;
                return false;
            }
            has_scale = true;
        }
        else if (arg == "-o" || arg == "--output-dir") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            args.output_dir = argv[i];
        }
        else if (arg == "-t") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            args.num_threads = std::stoi(argv[i]);
        }
        else if (arg == "--seed") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            args.seed = std::stoi(argv[i]);
        }
        else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        }
        else if (arg == "--use-npy") {
            args.use_npy = true;
        }
        else if (arg == "-e" || arg == "--cell-type") {
            if (++i >= argc) {
                std::cerr << "Error: Missing value for " << arg << std::endl;
                return false;
            }
            std::string cell_type = argv[i];
            if (cell_type == "quad") args.cell_type = CellType::QUAD;
            else if (cell_type == "simplex") args.cell_type = CellType::SIMPLEX;
            else {
                std::cerr << "Error: Invalid cell type: " << cell_type << std::endl;
                std::cerr << "Valid cell types: quad|simplex" << std::endl;
                return false;
            }
        }
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (!has_problem || !has_format || !has_scale) {
        std::cerr << "Error: Missing required arguments" << std::endl;
        print_help();
        return false;
    }
    if (args.problem != ProblemType::DIFFUSION && !has_split) {
        std::cerr << "Error: Non-diffusion datasets still require -s/--split train|test" << std::endl;
        print_help();
        return false;
    }
    if (args.problem == ProblemType::DIFFUSION && has_split) {
        std::cout << "Note: diffusion generation is unsplit; ignoring -s/--split for output layout.\n";
        args.split = DatasetSplit::ALL;
    }
    if (args.problem != ProblemType::DIFFUSION && args.cell_type != CellType::QUAD) {
        std::cout << "Note: -e/--cell-type is only supported for Diffusion (-p D); ignoring for this problem type.\n";
        args.cell_type = CellType::QUAD;
    }

    return true;
}

void print_configuration(const CommandLineArgs& args) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Configuration\n";
    std::cout << "Problem type: " << problem_type_to_string(args.problem) << "\n";
    std::cout << "Dataset split: " << split_to_string(args.split) << "\n";
    std::cout << "Output format: ";
    switch (args.format) {
        case OutputFormat::THETA_CNN: std::cout << "theta-cnn"; break;
        case OutputFormat::THETA_GNN: std::cout << "theta-gnn"; break;
        case OutputFormat::P_VALUE: std::cout << "p-value"; break;
        case OutputFormat::ALL: std::cout << "all"; break;
    }
    std::cout << "\n";
    std::cout << "Scale: " << scale_to_string(args.scale) << "\n";
    if (args.problem == ProblemType::DIFFUSION) {
        std::cout << "Cell type: " << cell_type_to_string(args.cell_type) << "\n";
    }
    std::cout << "Output directory: " << args.output_dir << "\n";
    std::cout << "Random seed: " << args.seed << "\n";
    if (args.num_threads > 0) {
        std::cout << "OpenMP threads: " << args.num_threads << "\n";
    } else {
        std::cout << "OpenMP threads: auto\n";
    }
    std::cout << "Verbose: " << (args.verbose ? "yes" : "no") << "\n";
    std::cout << "\n";
}


int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help();
            return 0;
        }
    }

    // Initialize MPI (needed for FEM solvers)
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Parse command-line arguments
    CommandLineArgs args;
    if (!parse_arguments(argc, argv, args)) {
        return 1;
    }

    // Set OpenMP threads
    if (args.num_threads > 0) {
        omp_set_num_threads(args.num_threads);
    }

    // Print configuration
    print_configuration(args);

    // Create generator and run
    try {
        UnifiedAMGDataGenerator generator(args);
        generator.generate();
    } catch (const std::exception& e) {
        std::cerr << "\n Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
