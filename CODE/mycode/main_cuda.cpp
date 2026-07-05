#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "TriMeshJ.h"
#include "jacobi_gpu.h"

static void print_usage(const char *program) {
    std::cerr << "Usage:\n"
              << "  " << program << " --preprocess-only <mesh.obj> [--dump-edges <edges.txt>]\n"
              << "  " << program << " <mesh.obj> [--output <jacobi.txt>]\n";
}

static bool write_edges(const std::string &filename, const std::vector<EdgeRecord> &edges) {
    std::ofstream out(filename.c_str());
    if (!out.is_open()) {
        std::cerr << "failed to open edge dump file: " << filename << "\n";
        return false;
    }

    out << "# edge_id e1 e2 link1 link2\n";
    for (size_t i = 0; i < edges.size(); ++i) {
        out << i << " "
            << edges[i].e1 << " "
            << edges[i].e2 << " "
            << edges[i].link1 << " "
            << edges[i].link2 << "\n";
    }
    return true;
}

static std::string default_output_filename(const std::string &mesh_file) {
    const std::string::size_type dot = mesh_file.find_last_of('.');
    if (dot == std::string::npos) {
        return mesh_file + "_gpu_jacobi.txt";
    }
    return mesh_file.substr(0, dot) + "_gpu_jacobi.txt";
}

static bool write_jacobi_edges(const std::string &filename,
                               const trimesh::TriMeshJ &mesh,
                               const std::vector<JacobiGpuResult> &results) {
    std::ofstream out(filename.c_str());
    if (!out.is_open()) {
        std::cerr << "failed to open output file: " << filename << "\n";
        return false;
    }

    size_t count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].is_jacobi) {
            ++count;
        }
    }

    out << "JacobiSet\n";
    out << count << "\n";
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].is_jacobi) {
            continue;
        }
        const int a = results[i].fv1;
        const int b = results[i].fv2;
        const trimesh::point &p1 = mesh.vertices[a];
        const trimesh::point &p2 = mesh.vertices[b];
        out << a << " " << p1[0] << " " << p1[1] << " " << p1[2] << " "
            << b << " " << p2[0] << " " << p2[1] << " " << p2[2] << "\n";
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const bool preprocess_only = (std::string(argv[1]) == "--preprocess-only");
    std::string mesh_file;
    std::string edge_dump_file;
    std::string output_file;

    if (preprocess_only) {
        if (argc != 3 && argc != 5) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        mesh_file = argv[2];
        if (argc == 5) {
            if (std::string(argv[3]) != "--dump-edges") {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            edge_dump_file = argv[4];
        }
    } else {
        if (argc != 2 && argc != 4) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        mesh_file = argv[1];
        output_file = default_output_filename(mesh_file);
        if (argc == 4) {
            if (std::string(argv[2]) != "--output") {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            output_file = argv[3];
        }
    }

    trimesh::TriMeshJ mesh(mesh_file);
    mesh.need_edges();
    mesh.need_neighbors();

    std::vector<EdgeRecord> interior_edges;
    interior_edges.reserve(mesh.edges.size());

    for (std::set<trimesh::Edge>::const_iterator it = mesh.edges.begin(); it != mesh.edges.end(); ++it) {
        std::pair<int, int> link = mesh.get_e_link(*it);
        if (link.first == -1 || link.second == -1) {
            continue;
        }

        EdgeRecord record;
        record.e1 = it->first;
        record.e2 = it->second;
        record.link1 = link.first;
        record.link2 = link.second;
        interior_edges.push_back(record);
    }

    std::cout << "mesh: " << mesh_file << "\n";
    std::cout << "vertices: " << mesh.vertices.size() << "\n";
    std::cout << "edges: " << mesh.edges.size() << "\n";
    std::cout << "faces: " << mesh.faces.size() << "\n";
    std::cout << "interior_edges: " << interior_edges.size() << "\n";

    if (!edge_dump_file.empty()) {
        if (!write_edges(edge_dump_file, interior_edges)) {
            return EXIT_FAILURE;
        }
        std::cout << "edge_dump: " << edge_dump_file << "\n";
    }

    if (preprocess_only) {
        return EXIT_SUCCESS;
    }

    std::vector<double> f(mesh.vertices.size());
    std::vector<double> g(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        f[i] = mesh.vertices[i][1];
        g[i] = mesh.vertices[i][0];
    }

    std::vector<JacobiGpuResult> results;
    GpuTiming timing;
    std::string error;
    if (!compute_jacobi_gpu(interior_edges, f, g, &results, &timing, &error)) {
        std::cerr << "GPU computation failed: " << error << "\n";
        return EXIT_FAILURE;
    }

    size_t jacobi_count = 0;
    size_t degenerate_count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].is_jacobi) {
            ++jacobi_count;
        }
        if (results[i].degenerate) {
            ++degenerate_count;
        }
    }

    if (!write_jacobi_edges(output_file, mesh, results)) {
        return EXIT_FAILURE;
    }

    std::cout << "jacobi_edges: " << jacobi_count << "\n";
    std::cout << "degenerate_edges: " << degenerate_count << "\n";
    std::cout << "gpu_kernel_ms: " << timing.kernel_ms << "\n";
    std::cout << "gpu_total_ms: " << timing.total_ms << "\n";
    std::cout << "output: " << output_file << "\n";

    return EXIT_SUCCESS;
}
