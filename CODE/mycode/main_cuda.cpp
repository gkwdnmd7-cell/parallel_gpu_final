#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "JacobiSet.h"
#include "TriMeshJ.h"
#include "jacobi_gpu.h"

static void print_usage(const char *program) {
    std::cerr << "Usage:\n"
              << "  " << program << " --preprocess-only <mesh.obj> [--dump-edges <edges.txt>]\n"
              << "  " << program << " <mesh.obj> [--output <jacobi.txt>] [--dump-degenerate <edges.txt>]\n";
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

static bool write_degenerate_edges(const std::string &filename,
                                   const std::vector<EdgeRecord> &edges,
                                   const std::vector<JacobiGpuResult> &results) {
    std::ofstream out(filename.c_str());
    if (!out.is_open()) {
        std::cerr << "failed to open degenerate dump file: " << filename << "\n";
        return false;
    }

    out << "# edge_id e1 e2 link1 link2 is_jacobi\n";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (!results[i].degenerate) {
            continue;
        }
        out << i << " "
            << edges[i].e1 << " "
            << edges[i].e2 << " "
            << edges[i].link1 << " "
            << edges[i].link2 << " "
            << results[i].is_jacobi << "\n";
    }
    return true;
}

static size_t apply_sos_fallback(const trimesh::TriMeshJ &mesh,
                                 const std::vector<double> &f,
                                 const std::vector<double> &g,
                                 const std::vector<EdgeRecord> &edges,
                                 std::vector<JacobiGpuResult> *results) {
    if (results == nullptr) {
        return 0;
    }

    JacobiSet js(&mesh, &f, &g);
    size_t fallback_count = 0;

    for (size_t i = 0; i < edges.size(); ++i) {
        if (!(*results)[i].degenerate) {
            continue;
        }

        const EdgeRecord &edge = edges[i];
        const bool f_lower_v1 = js.is_lowerLink(edge.e1, edge.e2, edge.link1, &g);
        const bool f_lower_v2 = js.is_lowerLink(edge.e1, edge.e2, edge.link2, &g);
        const bool g_lower_v1 = js.is_lowerLink(edge.e1, edge.e2, edge.link1, &f);
        const bool g_lower_v2 = js.is_lowerLink(edge.e1, edge.e2, edge.link2, &f);

        const bool f_crit = (f_lower_v1 == f_lower_v2);
        JacobiGpuResult &out = (*results)[i];
        out.fv1 = edge.e1;
        out.fv2 = edge.e2;
        out.min_f = !f_lower_v1;
        out.min_g = g_lower_v1;
        out.pos_align = js.alignment(edge.e1, edge.e2);
        out.is_jacobi = f_crit;
        out.degenerate = 1;

        (void)g_lower_v1;
        (void)g_lower_v2;
        ++fallback_count;
    }

    return fallback_count;
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
    std::string degenerate_dump_file;

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
        if (argc != 2 && argc != 4 && argc != 6) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        mesh_file = argv[1];
        output_file = default_output_filename(mesh_file);
        for (int argi = 2; argi < argc; argi += 2) {
            if (argi + 1 >= argc) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            const std::string flag = argv[argi];
            const std::string value = argv[argi + 1];
            if (flag == "--output") {
                output_file = value;
            } else if (flag == "--dump-degenerate") {
                degenerate_dump_file = value;
            } else {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
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

    const size_t sos_fallback_count = apply_sos_fallback(mesh, f, g, interior_edges, &results);

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
    if (!degenerate_dump_file.empty()) {
        if (!write_degenerate_edges(degenerate_dump_file, interior_edges, results)) {
            return EXIT_FAILURE;
        }
    }

    std::cout << "jacobi_edges: " << jacobi_count << "\n";
    std::cout << "degenerate_edges: " << degenerate_count << "\n";
    std::cout << "sos_fallback_edges: " << sos_fallback_count << "\n";
    std::cout << "gpu_kernel_ms: " << timing.kernel_ms << "\n";
    std::cout << "gpu_total_ms: " << timing.total_ms << "\n";
    std::cout << "output: " << output_file << "\n";
    if (!degenerate_dump_file.empty()) {
        std::cout << "degenerate_dump: " << degenerate_dump_file << "\n";
    }

    return EXIT_SUCCESS;
}
