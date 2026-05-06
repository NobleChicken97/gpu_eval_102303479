#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Edge {
    int src;
    int dst;
    float t;
    float w;
};

__global__ void temporal_influence_kernel(
    const int* d_src,
    const int* d_dst,
    const float* d_t,
    const float* d_w,
    int edge_count,
    float t_now,
    float lambda,
    float* d_scores) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= edge_count) {
        return;
    }

    float dt = t_now - d_t[i];
    if (dt < 0.0f) {
        dt = 0.0f;
    }
    float contrib = d_w[i] * expf(-lambda * dt);
    atomicAdd(&d_scores[d_dst[i]], contrib);
}

bool read_edges(const std::string& path, std::vector<Edge>& edges, int& max_node, float& max_time) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        Edge e{};
        if (!(iss >> e.src >> e.dst >> e.t >> e.w)) {
            continue;
        }
        edges.push_back(e);
        max_node = std::max(max_node, std::max(e.src, e.dst));
        max_time = std::max(max_time, e.t);
    }

    return true;
}

  
void generate_random_edges(int nodes, int edges_count, std::vector<Edge>& edges, float& max_time) {
    edges.reserve(edges_count);
    for (int i = 0; i < edges_count; ++i) {
        Edge e{};
        e.src = i % nodes;
        e.dst = (i * 13) % nodes;
        e.t = static_cast<float>(i) * 0.5f;
        e.w = 1.0f + static_cast<float>(i % 5) * 0.1f;
        max_time = std::max(max_time, e.t);
        edges.push_back(e);
    }
}

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string input_path;
    int nodes = 1000;
    int random_edges = 10000;
    float lambda = 0.01f;
    float t_now = -1.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--nodes" && i + 1 < argc) {
            nodes = std::stoi(argv[++i]);
        } else if (arg == "--edges" && i + 1 < argc) {
            random_edges = std::stoi(argv[++i]);
        } else if (arg == "--lambda" && i + 1 < argc) {
            lambda = std::stof(argv[++i]);
        } else if (arg == "--now" && i + 1 < argc) {
            t_now = std::stof(argv[++i]);
        }
    }

    std::vector<Edge> edges;
    int max_node = nodes - 1;
    float max_time = 0.0f;
    
    if (!input_path.empty()) {
        if (!read_edges(input_path, edges, max_node, max_time)) {
            std::cerr << "Failed to read input file: " << input_path << "\n";
            return 1;
        }
        nodes = std::max(nodes, max_node + 1);
    } else {
        generate_random_edges(nodes, random_edges, edges, max_time);
    }

    if (t_now < 0.0f) {
        t_now = max_time;
    }

    const int edge_count = static_cast<int>(edges.size());
    if (edge_count == 0 || nodes <= 0) {
        std::cerr << "No edges or nodes to process.\n";
        return 1;
    }

    std::vector<int> h_src(edge_count);
    std::vector<int> h_dst(edge_count);
    std::vector<float> h_t(edge_count);
    std::vector<float> h_w(edge_count);
    for (int i = 0; i < edge_count; ++i) {
        h_src[i] = edges[i].src;
        h_dst[i] = edges[i].dst;
        h_t[i] = edges[i].t;
        h_w[i] = edges[i].w;
    }

    int* d_src = nullptr;
    int* d_dst = nullptr;
    float* d_t = nullptr;
    float* d_w = nullptr;
    float* d_scores = nullptr;

    cuda_check(cudaMalloc(&d_src, edge_count * sizeof(int)), "cudaMalloc d_src");
    cuda_check(cudaMalloc(&d_dst, edge_count * sizeof(int)), "cudaMalloc d_dst");
    cuda_check(cudaMalloc(&d_t, edge_count * sizeof(float)), "cudaMalloc d_t");
    cuda_check(cudaMalloc(&d_w, edge_count * sizeof(float)), "cudaMalloc d_w");
    cuda_check(cudaMalloc(&d_scores, nodes * sizeof(float)), "cudaMalloc d_scores");

    cuda_check(cudaMemcpy(d_src, h_src.data(), edge_count * sizeof(int), cudaMemcpyHostToDevice), "copy src");
    cuda_check(cudaMemcpy(d_dst, h_dst.data(), edge_count * sizeof(int), cudaMemcpyHostToDevice), "copy dst");
    cuda_check(cudaMemcpy(d_t, h_t.data(), edge_count * sizeof(float), cudaMemcpyHostToDevice), "copy t");
    cuda_check(cudaMemcpy(d_w, h_w.data(), edge_count * sizeof(float), cudaMemcpyHostToDevice), "copy w");
    cuda_check(cudaMemset(d_scores, 0, nodes * sizeof(float)), "memset scores");

    int threads = 256;
    int blocks = (edge_count + threads - 1) / threads;
    temporal_influence_kernel<<<blocks, threads>>>(
        d_src, d_dst, d_t, d_w, edge_count, t_now, lambda, d_scores);
    cuda_check(cudaDeviceSynchronize(), "kernel sync");

    std::vector<float> h_scores(nodes);
    cuda_check(cudaMemcpy(h_scores.data(), d_scores, nodes * sizeof(float), cudaMemcpyDeviceToHost), "copy scores");

    std::vector<std::pair<int, float>> ranked;
    ranked.reserve(nodes);
    for (int i = 0; i < nodes; ++i) {
        ranked.push_back({i, h_scores[i]});
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    std::cout << "Top 5 nodes by temporal influence score:\n";
    for (int i = 0; i < 5 && i < static_cast<int>(ranked.size()); ++i) {
        std::cout << "  node=" << ranked[i].first << " score=" << ranked[i].second << "\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_t);
    cudaFree(d_w);
    cudaFree(d_scores);

    return 0;
}
