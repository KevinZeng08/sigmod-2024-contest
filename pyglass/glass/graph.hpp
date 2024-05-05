#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/memory.hpp"

namespace glass {

constexpr int EMPTY_ID = -1;

template <typename node_t> struct Graph {
  int N, K;

  node_t *data = nullptr;

  float *timestamps = nullptr;

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  std::vector<int> eps;

  Graph() = default;

  Graph(node_t *edges, int N, int K) : N(N), K(K), data(edges) {}

  Graph(int N, int K)
      : N(N), K(K), data((node_t *)alloc2M((size_t)N * K * sizeof(node_t))), 
              timestamps((float *)alloc2M((size_t)N * K * sizeof(float))) {}

  Graph(const Graph &g) : Graph(g.N, g.K) {
    this->eps = g.eps;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        at(i, j) = g.at(i, j);
        at_time(i, j) = g.at_time(i, j);
      }
    }
    if (g.initializer) {
      initializer = std::make_unique<HNSWInitializer>(*g.initializer);
    }
  }

  Graph(Graph &&g) noexcept : N(g.N), K(g.K), data(g.data), timestamps(g.timestamps), initializer(std::move(g.initializer)), eps(std::move(g.eps)) {
    g.data = nullptr;
    g.timestamps = nullptr;
  }

  void init(int N, int K) {
    data = (node_t *)alloc2M((size_t)N * K * sizeof(node_t));
    timestamps = (float *)alloc2M((size_t)N * K * sizeof(float));
    std::memset(data, -1, N * K * sizeof(node_t));
    std::memset(timestamps, 0, N * K * sizeof(float));
    this->K = K;
    this->N = N;
  }

  ~Graph() { 
    if (data != nullptr) {
      free(data); 
    }
    if (timestamps != nullptr) {
      free(timestamps);
    }
  }

  const int *edges(int u) const { return data + K * u; }

  int *edges(int u) { return data + K * u; }

  const float *get_timestamp(int u) const { return timestamps + K * u; }

  node_t at(int i, int j) const { return data[i * K + j]; }

  node_t &at(int i, int j) { return data[i * K + j]; }

  float at_time(int i, int j) const { return timestamps[i * K + j]; }

  float &at_time(int i, int j) { return timestamps[i * K + j]; }

  void prefetch(int u, int lines) const {
    mem_prefetch((char *)edges(u), lines);
    mem_prefetch((char*)get_timestamp(u), lines);
  }

  template <typename Pool, typename Computer>
  void initialize_search(Pool &pool, const Computer &computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    } else {
      for (auto ep : eps) {
        pool.insert(ep, computer(ep));
      }
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same_v<node_t, int32_t>);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, N * K * 4);
    writer.write((char *)timestamps, N * K * 4);
    if (initializer) {
      initializer->save(writer);
    }
    // printf("Graph Saving done\n");
  }

  void load(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    free(timestamps);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)alloc2M((size_t)N * K * 4);
    reader.read((char *)data, N * K * 4);
    timestamps = (float *)alloc2M((size_t)N * K * 4);
    reader.read((char *)timestamps, N * K * 4);
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
    // printf("Graph Loding done\n");
  }
};

} // namespace glass