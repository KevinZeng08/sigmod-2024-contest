#pragma once

#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/builder.hpp"
#include "glass/common.hpp"
#include "glass/graph.hpp"
#include "glass/hnswlib/hnswlib.h"
#include "glass/hnswlib/space_ip.h"
#include "glass/hnswlib/space_l2.h"
#include "glass/hnswlib/hnswalg.h"
#include <chrono>
#include <memory>

namespace glass {

struct HNSW : public Builder {
  int nb, dim;
  int M, efConstruction;
  constexpr static hnswlib::QuantType qtype = hnswlib::QuantType::SQ8;
  std::unique_ptr<hnswlib::HierarchicalNSW<int, qtype>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<int>> space = nullptr;

  Graph<int> final_graph;

  HNSW(int dim, const std::string &metric, int R = 32, int L = 200)
      : dim(dim), M(R / 2), efConstruction(L) {
    auto m = metric_map[metric];
    if (m == Metric::L2) {
      space = std::make_unique<hnswlib::L2Space>(dim);
    } 
  }

  void Build(float *data, float *timestamps, int N,int max_nb, int miss_end) override {
    nb = N;

    if(hnsw == nullptr) {
        hnsw = std::make_unique<hnswlib::HierarchicalNSW<int, qtype>>(space.get(), max_nb, M, efConstruction);
        if constexpr (qtype != hnswlib::QuantType::NONE) {
          hnsw->trainSQuant(data, nb);
        }
        hnsw->addPoint(data, 0);
#pragma omp parallel for schedule(dynamic) num_threads(32)
        for (int i = 1; i < nb; ++i) {
            hnsw->addPoint(data + i * dim, i);
        }
    }
    else{
        if constexpr (qtype != hnswlib::QuantType::NONE) {
          hnsw->trainSQuant(data, nb);
        }
#pragma omp parallel for schedule(dynamic) num_threads(32)
        for (int i = miss_end; i < nb; ++i) {
            hnsw->addPoint(data + i * dim, i);
        }
    }
    final_graph.init(nb, 2 * M);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      auto internal_id = hnsw->label_lookup_[i];
      int *edges = (int *)hnsw->get_linklist0(internal_id);
      for (int j = 1; j <= edges[0]; ++j) {
        int external_id = hnsw->getExternalLabel(edges[j]);
        final_graph.at(i, j - 1) = external_id;
        final_graph.at_time(i, j - 1) = timestamps[external_id];
      }
      // NOT in use! sort the edges by timestamp, to find valid position of edges while in-filtering
      // std::sort(ext_ids.begin(), ext_ids.end(), [&](int a, int b) {
      //   return timestamps[a] < timestamps[b];
      // });
    }
    auto initializer = std::make_unique<HNSWInitializer>(nb, M);
    initializer->ep = hnsw->getExternalLabel(hnsw->enterpoint_node_);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      auto internal_id = hnsw->label_lookup_[i];
      int level = hnsw->element_levels_[internal_id];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i].assign(level * M, -1);
        for (int j = 1; j <= level; ++j) {
          int *edges = (int *)hnsw->get_linklist(internal_id, j);
          for (int k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = hnsw->getExternalLabel(edges[k]);
          }
        }
      }
    }
    final_graph.initializer = std::move(initializer);
  }

  Graph<int> GetGraph() override { return final_graph; }
};
} // namespace glass