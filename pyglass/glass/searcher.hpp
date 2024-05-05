#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <functional>
#include <iostream>

#include "glass/common.hpp"
#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/quant.hpp"
#include "glass/utils.hpp"

namespace glass {

struct Filter {
  float v, l, r;
  std::function<bool(float, float)> check;

  Filter(float *query_meta)
      : v(query_meta[1]), l(query_meta[2]), r(query_meta[3]) {
    if (query_meta[0] == 0) {
      check = [&](float c, float t) -> bool { return true; };
    } else if (query_meta[0] == 1) {
      check = [&](float c, float t) -> bool { return c == v; };
    } else if (query_meta[0] == 2) {
      check = [&](float c, float t) -> bool { return l <= t && t <= r; };
    } else if (query_meta[0] == 3) {
      check = [&](float c, float t) -> bool {
        return c == v && l <= t && t <= r;
      };
    } else {
      std::cout << "type not in (0,1,2,3)!\n";
      exit(1);
    }
  }
};

struct SearcherBase {
  virtual void SetData(const float *data, float* labels, float* timestamps, int n, int dim, float alpha = 0.0f) = 0;
  virtual void Optimize(int num_threads = 0) = 0;
  virtual void Search(const float *q, int k, int *dst) const = 0;
  virtual void SearchFilter(const float *q, int k,int refine, int *dst, Filter *filter) = 0;
  virtual void SearchFilterSubTime(const float *q, int k,int refine, int *dst, float *dists, Filter *filter) = 0;
  virtual void SearchRangeFilter(const float *q, int k,int refine, int *dst, const std::pair<float, float> query_bound, const std::pair<int, int> id_bound) = 0;
  virtual void SearchCategoryRange(const float* q, int k,int refine, int *dst, const std::pair<float, float> query_bound, const std::pair<int, int> id_bound) = 0;
  virtual void SearchRangeFilterSubTime(const float* q, int k,int refine, int *dst, float* dists, const std::pair<float, float> query_bound, const std::pair<int, int> id_bound) = 0;
  virtual void SetEf(int ef) = 0;
  virtual ~SearcherBase() = default;
};

template <typename Quantizer> struct Searcher : public SearcherBase {

  int d;
  int nb;
  Graph<int> graph;
  Quantizer quant;

  float *labels;
  float *timestamps;

  // Search parameters
  int ef = 32;

  // Memory prefetch parameters
  int po = 5;
  int pl = 2;
  int po_nofil=5;

  // Optimization parameters
  constexpr static int kOptimizePoints = 1000;
  constexpr static int kTryPos = 10;
  constexpr static int kTryPls = 5;
  constexpr static int kTryK = 10;
  int sample_points_num;
  std::vector<float> optimize_queries;
  // TEST: not improve recall
  // LRUCache<uint64_t, int> cache;
  const int graph_po;

  Searcher(const Graph<int> &graph) : graph(graph), graph_po(graph.K / 16) {
  }

  Searcher(Graph<int> &&graph) noexcept : graph(std::move(graph)), graph_po(graph.K / 16) {
  }

  void SetData(const float *data, float* labels, float* timestamps, int n, int dim, float alpha) override {
    this->nb = n;
    this->d = dim;
    quant = Quantizer(d);
    quant.train(data, n);

    this->labels = labels;
    this->timestamps = timestamps;
  }

  void SetEf(int ef) override { this->ef = ef; }

  void Optimize(int num_threads = 0) override {
    if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
    }
    std::vector<int> try_pos(std::min(kTryPos, graph.K));
    std::vector<int> try_pls(
        std::min(kTryPls, (int)upper_div(quant.code_size, 64)));
    std::iota(try_pos.begin(), try_pos.end(), 1);
    std::iota(try_pls.begin(), try_pls.end(), 1);
    std::vector<int> dummy_dst(kTryK);
    printf("=============Start optimization=============\n");
    { // warmup
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
      for (int i = 0; i < sample_points_num; ++i) {
        Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
      }
    }

    float min_ela = std::numeric_limits<float>::max();
    int best_po = 0, best_pl = 0;
    for (auto try_po : try_pos) {
      for (auto try_pl : try_pls) {
        this->po = try_po;
        this->pl = try_pl;
        auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < sample_points_num; ++i) {
          Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
        }

        auto ed = std::chrono::high_resolution_clock::now();
        auto ela = std::chrono::duration<double>(ed - st).count();
        if (ela < min_ela) {
          min_ela = ela;
          best_po = try_po;
          best_pl = try_pl;
        }
      }
    }
    this->po = 1;
    this->pl = 1;
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < sample_points_num; ++i) {
      Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
    }
    auto ed = std::chrono::high_resolution_clock::now();
    float baseline_ela = std::chrono::duration<double>(ed - st).count();
    printf("settint best po = %d, best pl = %d\n"
           "gaining %.2f%% performance improvement\n============="
           "Done optimization=============\n",
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
    this->po = best_po;
    this->pl = best_pl;
  }

  void Search(const float *q, int k, int *dst) const override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
    graph.initialize_search(pool, computer);
    SearchImpl(pool, computer);
    quant.reorder(pool, q, dst, k);
  }

  //todo:search here.
  template <typename Pool, typename Computer>
  void SearchImpl(Pool &pool, const Computer &computer) const {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) {
          continue;
        }
        pool.vis.set(v);
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
      }
    }
  }

  // get G[u].size() neighbors of u that pass the filter by bfs
  inline std::vector<unsigned> bfs(unsigned u, Filter *filter) {
    std::vector<unsigned> cand;
    std::vector<bool> visited(nb);
    std::queue<unsigned> q;
    q.push(u);
    visited[u] = true;
    while (!q.empty()) {
      unsigned cur = q.front();
      q.pop();
      for (size_t i = 0; i < graph.K; i++) {
        int next = graph.at(cur, i);
        if (next == -1) break;
        if (!visited[next]) {
          if (filter->check(labels[next], timestamps[next])) {
            cand.push_back(next);
            if (cand.size() >= 32) break;
          }
          q.push(next);
          visited[next] = true;
        }
      }
      if (cand.size() >= 32) break;
    }
    return cand;
  }

  // get G[u].size() neighbors of u that pass the filter by 2-hop neighbors
  inline std::vector<unsigned> two_hop(unsigned u, Filter *filter) {
    std::vector<unsigned> cand;
    std::vector<bool> visited(nb);
    visited[u] = true;
    // 1-hop neighbors
    for (size_t i = 0; i < graph.K; i++) {
      int nei = graph.at(u, i);
      if (nei == -1) break;
      if (filter->check(labels[nei], timestamps[nei])) {
        cand.push_back(nei);
        if (cand.size() >= 32) break;
      }
      visited[nei] = true;
    }
    // 2-hop neighbors
    for (size_t i = 0; i < graph.K; i++) {
      int nei = graph.at(u, i);
      if (nei == -1) break;
      for (size_t j = 0; j < graph.K; j++) {
        int nei_2 = graph.at(nei, j);
        if (nei_2 == -1) break;
        if (!visited[nei_2]) {
          if (filter->check(labels[nei_2], timestamps[nei_2])) {
            cand.push_back(nei_2);
            if (cand.size() >= 32) break;
          }
          visited[nei_2] = true;
        }
      }
      if (cand.size() >= 32) break;
    }
    return cand;
  }

  void SearchFilter(const float *q, int k,int refine, int *dst, Filter *filter) override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
    graph.initialize_search(pool, computer);
    SearchFilterImpl(pool, computer, filter);
    quant.reorder(pool, q, dst, refine);
  }

  template <typename Pool, typename Computer>
  void SearchFilterImpl(Pool &pool, const Computer &computer, Filter *filter) {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po_nofil; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      int count1 = 0, count2 = 0; // count2: number of non-visited 1-hop neighbors
      for (int i = 0; i < graph.K; ++i) {
        if (count1 > count2) break;
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po_nofil < graph.K && graph.at(u, i + po_nofil) != -1) {
          int to = graph.at(u, i + po_nofil);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) continue;
        count2++;
        pool.vis.set(v);
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
        count1++;
      }
    }
  }

  void SearchCategoryRange(const float* q, int k,int refine, int *dst, const std::pair<float, float> query_bound, const std::pair<int, int> id_bound) override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
    // choose entry point hoping to satisfy range filter
    int num_ep = 10;
      int lbound = id_bound.first;
      int interval = (id_bound.second - id_bound.first) / num_ep;
      for (size_t i = 0; i < num_ep; ++i) {
        int point = lbound + interval * i;
        pool.insert(point, computer(point));
        pool.vis.set(point);
      }
    // graph.initialize_search(pool, computer);
    SearchCategoryRangeImpl(pool, computer, query_bound);
    quant.reorder(pool, q, dst, refine);
  }

  template <typename Pool, typename Computer>
  void SearchCategoryRangeImpl(Pool &pool, const Computer &computer, const std::pair<float, float> query_bound) {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      // count1: number of non-visited and satisify filter neighbors (both 1-hop 2-hop)
      // count2: number of non-visited 1-hop neighbors
      int count1 = 0, count2 = 0; 
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) continue;
        count2++;
        pool.vis.set(v);
        float t = graph.at_time(u, i);
        if (t < query_bound.first || t > query_bound.second) continue;
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
        count1++;
      }
      for (size_t i = 0; i < graph.K; i++) {
        if (count1 >= count2) break;
        int nei = graph.at(u, i);
        if (nei == -1) break;
        if (pool.vis1.get(nei)) continue;
        pool.vis1.set(nei);
        graph.prefetch(nei, graph_po); // prefetch edges
        for (size_t j = 0; j < graph.K; j++) {
          int nei_2 = graph.at(nei, j);
          if (nei_2 == -1) break;
          if (j + po < graph.K && graph.at(nei, j + po) != -1) {
            int to = graph.at(nei, j + po);
            computer.prefetch(to, pl);
          }
          if (pool.vis.get(nei_2)) continue;
          pool.vis.set(nei_2);
          float t = graph.at_time(nei, j);
          if (t < query_bound.first || t > query_bound.second) continue;
          auto cur_dist = computer(nei_2);
          pool.insert(nei_2, cur_dist);
          count1++;
          if (count1 >= count2) break;
        }
      }
    }
  }

  // for timestamp sub-index
 void SearchFilterSubTime(const float *q, int k,int refine, int *dst, float* dists, Filter *filter) override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
    graph.initialize_search(pool, computer);
    SearchFilterSubTimeImpl(pool, computer, filter);
    quant.reorder_dist(pool, q, dst, dists, refine);
  }

  template <typename Pool, typename Computer>
  void SearchFilterSubTimeImpl(Pool &pool, const Computer &computer, Filter *filter) {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      int count1 = 0, count2 = 0; // count2: number of non-visited 1-hop neighbors
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) continue;
        count2++;
        pool.vis.set(v);
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
        count1++;
      }
    }
  }
  // for timestamp sub-index
void SearchRangeFilterSubTime(const float* q, int k,int refine, int *dst, float* dists, const std::pair<float, float> query_bound, const std::pair<int, int> id_bound) override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
      // TEST: can't improve recall
      // choose entry point hoping to satisfy range filter
      float sel = (float)(id_bound.second - id_bound.first) / nb;
      if (sel < 1) {
        int num_ep = 10;
        int lbound = id_bound.first;
        int interval = (id_bound.second - id_bound.first) / num_ep;
        for (size_t i = 0; i < num_ep; ++i) {
          int point = lbound + interval * i;
          pool.insert(point, computer(point));
          pool.vis.set(point);
        }
      } else {
      }
    SearchRangeFilterSubTimeImpl(pool, computer, query_bound);
    quant.reorder_dist(pool, q, dst, dists, refine);
  }

  template <typename Pool, typename Computer>
  void SearchRangeFilterSubTimeImpl(Pool &pool, const Computer &computer, const std::pair<float, float> query_bound) {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      int count1 = 0, count2 = 0; // count2: number of non-visited 1-hop neighbors
      int rf_thr = 2;
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) continue;
        count2++;
        pool.vis.set(v);
        float t = graph.at_time(u, i);
        if (t < query_bound.first || t > query_bound.second) continue;
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
        count1++;
      }
      for (size_t i = 0; i < graph.K; i++) {
        if (count1 >= rf_thr * count2) break;
        int nei = graph.at(u, i);
        if (nei == -1) break;
        if (pool.vis1.get(nei)) continue;
        pool.vis1.set(nei);
        graph.prefetch(nei, graph_po); // prefetch edges
        for (size_t j = 0; j < graph.K; j++) {
          int nei_2 = graph.at(nei, j);
          if (nei_2 == -1) break;
          if (j + po < graph.K && graph.at(nei, j + po) != -1) {
            int to = graph.at(nei, j + po);
            computer.prefetch(to, pl);
          }
          if (pool.vis.get(nei_2)) continue;
          pool.vis.set(nei_2);
          float t = graph.at_time(nei, j);
          if (t < query_bound.first || t > query_bound.second) continue;
          auto cur_dist = computer(nei_2);
          pool.insert(nei_2, cur_dist);
          count1++;
          if (count1 >= rf_thr * count2) break;
        }
      }
    }
  }

  void SearchRangeFilter(const float* q, int k,int refine, int *dst, const std::pair<float, float> query_bound, const std::pair<int, int> id_bound) override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
      // choose entry point hoping to satisfy range filter
      float sel = (float)(id_bound.second - id_bound.first) / nb;
      if (sel < 1) {
        int num_ep = 10;
        int lbound = id_bound.first;
        int interval = (id_bound.second - id_bound.first) / num_ep;
        for (size_t i = 0; i < num_ep; ++i) {
          int point = lbound + interval * i;
          pool.insert(point, computer(point));
          pool.vis.set(point);
        }
      } else {
          graph.initialize_search(pool, computer);
      }
    SearchRangeFilterImpl(pool, computer, query_bound);
    quant.reorder(pool, q, dst, refine);
  }

  template <typename Pool, typename Computer>
  void SearchRangeFilterImpl(Pool &pool, const Computer &computer, const std::pair<float, float> query_bound) {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      int count1 = 0, count2 = 0; // count2: number of non-visited 1-hop neighbors
      int rf_thr = 2;
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) continue;
        count2++;
        pool.vis.set(v);
        float t = graph.at_time(u, i);
        if (t < query_bound.first || t > query_bound.second) continue;
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
        count1++;
      }
      for (size_t i = 0; i < graph.K; i++) {
        if (count1 >= rf_thr * count2) break;
        int nei = graph.at(u, i);
        if (nei == -1) break;
        if (pool.vis1.get(nei)) continue;
        pool.vis1.set(nei);
        graph.prefetch(nei, graph_po); // prefetch edges
        for (size_t j = 0; j < graph.K; j++) {
          int nei_2 = graph.at(nei, j);
          if (nei_2 == -1) break;
          if (j + po < graph.K && graph.at(nei, j + po) != -1) {
            int to = graph.at(nei, j + po);
            computer.prefetch(to, pl);
          }
          if (pool.vis.get(nei_2)) continue;
          pool.vis.set(nei_2);
          float t = graph.at_time(nei, j);
          if (t < query_bound.first || t > query_bound.second) continue;
          auto cur_dist = computer(nei_2);
          pool.insert(nei_2, cur_dist);
          count1++;
          if (count1 >= rf_thr * count2) break;
        }
      }
    }
  }
};

inline std::unique_ptr<SearcherBase> create_searcher(Graph<int> &&graph,
                                                     const std::string &metric,
                                                     int level = 1) {
  auto m = metric_map[metric];
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ8SymmetricQuantizer<Metric::L2>>>(std::move(graph));
    }
  else {
    printf("Quantizer type not supported\n");
    return nullptr;
  }
}

} // namespace glass
