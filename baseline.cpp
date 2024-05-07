/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <chrono>
#include <memory>
#include "io.h"
#include "hybrid_graph.h"
#include "hybrid_graph.cpp"

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
using std::pair;

using namespace glass;

int main(int argc, char **argv)
{
#define BUILD_INDEX
   string source_path = "dummy-data.bin";
   string query_path = "dummy-queries.bin";
   string knn_save_path = "output.bin";
  // 1m
  //  string source_path = "contest-data-release-1m.bin";
  //  string query_path = "contest-queries-release-1m.bin";
  //  string gt_path = "contest-gt-release-1m.bin";

  // 10m
  // string source_path = "/dataset/sigmod2024/large/contest-data-release-10m.bin";
  // string query_path = "/dataset/sigmod2024/large/contest-queries-release-10m.bin";
  // string gt_path = "/dataset/sigmod2024/large/contest-gt-release-10m.bin";

  auto start = std::chrono::high_resolution_clock::now();
  auto total_start = start;
  // Read data and query points
  vector<float> base_vecs, base_vecs_by_time, base_vecs_by_full_time;
  vector<float> base_labels, base_labels_by_time, base_labels_by_full_time, base_timestamps, base_timestamps_by_time, base_timestamps_by_full_time;
  vector<float> query_vecs, query_metas;
  vector<uint32_t> sorted_ids;
  vector<uint32_t> sorted_base_ids, sorted_base_ids_by_time, sorted_base_ids_by_full_time;
  unordered_map<int32_t, pair<uint32_t, uint32_t>> category_map, timestamp_map;
  vector<MetaData> sorted_both, sorted_timestamp;
  unordered_map<int32_t, vector<int>> category_query;

  uint32_t nb, nq;
  int maxc_id;
    int32_t max_count = 0;
    int32_t min_count = 1e9;
  // Read base data and query data
  ReadSortedBaseTimestamp(source_path, nb, base_vecs, base_labels, base_timestamps, sorted_base_ids, category_map,
              base_vecs_by_time, base_labels_by_time, base_timestamps_by_time, sorted_base_ids_by_time, maxc_id,
              base_vecs_by_full_time, base_labels_by_full_time, base_timestamps_by_full_time, sorted_base_ids_by_full_time, timestamp_map,
              max_count,min_count);
  ReadSortedQuery(query_path, nq, query_vecs, query_metas, sorted_ids, category_query, category_map);
  auto end = std::chrono::high_resolution_clock::now();
  cout << "ReadData time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << endl;

  vector<vector<uint32_t>> knn_results; // for saving knn results

  const int K = 100; // To find 100-NN

  HybridGraph h_graph;
  h_graph.sorted_ids = std::move(sorted_ids);
  h_graph.sorted_base_ids = std::move(sorted_base_ids);
  h_graph.category_map = std::move(category_map);
  h_graph.sorted_base_ids_by_time = std::move(sorted_base_ids_by_time);
  h_graph.bf_thr = 0.045;
  h_graph.cat_graph_thr = h_graph.bf_thr; // same as bf_thr to avoid redundant category index build
  h_graph.maxc_id = maxc_id;
  h_graph.timestamp_map = std::move(timestamp_map);
  h_graph.sorted_base_ids_by_full_time = std::move(sorted_base_ids_by_full_time);
  h_graph.maxc_size=max_count;
  h_graph.minc_size=min_count;
  h_graph.category_query = std::move(category_query);
#ifdef BUILD_INDEX
  // Build Index
  BuildParams build_params;
  start = std::chrono::high_resolution_clock::now();
  // Build the largest category index
  h_graph.Build(base_vecs_by_time.data(), base_labels_by_time.data(), base_timestamps_by_time.data(), nb, build_params);
  // Build category index with vectors and attributes sorted by category
  h_graph.BuildCategoryIndex(base_vecs.data(), base_labels.data(), base_timestamps.data(), nb, build_params);
  // Build timestamp index with vectors sorted by timestamp
  h_graph.BuildTimestampIndex(base_vecs_by_full_time.data(), base_labels_by_full_time.data(), base_timestamps_by_full_time.data(), nb, build_params);
  end = std::chrono::high_resolution_clock::now();
  cout << "Build time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << endl;
#endif


  // Search Index
  vector<uint32_t> ids;
  ids.resize(nq * K);

  uint32_t dim = h_graph.dim;

  int level = 3; // 0: float, 1: SQ8, 2: SQ4, 3: Symmetric SQ8
  glass::Graph<int> graph;
  h_graph.searcher = std::unique_ptr<glass::SearcherBase>(
      glass::create_searcher(std::move(graph), h_graph.metric, level));
  h_graph.searcher->SetData(base_vecs_by_time.data(), base_labels_by_time.data(), base_timestamps_by_time.data(), nb, dim);

  // create category index searcher
  vector<int> categories;
  for (auto& [category, p] : h_graph.category_map) {
    if (p.second < h_graph.cat_graph_thr * nb) {
      continue;
    }
    categories.push_back(category);
  }
  for (int v : categories) {
    auto &c_index = h_graph.category_index[v];
    h_graph.category_searcher[v] = std::unique_ptr<glass::SearcherBase>(
      glass::create_searcher(std::move(((glass::HNSW *)c_index.get())->final_graph), h_graph.metric, level));
      auto& searcher = h_graph.category_searcher[v];
      auto& [start_id, n] = h_graph.category_map[v];
    searcher->SetData(base_vecs.data() + start_id * dim, base_labels.data() + start_id, base_timestamps.data() + start_id, n, dim);
  }

  // create timestamp searcher
  vector<int> timestamps;
  for (auto& [t, p] : h_graph.timestamp_map) {
    if (p.second == 0) continue;
    timestamps.push_back(t);
  }
  for (int t : timestamps) {
    auto &t_index = h_graph.timestamp_index[t];
    h_graph.timestamp_searcher[t] = std::unique_ptr<glass::SearcherBase>(
      glass::create_searcher(std::move(((glass::HNSW *)t_index.get())->final_graph), h_graph.metric, level));
    auto& searcher = h_graph.timestamp_searcher[t];
    auto& [start_id, n] = h_graph.timestamp_map[t];
    searcher->SetData(base_vecs_by_full_time.data() + start_id * dim, base_labels_by_full_time.data() + start_id, base_timestamps_by_full_time.data() + start_id, n, dim);
  }

  // SQ8 symmetric
  auto &quant = ((Searcher<SQ8SymmetricQuantizer<Metric::L2>>*)h_graph.searcher.get())->quant;
  h_graph.SortDataset(quant, base_labels_by_time.data(), base_timestamps_by_time.data(), nb);

  SearchParams search_params;
  start = std::chrono::high_resolution_clock::now();
  h_graph.BatchSearch(base_vecs.data(), base_vecs_by_time.data(), base_vecs_by_full_time.data(), query_vecs.data(), query_metas.data(), nq, ids.data(), search_params);
  end = std::chrono::high_resolution_clock::now();
  cout << "Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;

  // Get KNN results
  for (uint32_t i = 0; i < nq; i++)
  {
    vector<uint32_t> knn;
    for (uint32_t j = 0; j < K; j++)
    {
      knn.push_back(ids[i * K + j]);
    }
    knn_results.push_back(knn);
  }
  // Save KNN results
  SaveKNN(knn_results, "output.bin");

  end = std::chrono::high_resolution_clock::now();
  cout << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(end - total_start).count() << "s" << endl;

  return 0;
}