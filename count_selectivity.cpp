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
#include "utils.h"
#include "hybrid_graph.h"

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;
using std::vector;

int main(int argc, char **argv)
{
  // Submission use
  // string source_path = "dummy-data.bin";
  // string query_path = "dummy-queries.bin";
  // string knn_save_path = "output.bin";

  // 1m
  // string source_path = "/dataset/sigmod2024/medium/contest-data-release-1m.bin";
  // string query_path = "/dataset/sigmod2024/medium/contest-queries-release-1m.bin";
  // string gt_path = "/dataset/sigmod2024/medium/contest-gt-release-1m.bin";

  // 10m
  string source_path = "/dataset/sigmod2024/large/contest-data-release-10m.bin";
  string query_path = "/dataset/sigmod2024/large/contest-queries-release-10m.bin";
  string gt_path = "/dataset/sigmod2024/large/contest-gt-release-10m.bin";

  auto start = std::chrono::high_resolution_clock::now();
  // Read data and query points
  // TODO aligned alloc for vectors
  vector<float> base_vecs, base_timestamps;
  vector<float> base_labels;
  vector<float> query_vecs, query_metas;
  uint32_t nb, nq;
  ReadBase(source_path, nb, base_vecs, base_labels, base_timestamps);
  ReadQuery(query_path, nq, query_vecs, query_metas);
  // TODO query reordering: 将type相同 categorical attribute相同 或 timestamp attribute交集最大的query放在一起
  cout << "# of base vectors: " << nb << endl;
  cout << "# of query vectors: " << nq << endl;
  auto end = std::chrono::high_resolution_clock::now();
  cout << "ReadData time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << endl;

  vector<vector<uint32_t>> knn_results; // for saving knn results

  const int K = 100; // To find 100-NN

  // Build Index
  HybridGraph h_graph;
  h_graph.SortDataset(base_labels.data(), base_timestamps.data(), nb);
  std::vector<float> selectivity(nq);
  std::vector<float> types(nq);
  for (size_t i = 0; i < nq; i++) {
    float *query_meta = query_metas.data() + i * 4;
    uint32_t type = query_meta[0];
    uint32_t v = query_meta[1];
    float l = query_meta[2];
    float r = query_meta[3];
    types[i] = type;
    if (type == 0) {
        selectivity[i] = 1;
    } else if (type == 1) {
      auto cmp = [](std::pair<uint32_t, float> p1, std::pair<uint32_t, float> p2) {
        return p1.first < p2.first;
      };
      auto s = std::lower_bound(h_graph.sorted_both.begin(), h_graph.sorted_both.end(), std::make_pair(v, l), cmp);
      auto e = std::upper_bound(h_graph.sorted_both.begin(), h_graph.sorted_both.end(), std::make_pair(v, r), cmp);
      selectivity[i] = ((float)(e - s)) / nb;
    } else if (type == 3) {
      auto s = std::lower_bound(h_graph.sorted_both.begin(), h_graph.sorted_both.end(), std::make_pair(v, l));
      auto e = std::upper_bound(h_graph.sorted_both.begin(), h_graph.sorted_both.end(), std::make_pair(v, r));
      selectivity[i] = ((float)(e - s)) / nb;
    } else if (type == 2) {
      auto s = std::lower_bound(h_graph.sorted_timestamp.begin(), h_graph.sorted_timestamp.end(), l);
      auto e = std::upper_bound(h_graph.sorted_timestamp.begin(), h_graph.sorted_timestamp.end(), r);
      selectivity[i] = ((float)(e - s)) / nb;
    }
  }

  std::ofstream out("types.bin", std::ios::binary);
  out.write((char*)types.data(), nq * 4);
  std::ofstream out1("selectivity.bin", std::ios::binary);
  out1.write((char*)selectivity.data(), nq * 4);

  return 0;
}