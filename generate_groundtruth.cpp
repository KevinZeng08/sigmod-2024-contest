#include "utils.h"
#include "io.h"

int main(int argc, char **argv) {
    // 1m data
    // const std::string source_path = "contest-data-release-1m.bin";
    // const std::string query_path = "contest-queries-release-1m.bin";
    // const std::string gt_path = "contest-gt-release-1m.bin";

    // 10m data
//    const std::string source_path = "contest-data-release-10m.bin";
//    const std::string query_path = "contest-queries-release-10m.bin";
//    const std::string gt_path = "contest-gt-release-10m.bin";

    const std::string source_path = "/home/cactus/sigmod-contest-2024/data/dummy-data.bin";
    const std::string query_path = "/home/cactus/sigmod-contest-2024/data/dummy-queries.bin";
    const std::string gt_path = "/home/cactus/sigmod-contest-2024/data/dummy-gt.bin";

    uint32_t num_data_dimensions = 102;

    // Read data points
    vector<vector<float>> nodes;
    ReadBin(source_path, num_data_dimensions, nodes);
    std::cout << nodes.size() << "\n";
    // Read queries
    uint32_t num_query_dimensions = num_data_dimensions + 2;
    vector<vector<float>> queries;
    ReadBin(query_path, num_query_dimensions, queries);

    // Generate ground truth and save to disk
    vector<vector<uint32_t>> gt;
    GetGroundTruth(nodes, queries, gt);
    SaveGroundTruth(gt, gt_path);

    // Read ground truth
    const int K = 100;
    vector<vector<uint32_t>> knns;
    ReadGroundTruth(gt_path, K, knns);

    // Calculate recall
    float recall = GetKNNRecall(knns, gt);
    std::cout << "Recall: " << recall << "\n";

    return 0;
}