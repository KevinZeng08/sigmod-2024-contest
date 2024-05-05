#pragma once
#include "glass/graph.hpp"
#include "glass/searcher.hpp"
#include "glass/hnsw/hnsw.hpp"
#include "glass/quant/sq8_quant.hpp"
#include "glass/memory.hpp"
#include "utils.h"
struct BuildParams
{
    int M = 28;               // number of neighbors
    int efConstruction = 200; // build candidate size
};

struct SearchParams {
    int K = 100; // number of KNN neighbors
    int ef = 1200; // search candidiate size
    // ------------------- type0 -------------------
    int ef0_full = 425;
    // ------------------- type1 -------------------
    int ef1min = 1800; // min number of candidates for type1
    int ef1max = 2500; //max number of candidates for type1
    int ef1 = 200;
    // ------------------- type2 -------------------
    int ef2_full_large = 480; // number of candidates for type2 large selectivity, full graph search
    int ef2_full_small = 780; // number of candidates for type2 small selectivity, full graph search
    int ef2_full_medium = 630; // number of candidates for type2 medium selectivity, full graph search
    int ef2_medium_small = 1180; // number of candidates for type2 small selectivity, filtered graph search
    int ef2_medium_large = 680; // number of candidates for type2 large selectivity, filtered graph search
    int ef2_medium_medium = 780; // number of candidates for type2 medium selectivity, filtered graph search
    int ef2_large = 400; // number of candidates for type2_large
    // ------------------- type3 -------------------
    int ef3min =      1800;     //2800;     // number of candidates for type3
    int ef3max =      2800;     //3200;
};


struct HybridGraph
{
    HybridGraph() = default;
    ~HybridGraph() = default;

    void Build(float *vecs, float *labels, float *timestamps, int nb, const BuildParams &params);

    void BuildCategoryIndex(float *vecs, float *labels, float *timestamps, int nb, const BuildParams &params);

    void BuildTimestampIndex(float *vecs, float *labels, float *timestamps, int nb, const BuildParams &params);

    void BatchSearch(float *base, float *base_by_time, float *base_by_full_time, float *vecs, float *metas, int nq, uint32_t *ids, const SearchParams &params);

    void SortDataset(glass::SQ8SymmetricQuantizer<glass::Metric::L2> &quant, float *labels, float *timestamps, int nb);

    // intervals: (timestamp identifier) -> (range filter type, (filtered start id, number of vectors))
    void SplitInterval(const std::pair<float, float> &range, std::unordered_map<int, std::pair<RFType, std::pair<int, int>>> &intervals,
                        float bf_thr = 0.2);

    void TestSplitInterval();

    uint32_t dim = 100; // vector dimension
    std::string metric = "L2";

    // sorted query ids by <query_type, label, timestamp>
    std::vector<uint32_t> sorted_ids;

    // sorted ids by label + timestamp to get selectivity
    std::vector<MetaData> sorted_both;
    // sorted ids by timestamp to get selectivity
    std::vector<MetaData> sorted_timestamp;

    std::vector<char, glass::align_alloc<char>> codes_both;
    std::vector<char, glass::align_alloc<char>> codes_time;
    std::vector<float> mi, dif;

    // symmetric SQ8
    std::vector<char, glass::align_alloc<char>> codes_query;
    float alpha = 0.0f;

    // bruteforce
    float bf_thr = 0.05;
    float bf_thr3 = 0.08;
    uint32_t bf_refine_k = 140; // number of neighbors for bruteforce refine
    uint32_t graph_full_refine_k = 150;
    uint32_t graph_part_refine_k = 150;
    uint32_t maxc_size,minc_size;
    float ef1_category_slope,ef3_category_slope;
    uint32_t global_graph_refine_k = 150; // number of candidates for refine of sub-index graph search

    // pure ANN search
    std::unique_ptr<glass::Builder> index = nullptr;
    std::unique_ptr<glass::SearcherBase> searcher = nullptr;

    // categorical hybrid search
    // sorted base vector ids by category sorted id->real id
    std::vector<uint32_t> sorted_base_ids;
    std::unordered_map<int32_t, std::pair<uint32_t, uint32_t>> category_map; // category -> (start id, number of vectors)
    std::unordered_map<int32_t, std::unique_ptr<glass::Builder>> category_index; // category -> index
    std::unordered_map<int32_t, std::unique_ptr<glass::SearcherBase>> category_searcher; // category -> searcher
    float cat_graph_thr = 0.05; // threshold for building category sub-graph
    uint32_t maxc_id; // id of the largest category
    std::unordered_map<int32_t, std::vector<int>> category_query; // category -> query ids

    // timestamp range search
    // sorted base vector ids by timestamp, for range filter search
    std::vector<uint32_t> sorted_base_ids_by_time;
    std::vector<uint32_t> sorted_base_ids_by_full_time;
    std::unordered_map<int32_t, std::pair<uint32_t, uint32_t>> timestamp_map; // timestamp -> (start id, number of vectors)
    std::unordered_map<int32_t, std::unique_ptr<glass::Builder>> timestamp_index; // timestamp -> index
    std::unordered_map<int32_t, std::unique_ptr<glass::SearcherBase>> timestamp_searcher; // timestamp -> searcher

//    std::unordered_map<int32_t, std::vector<std::pair<int, RFType>>> timestamp_query; // timestamp -> query ids
    std::vector<std::pair<int, RFType>>timestamp_query[10]; // timestamp -> query ids
};