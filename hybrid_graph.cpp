#include <omp.h>
#include <iostream>
#include <numeric>

#include "hybrid_graph.h"
#include "bruteforce.h"
#include "timer.h"

using namespace glass;
using std::cout;
using std::endl;
using PIIRFType = std::pair<RFType, std::pair<int, int>>;

auto cmp_by_both = [](MetaData a, MetaData b)
{
    return (a.label < b.label) || (a.label == b.label && a.timestamp < b.timestamp);
};

auto cmp_by_label = [](MetaData a, MetaData b)
{
    return a.label < b.label;
};

auto cmp_by_time = [](MetaData a, MetaData b)
{
    return a.timestamp < b.timestamp;
};

void HybridGraph::Build(float *vecs, float *labels, float *timestamps, int nb, const BuildParams &params)
{
    // build the biggest category graph and the complete graph
    // auto start = std::chrono::high_resolution_clock::now();
    auto category = maxc_id;
    auto p = category_map[maxc_id];
    int n = p.second;
    category_index[category] = std::unique_ptr<glass::Builder>(
        (glass::Builder *)new glass::HNSW(dim, metric, 2 * params.M, params.efConstruction));
    auto &c_index = category_index[category];
    c_index->Build(vecs, timestamps, n, n);
    // auto end = std::chrono::high_resolution_clock::now();
    // cout << "Category[" << category << "]graph with " << n << " points"
    //      << ", building cost: "
    //      << std::chrono::duration<double>(end - start).count() << "s" << endl;
    // ((glass::HNSW *)c_index.get())->final_graph.save(std::to_string(category) + ".hnsw");
}

void HybridGraph::BuildCategoryIndex(float *vecs, float *labels, float *timestamps, int nb, const BuildParams &params)
{
    // Build category index
    int min_cnt = cat_graph_thr * nb;
    for (auto &[category, p] : category_map)
    {
        // auto start = std::chrono::high_resolution_clock::now();
        int n = p.second;
        if (n < min_cnt || maxc_id == category)
        {
            continue;
        }
        uint32_t start_id = p.first;
        float *data = vecs + start_id * dim;
        float *time_data = timestamps + start_id;
        category_index[category] = std::unique_ptr<glass::Builder>(
            (glass::Builder *)new glass::HNSW(dim, metric, 2 * params.M, params.efConstruction));
        auto &c_index = category_index[category];
        c_index->Build(data, time_data, n, n);
        // auto end = std::chrono::high_resolution_clock::now();
        // cout << "Category[" << category << "]graph with " << n << " points, building cost: " << std::chrono::duration<double>(end - start).count() << "s" << endl;
        // ((glass::HNSW *)c_index.get())->final_graph.save(std::to_string(category) + ".hnsw");
    }
}

void HybridGraph::BuildTimestampIndex(float *vecs, float *labels, float *timestamps, int nb, const BuildParams &params)
{
    for (auto &[t_start, p] : timestamp_map)
    {
        // auto start = std::chrono::high_resolution_clock::now();
        int n = p.second;
        uint32_t start_id = p.first;
        float *data = vecs + start_id * dim;
        float *time_data = timestamps + start_id;
        timestamp_index[t_start] = std::unique_ptr<glass::Builder>(
            (glass::Builder *)new glass::HNSW(dim, metric, 2 * params.M, params.efConstruction));
        auto &t_index = timestamp_index[t_start];
        t_index->Build(data, time_data, n, n);
        // auto end = std::chrono::high_resolution_clock::now();
        // cout << "Timestamp[" << t_start << "]graph with " << n << " points, building cost: " << std::chrono::duration<double>(end - start).count() << "s" << endl;
        // ((glass::HNSW *)t_index.get())->final_graph.save("t" + std::to_string(t_start) + ".hnsw");
    }
}

void HybridGraph::SplitInterval(const std::pair<float, float> &range, std::unordered_map<int, PIIRFType> &intervals, float bf_thr)
{
    auto [l, r] = range;
    for (const auto &[t, p] : timestamp_map)
    {
        float ts = (float)t / 10;
        float te = (float)(t + interval) / 10;
        auto [start_id, n] = p;
        if (r < ts || l > te)
        {
            continue;
        }
        // FULL
        if (l <= ts && r >= te)
        {
            intervals[t] = {RFType::FULL, {0, n}};
            continue;
        }
        // selectivity
        float sel = 1.0f;
        float l_bound = std::max(ts, l);
        float r_bound = std::min(te, r);
        float *timestamps = ((Searcher<SQ8SymmetricQuantizer<Metric::L2>> *)timestamp_searcher[t].get())->timestamps;
        auto s = std::lower_bound(timestamps, timestamps + n, l_bound);
        auto e = std::upper_bound(timestamps, timestamps + n, r_bound);
        sel = ((float)(e - s)) / n;
        if (sel < bf_thr) // TODO tunable parameter: threshold for bruteforce of timestamp sub-index
        {
            intervals[t] = {RFType::SMALL, {s - timestamps, e - s}};
        }
        else
        {
            intervals[t] = {RFType::MEDIUM, {s - timestamps, e - s}};
        }
    }

    // print split result
    // printf("[%.4f, %.4f]\n", l, r);
    // for (const auto &[t, p] : intervals)
    // {
    //     printf("([%d, %d], RFType: %d, start: %d, num: %d), ", t, t+1, p.first, p.second.first, p.second.second);
    // }
    // printf("\n");
}

void HybridGraph::TestSplitInterval()
{
    std::unordered_map<int, PIIRFType> intervals;
    SplitInterval(std::make_pair(0.0154, 0.2466), intervals);
    for (const auto &[t, p] : intervals)
    {
        printf("([%d, %d], RFType: %d, num: %d), ", t, t + 1, p.first, p.second.second);
    }
    printf("\n");
}

void HybridGraph::BatchSearch(float *base, float *base_by_time, float *base_by_full_time, float *vecs, float *metas, int nq, uint32_t *ids, const SearchParams &params)
{
    // std::vector<QueryStats> stats(nq);
    // Initialize Searcher
    int k = params.K;
    int nb = 10000000;
    ef1_category_slope=(float)(params.ef1max - params.ef1min)/(maxc_size - minc_size);
    ef3_category_slope=(float)(params.ef3max - params.ef3min)/(maxc_size - minc_size);
    codes_query.resize(nq * 112);
    auto &quant = ((Searcher<SQ8SymmetricQuantizer<Metric::L2>> *)searcher.get())->quant;
    for (size_t i = 0; i < nq; i++)
    {
        float *q = vecs + i * dim;
        quant.encode(q, codes_query.data() + i * 112);
    }

    omp_set_num_threads(32);

    std::vector<float> sels(nq);
    Timer<std::chrono::milliseconds> timer;
    timer.reset();
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < nq; ++i)
    {
        float type = metas[i * 4];
        float v = metas[i * 4 + 1];
        float l = metas[i * 4 + 2];
        float r = metas[i * 4 + 3];
        float sel = 1;
        auto s = sorted_both.begin();
        auto e = sorted_both.end();
        char *filtered_codes;
        if (type == 1)
        {
            s = std::lower_bound(sorted_both.begin(), sorted_both.end(), MetaData(0, v, l), cmp_by_label);
            e = std::upper_bound(sorted_both.begin(), sorted_both.end(), MetaData(0, v, r), cmp_by_label);
            filtered_codes = codes_both.data() + (s - sorted_both.begin()) * 112;
            sel = ((float)(e - s)) / nb;
        }
        else if (type == 2)
        {
            s = std::lower_bound(sorted_timestamp.begin(), sorted_timestamp.end(), MetaData(0, v, l), cmp_by_time);
            e = std::upper_bound(sorted_timestamp.begin(), sorted_timestamp.end(), MetaData(0, v, r), cmp_by_time);
            filtered_codes = codes_time.data() + (s - sorted_timestamp.begin()) * 112;
            sel = ((float)(e - s)) / nb;
        }
        else if (type == 3)
        {
            s = std::lower_bound(sorted_both.begin(), sorted_both.end(), MetaData(0, v, l), cmp_by_both);
            e = std::upper_bound(sorted_both.begin(), sorted_both.end(), MetaData(0, v, r), cmp_by_both);
            filtered_codes = codes_both.data() + (s - sorted_both.begin()) * 112;
            sel = ((float)(e - s)) / nb;
        }
        sels[i] = sel;
        if (sel < bf_thr || (type == 3 && sel < bf_thr3))
        {        
            float *cur_q = vecs + i * dim;
            uint32_t query_id = sorted_ids[i];
            int *id = (int *)ids + query_id * k;
            auto K = std::min<size_t>(bf_refine_k, e - s);
            std::vector<int> tmp(K);

            // Symmetric SQ8
            int8_t *encode_q = (int8_t *)codes_query.data() + i * 112;
            bruteforce(dim, K, tmp.data(), e - s, filtered_codes, alpha, 1, encode_q);
            for (size_t j = 0; j < K; j++)
            {
                tmp[j] = (*(s + tmp[j])).id;
            }
            maxPQIFCS<float>point_dist(k);
            for (size_t j = 0; j < K; j++)
            {
                if (j + 1 < K)
                {
                    glass::mem_prefetch((char *)(base_by_time + tmp[j + 1] * dim), 2);
                }
                point_dist.maybe_pop_emplace(tmp[j], L2SqrRef(cur_q, base_by_time + tmp[j] * dim, dim));
            }
            for (size_t l = 0; l < k; ++l)
            {
                id[k - l - 1] = sorted_base_ids_by_time[point_dist.data_[l + 1].id];
            }
        }
    }

    timer.end();
    std::cout << "Bruteforce search time: " << std::chrono::duration<double>(timer.getElapsedTime()).count() << "s" << std::endl;

    timer.reset();
    Timer<std::chrono::milliseconds> graph_timer;
    graph_timer.reset();
    // search type1 & type3 
    for (auto &[v, q_ids] : category_query)
    {
        auto &searcher = category_searcher[v];
#pragma omp parallel for schedule(dynamic)
        for (int q_id : q_ids)
        {
            float type = metas[q_id * 4];
            float v = metas[q_id * 4 + 1];
            float l = metas[q_id * 4 + 2];
            float r = metas[q_id * 4 + 3];
            float sel = sels[q_id];

            if (type == 1 && sel < bf_thr)
                continue;
            if (type == 3 && sel < bf_thr3)
                continue;

            float *cur_q = vecs + q_id * dim;
            uint32_t query_id = sorted_ids[q_id];
            int *id = (int *)ids + query_id * k;
            int32_t category = (int32_t)v;
            int32_t K;
            if (type == 1)
            {
                K = ceil(params.ef1min + ef1_category_slope * category_map[category].second);
            }
            else
            {
                K = ceil(params.ef3min + ef3_category_slope * category_map[category].second);
            }
            std::vector<int> tmp(graph_full_refine_k);
            uint32_t start_id = category_map[category].first;
            if (type == 1)
            {
                searcher->SearchFilter(cur_q, K, graph_full_refine_k, tmp.data(), nullptr);
            }
            else
            {
                auto s = std::lower_bound(sorted_both.begin(), sorted_both.end(), MetaData(0, v, l), cmp_by_both);
                auto e = std::upper_bound(sorted_both.begin(), sorted_both.end(), MetaData(0, v, r), cmp_by_both);
                auto s_id = (s - sorted_both.begin()) - start_id;
                auto e_id = (e - sorted_both.begin()) - start_id;
                assert(s_id > 0 && e_id > 0);
                searcher->SearchCategoryRange(cur_q, K, graph_full_refine_k, tmp.data(), std::make_pair(l, r), std::make_pair(s_id, e_id));
            }
            maxPQIFCS<float>point_dist(k);
            for (size_t j = 0; j < graph_full_refine_k; j++)
            {
                if (j + 1 < graph_full_refine_k)
                {
                    glass::mem_prefetch((char *)(base + (tmp[j + 1] + start_id) * dim), 2);
                }
                point_dist.maybe_pop_emplace(tmp[j] + start_id, L2SqrRef(cur_q, base + (tmp[j] + start_id) * dim, dim));
            }
            for (size_t l = 0; l < k; ++l)
            {
                id[k - l - 1] = sorted_base_ids[point_dist.data_[l + 1].id];
            }
        }
    }
    timer.end();
    std::cout << "Type13 search time: " << std::chrono::duration<double>(timer.getElapsedTime()).count() << "s" << std::endl;

    timer.reset();
    std::vector<maxPQIFCS<int>> candidate_pool;
    candidate_pool.resize(nq);

    // search type0
    for (auto &[t, p] : timestamp_map) {
        auto &searcher = timestamp_searcher[t];
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nq; ++i) {
            int q_id = i;

            float type = metas[q_id * 4];
            if (type == 0) {
                float *cur_q = vecs + q_id * dim;
                if (candidate_pool[q_id].size() == 0) {
                    candidate_pool[q_id].resize(global_graph_refine_k);
                }

                auto& candidate_dists = candidate_pool[q_id];
                auto& [start_id, n] = timestamp_map[t];

                int K = params.ef0_full;
                std::vector<int> tmp(graph_full_refine_k);
                std::vector<float> dists(graph_full_refine_k);
                searcher->SearchFilterSubTime(cur_q, K, graph_full_refine_k, tmp.data(), dists.data(), nullptr);
                for (size_t j = 0; j < graph_full_refine_k; j++)
                { // todo: avoid comparing too much time
                    candidate_dists.maybe_pop_emplace(tmp[j] + start_id, dists[j]);
                }
            } 
        }
    }
    timer.end();
    std::cout << "Type0 search time: " << std::chrono::duration<double>(timer.getElapsedTime()).count() << "s" << std::endl;

    // search type2
    timer.reset();
//    timestamp_query.clear();
    std::vector<std::unordered_map<int, PIIRFType>> interval_pool;
    interval_pool.resize(nq);
    // get type2 query information
#pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < nq; ++i) {
        float type = metas[i * 4];
        float l = metas[i * 4 + 2];
        float r = metas[i * 4 + 3];
        float sel = sels[i];
        
        if (sel < bf_thr || (type == 3 && sel < bf_thr3))
        {
            // todo:make a tag instead of check again
            continue;
        }
        else if (type == 2) {
            if (sel < 0.2)
            {
                SplitInterval(std::make_pair(l, r), interval_pool[i], 0.5);
            }
            else
            {
                SplitInterval(std::make_pair(l, r), interval_pool[i]);
            }            
        }
    }

    for (size_t i = 0; i < nq; ++i) {
        auto &intervals = interval_pool[i];
        if (!intervals.empty()) {
            for (auto &[t, p]: intervals) {
                auto rf_type = p.first;
                timestamp_query[t].emplace_back(i, rf_type);
            }
        }
    }

//    for (auto &[t, q_ids] : timestamp_query) {
        for(int t=0;t<=9;t++) {
            auto& q_ids = timestamp_query[t];
        auto &searcher = timestamp_searcher[t];
#pragma omp parallel for schedule(dynamic)
        for (auto &[q_id, rf_type] : q_ids) {
            float type = metas[q_id * 4];
            float l = metas[q_id * 4 + 2];
            float r = metas[q_id * 4 + 3];
            float sel = sels[q_id];
            if (sel < bf_thr || (type == 3 && sel < bf_thr3)) {
                continue;
            }
            if (type == 2) {
                float *cur_q = vecs + q_id * dim;
                if (candidate_pool[q_id].size() == 0) {
                    candidate_pool[q_id].resize(global_graph_refine_k);
                }

                auto& candidate_dists = candidate_pool[q_id];
                auto& p = interval_pool[q_id][t];
                auto& [start_id, n] = timestamp_map[t];

                if (rf_type == RFType::SMALL) {
                    int s = p.second.first;
                    uint32_t n_points = p.second.second;
                    auto &quant = ((Searcher<SQ8SymmetricQuantizer<Metric::L2>> *)searcher.get())->quant;
                    int8_t *encode_q = (int8_t *)glass::alloc64B(quant.code_size);
                    quant.encode(cur_q, (char *)encode_q);
                    char *codes = quant.codes + s * quant.code_size;
                    auto K = std::min(bf_refine_k, n_points);
                    std::vector<int> tmp(K);
                    std::vector<float> dists(K);

                    bruteforce_subgraph(dim, K, tmp.data(), dists.data(), n_points, codes, quant.alpha, 1, encode_q);

                    // id map
                    for (size_t j = 0; j < K; ++j)
                    {
                        tmp[j] = tmp[j] + start_id + s;
                        // avoid redundant heap emplace
                        candidate_dists.maybe_pop_emplace(tmp[j], dists[j]);
                    }
                    free(encode_q);
                } else if (rf_type == RFType::MEDIUM) {
                    int K = params.ef2_medium_small;
                    if (sel > 0.6)
                    {
                        K = params.ef2_medium_large;
                    }
                    else if (sel > 0.3)
                    {
                        K = params.ef2_medium_medium;
                    }
                    std::vector<int> tmp(graph_part_refine_k);
                    std::vector<float> dists(graph_part_refine_k);
                    int s = p.second.first; // first element ID that satisfy the filter
                    uint32_t n_points = p.second.second;
                    searcher->SearchRangeFilterSubTime(cur_q, K, graph_part_refine_k, tmp.data(), dists.data(), std::make_pair(l, r), std::make_pair(s, s + n_points - 1));
                    for (size_t j = 0; j < graph_part_refine_k; j++)
                    {
                        candidate_dists.maybe_pop_emplace(tmp[j] + start_id, dists[j]);
                    }
                } else if (rf_type == RFType::FULL) {
                    int K = params.ef2_full_large;
                    if (sel < 0.3)
                    {
                        K = params.ef2_full_small;
                    }
                    else if (sel < 0.6)
                    {
                        K = params.ef2_full_medium;
                    }
                    std::vector<int> tmp(graph_full_refine_k);
                    std::vector<float> dists(graph_full_refine_k);
                    searcher->SearchFilterSubTime(cur_q, K, graph_full_refine_k, tmp.data(), dists.data(), nullptr);
                    for (size_t j = 0; j < graph_full_refine_k; j++)
                    {
                        candidate_dists.maybe_pop_emplace(tmp[j] + start_id, dists[j]);
                    }
                }
            }
        }
    }
    timer.end();
    std::cout << "Type2 search time: " << std::chrono::duration<double>(timer.getElapsedTime()).count() << "s" << std::endl;

    timer.reset();
    // global refine only for type 0 and type 2
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < nq; ++i) {
        float type = metas[i * 4];
        float v = metas[i * 4 + 1];
        float l = metas[i * 4 + 2];
        float r = metas[i * 4 + 3];
        float sel = sels[i];
        uint32_t query_id = sorted_ids[i];
        // stats[query_id] = QueryStats{type, sel, 0.0f};
        if (sel < bf_thr || (type == 3 && sel < bf_thr3)) {
            continue;
        }
        if (type == 0 || type == 2) {
            int *id = (int *)ids + query_id * k;
            float *cur_q = vecs + i * dim;
            auto& candidate_dists = candidate_pool[i];
            maxPQIFCS<float>point_dist(k);
            for (size_t j = 1; j <= candidate_dists.size(); j++) {
                int id = candidate_dists.data_[j].id;
                int next_id = candidate_dists.data_[j + 1].id;
                if (j + 1 <= candidate_dists.size()) {
                    glass::mem_prefetch((char *)(base_by_full_time + next_id * dim), 2);
                }
                point_dist.maybe_pop_emplace(id, L2SqrRef(cur_q, base_by_full_time + id * dim, dim));
            }
            for (size_t l = 0; l < k; ++l) {
                id[k - l - 1] = sorted_base_ids_by_full_time[point_dist.data_[l + 1].id];
            }
        }
    }
    timer.end();
    std::cout << "Type02 refine time: " << std::chrono::duration<double>(timer.getElapsedTime()).count() << "s" << std::endl;
    graph_timer.end();
    std::cout << "Graph search time: " << std::chrono::duration<double>(graph_timer.getElapsedTime()).count() << "s" << std::endl;

    // std::ofstream out_stats("query_stats.bin", std::ios::binary);
    // out_stats.write((char *)stats.data(), nq * sizeof(QueryStats));
    // out_stats.close();
}

void HybridGraph::SortDataset(SQ8SymmetricQuantizer<Metric::L2> &quant, float *labels, float *timestamps, int nb)
{
    for (size_t i = 0; i < nb; i++)
    {
        sorted_both.emplace_back(MetaData(i, labels[i], timestamps[i]));
        sorted_timestamp.push_back(MetaData(i, labels[i], timestamps[i]));
    }
    std::sort(sorted_both.begin(), sorted_both.end(), cmp_by_both);
    std::sort(sorted_timestamp.begin(), sorted_timestamp.end(), cmp_by_time);

    codes_both.resize(nb * 112);
    codes_time.resize(nb * 112);
    alpha = quant.alpha;

    for (size_t i = 0; i < nb; i++)
    {
        memcpy(codes_both.data() + i * 112, quant.codes + sorted_both[i].id * 112, 112);
        memcpy(codes_time.data() + i * 112, quant.codes + sorted_timestamp[i].id * 112, 112);
    }
}
