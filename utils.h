#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <assert.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "distance.h"
#include "thread_pool.h"

using std::cout;
using std::endl;
using std::priority_queue;
using std::string;
using std::unique_ptr;
using std::vector;
using std::unordered_map;

using PII = std::pair<uint32_t, uint32_t>; // <start_id, num_points>

#include <queue>

struct MetaData {
    int id;
    float label;
    float timestamp;

    MetaData(int id, float label, float timestamp): id(id), label(label), timestamp(timestamp) {}
};
// max heap with max size *k*
template <typename DisT, typename IdT>
class ResultMaxHeap
{
public:
    ResultMaxHeap(size_t k) : k_(k){};

    inline std::pair<DisT, IdT>
    Pop()
    {
        if (pq.empty())
        {
            throw std::runtime_error("pop from empty heap");
        }
        std::pair<DisT, IdT> res = pq.top();
        pq.pop();
        return res;
    }

    inline void
    Push(DisT dis, IdT id)
    {
        if (pq.size() < k_)
        {
            pq.emplace(dis, id);
            return;
        }

        if (dis < pq.top().first)
        {
            pq.pop();
            pq.emplace(dis, id);
        }
    }

    inline size_t
    Size()
    {
        return pq.size();
    }

private:
    size_t k_;
    std::priority_queue<std::pair<DisT, IdT>> pq;
};

inline void Bruteforce(const vector<vector<float>> &nodes, const vector<vector<float>> &queries, vector<vector<uint32_t>> &gt)
{
    // brute force to get ground truth
    uint32_t n = nodes.size();
    uint32_t d = nodes[0].size();
    d -= 2; // skip first 2 dimensions
    uint32_t nq = queries.size();

    const int K = 100;
    gt.resize(nq);

    auto pool = ThreadPool::GetSearchPool();
    vector<std::future<void>> futures;

    for (size_t i = 0; i < nq; i++)
    {
        futures.emplace_back(pool->push([&, i]()
                                        {
                                 uint32_t query_type = queries[i][0];
                                 int32_t v = queries[i][1];
                                 float l = queries[i][2];
                                 float r = queries[i][3];
                                // skip first 4 dimensions for queries
                                 const float *query_vec = queries[i].data() + 4;
                                 ResultMaxHeap<float, uint32_t> heap(K);
                                 for (uint j = 0; j < n; ++j)
                                 {
                                     // skip first 2 dimensions
                                     const float *base_vec = nodes[j].data() + 2;
                                     int32_t bv = nodes[j][0];
                                     if (query_type == 0)
                                     {
                                         float dist = normal_l2(base_vec, query_vec, d);
                                         heap.Push(dist, j);
                                     }
                                     else if (query_type == 1)
                                     {
                                         if (v == bv)
                                         {
                                             float dist = normal_l2(base_vec, query_vec, d);
                                             heap.Push(dist, j);
                                         }
                                     }
                                     else if (query_type == 2)
                                     {
                                         if (nodes[j][1] >= l && nodes[j][1] <= r)
                                         {
                                             float dist = normal_l2(base_vec, query_vec, d);
                                             heap.Push(dist, j);
                                         }
                                     }
                                     else if (query_type == 3)
                                     {
                                         if (v == bv && nodes[j][1] >= l && nodes[j][1] <= r)
                                         {
                                             float dist = normal_l2(base_vec, query_vec, d);
                                             heap.Push(dist, j);
                                         }
                                     }
                                 }

                                 gt[i].resize(K);
                                 if (heap.Size() < K) {
                                    cout << "id: " << i << endl;
                                    cout << "query type: " << query_type << " v: " << v << " l: " << l << " r: " << r << endl;
                                    cout << "K: " << heap.Size() << endl;
                                 }
                                 for (int j = K - 1; j >= 0; j--)
                                 {
                                     auto res = heap.Pop();
                                     gt[i][j] = res.second;
                                 } }));
    }

    for (auto &future : futures)
    {
        future.get();
    }
}

/// @brief Calculate groundtruth given source data and queries
inline void GetGroundTruth(const vector<vector<float>> &nodes, const vector<vector<float>> &queries, vector<vector<uint32_t>> &gt)
{
    Bruteforce(nodes, queries, gt);
}

/// @brief Save ground truth to a binary file
/// @param gt ground truth (nq * dim)
inline void SaveGroundTruth(const vector<vector<uint32_t>> &gt, const string &gt_path)
{
    cout << "Writing Ground Truth: " << gt_path << endl;
    std::ofstream ofs;
    ofs.open(gt_path, std::ios::binary);
    if (!ofs.is_open())
    {
        cout << "open file error" << endl;
        return;
    }
    uint32_t N = gt.size();
    uint32_t dim = gt[0].size();
    cout << "# of points: " << N << " dim: " << dim << endl;
    ofs.write((char *)&N, sizeof(uint32_t));
    for (size_t i = 0; i < N; i++)
    {
        ofs.write((char *)(gt[i].data()), dim * sizeof(uint32_t));
    }
    ofs.close();
    cout << "Finish Writing Ground Truth" << endl;
}

/// @brief
/// @param gt_path
/// @param num_dimensions equal to topK
/// @param gt ground truth (nq * topK)
inline void ReadGroundTruth(const string &gt_path, const int num_dimensions, vector<vector<uint32_t>> &gt)
{
    cout << "Reading Ground Truth: " << gt_path << endl;
    std::ifstream ifs;
    ifs.open(gt_path, std::ios::binary);
    if (!ifs.is_open())
    {
        cout << "open file error" << endl;
        return;
    }
    uint32_t N;
    ifs.read((char *)&N, sizeof(uint32_t));
    gt.resize(N);
    cout << "# of points: " << N << endl;
    vector<uint32_t> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(uint32_t)))
    {
        vector<uint32_t> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++)
        {
            row[d] = static_cast<uint32_t>(buff[d]);
        }
        gt[counter++] = std::move(row);
    }
    ifs.close();
    cout << "Finish Reading Ground Truth" << endl;
}

/// @brief Calculate recall based on query results and ground truth information
inline float GetKNNRecall(const vector<vector<uint32_t>> &knns, const vector<vector<uint32_t>> &gt)
{
    std::vector<int> recalls(gt.size()); 
    assert(knns.size() == gt.size());

    uint64_t total_correct = 0;
    size_t nq = knns.size();
    size_t topk = knns[0].size();

    for (size_t i = 0; i < nq; i++)
    {
        size_t correct = 0;
        for (size_t j = 0; j < topk; j++)
        {
            for (size_t k = 0; k < topk; k++)
            {
                if (knns[i][k] == gt[i][j])
                {
                    correct++;
                    break;
                }
            }
        }
        recalls[i] = correct;
        total_correct += correct;
    }
    std::ofstream out("recall1.bin", std::ios::binary);
    out.write((char*)recalls.data(), nq * 4);
    return (float)total_correct / nq / topk;
}

const int vec_dim = 100;

inline void ReadBase(const string &source_path, uint32_t &N, vector<float> &vecs, vector<float> &labels, vector<float> &timestamps)
{
    cout << "Reading Data: " << source_path << endl;
    std::ifstream ifs;
    ifs.open(source_path, std::ios::binary);
    assert(ifs.is_open());
    ifs.read((char *)&N, sizeof(uint32_t));

    vecs.resize(N * vec_dim);
    labels.resize(N);
    timestamps.resize(N);

    cout << "# of points: " << N << endl;
    for (size_t i = 0; i < N; i++)
    {
        ifs.read((char *)&labels[i], sizeof(float));
        ifs.read((char *)&timestamps[i], sizeof(float));
        ifs.read((char *)(vecs.data() + i * vec_dim), vec_dim * sizeof(float));
    }
}
//
//void ReadSortedBase(const string &source_path, uint32_t &N, vector<float> &vecs, vector<float> &labels, vector<float> &timestamps, vector<uint32_t> &sorted_base_ids,
//                    unordered_map<int32_t, PII> &category_map,size_t& maxc_id)
//{
//    size_t maxc=0;
//    cout << "Reading Data: " << source_path << endl;
//    std::ifstream ifs;
//    ifs.open(source_path, std::ios::binary);
//    assert(ifs.is_open());
//    ifs.read((char *)&N, sizeof(uint32_t));
//
//    auto tmp_vecs = std::make_unique<float[]>(N * vec_dim);
//    auto tmp_labels = std::make_unique<float[]>(N);
//    auto tmp_timestamps = std::make_unique<float[]>(N);
//    vecs.resize(N * vec_dim);
//    labels.resize(N);
//    timestamps.resize(N);
//
//    cout << "# of points: " << N << endl;
//    unordered_map<int32_t, uint32_t> category_count;
//    for (size_t i = 0; i < N; i++)
//    {
//        ifs.read((char *)&tmp_labels[i], sizeof(float));
//        category_count[tmp_labels[i]]++;
//        ifs.read((char *)&tmp_timestamps[i], sizeof(float));
//        ifs.read((char *)(tmp_vecs.get() + i * vec_dim), vec_dim * sizeof(float));
//    }
//    // sort base vectors by category, to build sub-index with category
//    sorted_base_ids.resize(N);
//    std::iota(sorted_base_ids.begin(), sorted_base_ids.end(), 0);
//    auto cmp = [&](const uint32_t &a, const uint32_t &b) {
//        int32_t ca = tmp_labels[a];
//        int32_t cb = tmp_labels[b];
//        if(category_count[ca]!=category_count[cb])
//        {
//            return category_count[ca]>category_count[cb];
//        }
//        else if (ca != cb)
//        {
//            return ca < cb;
//        }
//        else
//        {
//            return tmp_timestamps[a] < tmp_timestamps[b];
//        }
//    };
//    std::sort(sorted_base_ids.begin(), sorted_base_ids.end(), cmp);
//    size_t category_end = 0;
//    for (size_t i = 0; i < N; ++i)
//    {
//        uint32_t rank = sorted_base_ids[i];
//        memcpy(vecs.data() + i * vec_dim, tmp_vecs.get() + rank * vec_dim, vec_dim * sizeof(float));
//        labels[i] = tmp_labels[rank];
//        timestamps[i] = tmp_timestamps[rank];
//
//        int32_t v = labels[i];
//        if (i < category_end || category_count[v] < 100) {
//            continue;
//        }
//        if (category_map.find(v) == category_map.end()) {
//            category_map[v] = {i, category_count[v]};
//            category_end = i + category_count[v];
//            if(maxc<category_count[v]){
//                maxc=category_count[v];
//                maxc_id=v;
//            }
//        }
//    }
//
//
//}

constexpr int interval = 1; // the interval to build timestamp sub-index
constexpr float cat_thr = 0.045;

inline void ReadSortedBaseTimestamp(const string &source_path, uint32_t &N, vector<float> &vecs, vector<float> &labels, vector<float> &timestamps, vector<uint32_t> &sorted_base_ids,
                    unordered_map<int32_t, PII> &category_map, vector<float> &vecs_by_time, vector<float> &labels_by_time, vector<float> &timestamps_by_time, vector<uint32_t> &sorted_base_ids_by_time,int& maxc_id,
                    vector<float> &vecs_by_full_time, vector<float> &labels_by_full_time, vector<float> &timestamps_by_full_time, vector<uint32_t> &sorted_base_ids_by_full_time,
                    unordered_map<int32_t, PII> &timestamp_map,int& max_count,int& min_count)
{
    cout << "Reading Data: " << source_path << endl;
    std::ifstream ifs;
    ifs.open(source_path, std::ios::binary);
    assert(ifs.is_open());
    ifs.read((char *)&N, sizeof(uint32_t));

    auto tmp_vecs = std::make_unique<float[]>(N * vec_dim);
    auto tmp_labels = std::make_unique<float[]>(N);
    auto tmp_timestamps = std::make_unique<float[]>(N);
    vecs.resize(N * vec_dim);
    labels.resize(N);
    timestamps.resize(N);
    vecs_by_time.resize(N * vec_dim);
    labels_by_time.resize(N);
    timestamps_by_time.resize(N);
    vecs_by_full_time.resize(N * vec_dim);
    labels_by_full_time.resize(N);
    timestamps_by_full_time.resize(N);

    cout << "# of points: " << N << endl;
    unordered_map<int32_t, uint32_t> category_count;
    for (size_t i = 0; i < N; i++)
    {
        ifs.read((char *)&tmp_labels[i], sizeof(float));
        category_count[tmp_labels[i]]++;
        ifs.read((char *)&tmp_timestamps[i], sizeof(float));
        ifs.read((char *)(tmp_vecs.get() + i * vec_dim), vec_dim * sizeof(float));
    }
    // find largest category id

    for (auto& [label, count] : category_count) {
        if (count > max_count) {
            max_count = count;
            maxc_id = label;
        }
        if(count < min_count){
            min_count = count;
        }
    }
    // sort base vectors by category, to build sub-index with category
    sorted_base_ids.resize(N);
    sorted_base_ids_by_time.resize(N);
    sorted_base_ids_by_full_time.resize(N);
    std::iota(sorted_base_ids.begin(), sorted_base_ids.end(), 0);
    std::iota(sorted_base_ids_by_time.begin(), sorted_base_ids_by_time.end(), 0);
    std::iota(sorted_base_ids_by_full_time.begin(), sorted_base_ids_by_full_time.end(), 0);
    auto cmp = [&](const uint32_t &a, const uint32_t &b) {
        int32_t ca = tmp_labels[a];
        int32_t cb = tmp_labels[b];
        if (ca != cb)
        {
            return ca < cb;
        }
        else
        {
            return tmp_timestamps[a] < tmp_timestamps[b];
        }
    };
    auto cmp_by_time = [&] (const uint32_t &a, const uint32_t &b) {
        int32_t ca = tmp_labels[a];
        int32_t cb = tmp_labels[b];
        if (ca == maxc_id) { // largest category first to avoid redundant build of largest category
            if (ca == cb) {
                return tmp_timestamps[a] < tmp_timestamps[b];
            }
            return true;
        }
        if (cb == maxc_id) {
            return false;
        }
        return tmp_timestamps[a] < tmp_timestamps[b];
    };
    auto cmp_by_full_time = [&] (const uint32_t &a, const uint32_t &b) {
        return tmp_timestamps[a] < tmp_timestamps[b];
    };
    std::sort(sorted_base_ids.begin(), sorted_base_ids.end(), cmp);
    std::sort(sorted_base_ids_by_time.begin(), sorted_base_ids_by_time.end(), cmp_by_time);
    std::sort(sorted_base_ids_by_full_time.begin(), sorted_base_ids_by_full_time.end(), cmp_by_full_time);
    // for the largest category, we force its order in both id map to be the same, (unstablity of std::sort)
    // so we don't have to change other strategy for the largest category
    int start_id = std::find_if(sorted_base_ids.begin(), sorted_base_ids.end(), [&](uint32_t id) {
        return tmp_labels[id] == maxc_id;
    }) - sorted_base_ids.begin();
    int max_cat_count = category_count[maxc_id];
    for (int i = 0; i < max_cat_count; ++i) {
        sorted_base_ids_by_time[i] = sorted_base_ids[i + start_id];
    }
    size_t category_end = 0;
    int min_cat_count = N * cat_thr;
    for (size_t i = 0; i < N; ++i)
    {
        uint32_t rank = sorted_base_ids[i];
        uint32_t rank_by_time = sorted_base_ids_by_time[i];
        uint32_t rank_by_full_time = sorted_base_ids_by_full_time[i];
        memcpy(vecs.data() + i * vec_dim, tmp_vecs.get() + rank * vec_dim, vec_dim * sizeof(float));
        memcpy(vecs_by_time.data() + i * vec_dim, tmp_vecs.get() + rank_by_time * vec_dim, vec_dim * sizeof(float));
        memcpy(vecs_by_full_time.data() + i * vec_dim, tmp_vecs.get() + rank_by_full_time * vec_dim, vec_dim * sizeof(float));
        labels[i] = tmp_labels[rank];
        timestamps[i] = tmp_timestamps[rank];
        labels_by_time[i] = tmp_labels[rank_by_time];
        timestamps_by_time[i] = tmp_timestamps[rank_by_time];
        labels_by_full_time[i] = tmp_labels[rank_by_full_time];
        timestamps_by_full_time[i] = tmp_timestamps[rank_by_full_time];

        int32_t v = labels[i];
        if (i < category_end || category_count[v] < min_cat_count) {
            continue;
        }
        if (category_map.find(v) == category_map.end()) {
            category_map[v] = {i, category_count[v]};
            category_end = i + category_count[v];
        }
    }
    // 0 -> [0.0, 0.1], 1 -> [0.1, 0.2], ..., 9 -> [0.9, 1.0]
    // id -> {start_id, num}
     for (int i = 0; i <= 9; i += interval) {
        auto s = std::lower_bound(timestamps_by_full_time.begin(), timestamps_by_full_time.end(), 0.1 * i);
        auto e = std::upper_bound(timestamps_by_full_time.begin(), timestamps_by_full_time.end(), 0.1 * (i + interval));
        int num = e - s;
        if (num == 0) continue;
        timestamp_map[i] = {s - timestamps_by_full_time.begin(), num};
    }
}

// type of range filter
enum RFType {
    // for specific timestamp range index, we have four situations and corresponding search strategies
    // SMALL: brute-force
    // MEDIUM: in-filter
    // LARGE: post-filter (TODO)
    // FULL: no filter
    SMALL=0, MEDIUM, LARGE, FULL
};

inline void ReadSortedQuery(const string &query_path, uint32_t &N, vector<float> &vecs, vector<float> &metas, vector<uint32_t> &sorted_ids,
                            unordered_map<int32_t, vector<int>> &category_query, const unordered_map<int32_t, PII> &category_map)
{
    cout << "Reading Query: " << query_path << endl;
    std::ifstream ifs;
    ifs.open(query_path, std::ios::binary);
    assert(ifs.is_open());
    ifs.read((char *)&N, sizeof(uint32_t));

    auto tmp_vecs = std::make_unique<float[]>(N * vec_dim);
    auto tmp_metas = std::make_unique<float[]>(N * 4);
    vecs.resize(N * vec_dim);
    metas.resize(4 * N);
    cout << "# of points: " << N << endl;
    for (size_t i = 0; i < N; i++)
    {
        ifs.read((char *)(tmp_metas.get() + i * 4), sizeof(float) * 4);
        ifs.read((char *)(tmp_vecs.get() + i * vec_dim), vec_dim * sizeof(float));
    }

    sorted_ids.resize(N);
    std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

    auto cmp = [&](const uint32_t &a, const uint32_t &b) {
        uint32_t ta = tmp_metas[a * 4];
        uint32_t tb = tmp_metas[b * 4];
        int32_t ca = tmp_metas[a * 4 + 1];
        int32_t cb = tmp_metas[b * 4 + 1];
        float la = tmp_metas[a * 4 + 2];
        float lb = tmp_metas[b * 4 + 2];
        float ra = tmp_metas[a * 4 + 3];
        float rb = tmp_metas[b * 4 + 3];
        if (ta != tb)
        {
            return ta < tb;
        } else if (ca != cb)
        {
            return ca < cb;
        } else if (la != lb)
        {
            return la < lb;
        } else
        {
            return ra < rb;
        }
    };
    std::sort(sorted_ids.begin(), sorted_ids.end(), cmp);
    for (size_t i = 0; i < N; ++i) {
        uint32_t rank = sorted_ids[i];
        // type1 3
        int32_t v = tmp_metas[rank * 4 + 1];
        if (v != -1 && category_map.find(v) != category_map.end()) {
            category_query[v].push_back(i);
        }
        memcpy(vecs.data() + i * vec_dim, tmp_vecs.get() + rank * vec_dim, vec_dim * sizeof(float));
        memcpy(metas.data() + i * 4, tmp_metas.get() + rank * 4, 4 * sizeof(float));
    }
}

inline void ReadQuery(const string &query_path, uint32_t &N, vector<float> &vecs, vector<float> &metas)
{
    cout << "Reading Query: " << query_path << endl;
    std::ifstream ifs;
    ifs.open(query_path, std::ios::binary);
    assert(ifs.is_open());
    ifs.read((char *)&N, sizeof(uint32_t));

    vecs.resize(N * vec_dim);
    metas.resize(4 * N);
    cout << "# of points: " << N << endl;
    for (size_t i = 0; i < N; i++)
    {
        ifs.read((char *)(metas.data() + i * 4), sizeof(float) * 4);
        ifs.read((char *)(vecs.data() + i * vec_dim), vec_dim * sizeof(float));
    }
}

struct QueryStats {
    float type, selectivity, time;
};

inline void ReadStats(uint32_t nq, const string &stas_path = "query_stats.bin") {
    cout << "=== Reading Query Stats ===\n";
    vector<QueryStats> q;
    q.resize(nq);
    std::ifstream inFile("query_stats.bin", std::ios::binary); // 以二进制模式打开文件

    // 获取文件大小
    inFile.seekg(0, std::ios::end);
    std::streamsize size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    vector<float> buffer(size / sizeof(float));
    inFile.read(reinterpret_cast<char*>(buffer.data()), size); // 从文件中读取数据到缓冲区
    int i=0;
    for (const auto& value : buffer) {
        if(i%3==0)q[i/3].type=value;
        if(i%3==1)q[i/3].selectivity=value;
        if(i%3==2)q[i/3].time=value;
        i++;
    }

    // calculate time of different query types
    float type_time[4] = {0};
    int type_count[4] = {0};
    for (const auto& stats : q) {
        type_time[(int)stats.type] += stats.time;
        type_count[(int)stats.type]++;
    }
    for (int i = 0; i < 4; ++i) {
        cout << "Type[" << i << "] count: " << type_count[i] << endl;
        cout << "Type[" << i << "] total time: " << type_time[i] / 1000 << " s" << endl;
        cout << "Type[" << i << "] average time: " << type_time[i] / type_count[i] << " ms" << endl;
    }

    // calculate time of bruteforce and graph
    float bf_time = 0, graph_time = 0;
    float bf_count = 0, graph_count = 0;
    float bf_thr = 0.05;
    for (const auto& stats : q) {
        if (stats.selectivity < bf_thr) {
            bf_time += stats.time;
            bf_count++;
        } else {
            graph_time += stats.time;
            graph_count++;
        }
    }
    cout << "Bruteforce count: " << bf_count << endl;
    cout << "Graph count: " << graph_count << endl; 
    cout << "Bruteforce average time: " << bf_time / bf_count << " ms" << endl;
    cout << "Graph average time: " << graph_time / graph_count << " ms" << endl;
 
    inFile.close(); // 关闭文件
}

inline void CalculateSquareSum(int8_t* codes, int N, int dim) {
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += codes[i * dim + j] * codes[i * dim + j];
        }
        if (sum > 65536) {
            throw std::runtime_error("sum > 65536");
        }
    }
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            int ip = 0;
            for (int k = 0; k < dim; k++) {
                ip += codes[i * dim + k] * codes[j * dim + k];
            }
            if (ip > 32768 || ip < -32767) {
                throw std::runtime_error("ip overflow");
            }
        }
    }
}