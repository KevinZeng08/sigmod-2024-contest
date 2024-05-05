// #include <mkl.h>
#include <cstdint>
#include <queue>
#include <iostream>

#include "glass/simd/distance.hpp"
#include "glass/memory.hpp"

//struct maxPQIFCS
//{
//    float *data;
//    int *id;
//    int cnt=0;
//    explicit maxPQIFCS(int n){
//        data=new float[n+5];
//        id=new int[n+5];
//    }
//    ~maxPQIFCS() {
//        delete[] data;
//        delete[] id;
//    }
//    inline bool empty(){
//        return cnt==0;
//    }
//    void emplace(int ids,float dists)
//    {
//        int pl= std::lower_bound(data+1,data+cnt+1,dists)-data;
//        std::memmove(data+pl+1,data+pl,(cnt-pl+1)*sizeof(int));
//        std::memmove(id+pl+1,id+pl,(cnt-pl+1)*sizeof(int));
//        cnt++;
//        data[pl]=dists;
//        id[pl]=ids;
//    }
//    inline std::pair<int,float> top(){
//        return std::make_pair(id[cnt],data[cnt]);
//    }
//    inline std::pair<int,float> tail(){
//        return std::make_pair(data[1],id[1]);
//    }
//    inline void pop(){
//        cnt--;
//    }
//    inline void pop_tail(){
//
//    }
//    inline int size(){
//        return cnt;
//    }
//    std::vector<node, glass::align_alloc<node>> data_;
//};
template <typename dist_t=float>
struct node{
    dist_t distance;
    int id;
};

template <typename dist_t = float>
struct maxPQIFCS
{
    int capacity_,cnt;
    bool full;
    std::vector<node<dist_t>, glass::align_alloc<node<dist_t>>> data_;
    maxPQIFCS() : capacity_(0),cnt(0),full(false) {};
    maxPQIFCS(int capacity):
            capacity_(capacity), data_(capacity + 5),cnt(0),full(false) {}
    
    void resize(int capacity) {
        capacity_ = capacity;
        data_.resize(capacity + 5);
        cnt = 0;
        full = false;
    }

    inline std::pair<int,dist_t> top(){
        return std::make_pair(data_[1].id,data_[1].distance);
    }
    inline int size(){return cnt;}
    void push_down(int x) {
        int t;
        while((x<<1)<=cnt){
            t=x<<1;
            if((t|1)<=cnt&&data_[t|1].distance>data_[t].distance)t|=1;
            if(data_[t].distance<=data_[x].distance)break;
            std::swap(data_[x],data_[t]);
            x = t;
        }
    }
    void maybe_pop_emplace(int ids, dist_t dists)
    {
        if(!full){
            emplace(ids,dists);
            return;
        }
        if(data_[1].distance<=dists)return;
        data_[1].id=ids;
        data_[1].distance=dists;
        push_down(1);
    }
    void must_pop_emplace(int ids, dist_t dists)
    {
        if(data_[1].distance<=dists)return;
        data_[1].id=ids;
        data_[1].distance=dists;
        push_down(1);
    }
    void emplace(int ids,dist_t dists)
    {
        data_[++cnt]={dists,ids};
        if(cnt==capacity_)full=true;
        int now=cnt;
        while(now!=1&&data_[now].distance>data_[now>>1].distance)
            std::swap(data_[now],data_[now>>1]),now>>=1;
    }
    void build(){
        for(int i=cnt/2;i;i--)push_down(i);
    }
};

//using pairIF = std::pair<int, float>;
//struct cmpmaxstruct
//{
//    bool operator()(const pairIF &l, const pairIF &r)
//    {
//        return l.second < r.second;
//    };
//};
//using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

// void compute_l2sq(float *const points_l2sq, const float *const matrix,
//                   const int64_t num_points, const uint64_t dim) {
//   for (int64_t d = 0; d < num_points; ++d) {
//     points_l2sq[d] = cblas_sdot((int64_t)dim, matrix + d * dim, 1, matrix + d * dim, 1);
//   }
// }

// void distsq_to_points(const size_t dim,
//                       float *dist_matrix, // Col Major, cols are queries, rows are points
//                       size_t npoints, const float *const points,
//                       const float *const points_l2sq, // points in Col major
//                       size_t nqueries, const float *const queries,
//                       const float *const queries_l2sq, // queries in Col major
//                       float *ones_vec = NULL)          // Scratchspace of num_data size and init to 1.0
// {
//     bool ones_vec_alloc = false;
//     if (ones_vec == NULL)
//     {
//         ones_vec = new float[nqueries > npoints ? nqueries : npoints];
//         std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float)1.0);
//         ones_vec_alloc = true;
//     }
//     cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, points, dim, queries, dim,
//                 (float)0.0, dist_matrix, npoints);
//     cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, points_l2sq, npoints,
//                 ones_vec, nqueries, (float)1.0, dist_matrix, npoints);
//     cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, ones_vec, npoints,
//                 queries_l2sq, nqueries, (float)1.0, dist_matrix, npoints);
//     if (ones_vec_alloc)
//         delete[] ones_vec;
// }

// void exact_knn(const size_t dim, const size_t k,
//                int *const closest_points, // k * num_queries preallocated,
//                                              // col major, queries columns
//                size_t npoints,
//                float *points_in,                   // points in Col major
//                size_t nqueries, float *queries_in) // queries in Col major
// {
//   float *points_l2sq = new float[npoints];
//   float *queries_l2sq = new float[nqueries];
//   compute_l2sq(points_l2sq, points_in, npoints, dim);
//   compute_l2sq(queries_l2sq, queries_in, nqueries, dim);

//   float *points = points_in;
//   float *queries = queries_in;
//   float *dist_matrix = new float[(size_t)nqueries * (size_t)npoints];

//   distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, nqueries,
//                    queries, queries_l2sq);

//   for (long long q = 0; q < nqueries; q++) {
//     maxPQIFCS point_dist;
//     for (size_t p = 0; p < k; p++)
//       point_dist.emplace(p, dist_matrix[p + q * npoints]);
//     for (size_t p = k; p < npoints; p++) {
//       if (point_dist.top().second > dist_matrix[p + q * npoints])
//         point_dist.emplace(p, dist_matrix[p + q * npoints]);
//       if (point_dist.size() > k)
//         point_dist.pop();
//     }
//     for (size_t l = 0; l < k; ++l) {
//       closest_points[(k - 1 - l) + q * k] = point_dist.top().first;
//       point_dist.pop();
//     }
//   }
//   delete[] dist_matrix;
//   delete[] points_l2sq;
//   delete[] queries_l2sq;
// }

void printvector(size_t x,float* raw_codes){
    float* now=raw_codes+x*100;
    for(int i=0;i<100;i++)
    {
        std::cout<<now[i]<<' ';
    }
    std::cout<<std::endl;
}
//void bruteforce(const size_t dim, const size_t k,
//                int *const closest_points,
//                size_t npoints,
//                char *codes_in,
//                float *mi,
//                float *dif,
//                size_t nqueries, float *queries_in) {
//    std::vector<float> dists(npoints);
//    _mm_prefetch(mi, _MM_HINT_T0);
//    _mm_prefetch(dif, _MM_HINT_T0);
//    for (int i = 0; i < 2; ++i) {
//        _mm_prefetch(codes_in + (i) * 112, _MM_HINT_T0);
//    }
//    for (size_t i = 0; i < npoints; i++) {
//        dists[i] = glass::L2SqrSQ8_ext(queries_in, (uint8_t *) codes_in + i * 112, 112, mi, dif);
//        if (i + 2 < npoints) {
//            _mm_prefetch(codes_in + (i + 5) * 112, _MM_HINT_T0);
//        }
//    }
//    maxPQIFCS point_dist;
//    for (size_t p = 0; p < k; p++)
//        point_dist.emplace(p, dists[p]);
//    for (size_t p = k; p < npoints; p++) {
//        if (point_dist.top().second > dists[p]) {
//
//            point_dist.emplace(p, dists[p]);
//            if (point_dist.size() > k)
//                point_dist.pop();
//        }
//    }
//        for (size_t l = 0; l < k; ++l) {
//            closest_points[k - 1 - l] = point_dist.top().first;
//            point_dist.pop();
//        }
//}

void bruteforce(const size_t dim, const size_t k,
                int *const closest_points,
                size_t npoints,
                char *codes_in,
                float alpha,
                size_t nqueries, int8_t *queries_in) {
    std::vector<int, glass::align_alloc<int>> dists(npoints);
    for (size_t i = 0; i < npoints; i++) {
        dists[i] = glass::L2SqrSQ8_sym(queries_in, (int8_t *) codes_in + i * 112, 112);
    }
    maxPQIFCS<int>point_dist(k);
//    maxPQIFCS point_dist;
   for (size_t p = 0; p < k; p++)
       point_dist.emplace(p, dists[p]);
    for (size_t p = k; p < npoints; p++) {
        point_dist.must_pop_emplace(p, dists[p]);
            // if (point_dist.size() > k)
    }
        for (size_t l = 0; l < k; ++l) {
            closest_points[k - l - 1] = point_dist.data_[l+1].id;
        }
}

void bruteforce_subgraph(const size_t dim, const size_t k,
                int *const closest_points,
                float *distances,
                size_t npoints,
                char *codes_in,
                float alpha,
                size_t nqueries, int8_t *queries_in) {
    std::vector<int, glass::align_alloc<int>> dists(npoints);
    for (size_t i = 0; i < npoints; i++) {
        dists[i] = glass::L2SqrSQ8_sym(queries_in, (int8_t *) codes_in + i * 112, 112);
    }
    maxPQIFCS<int>point_dist(k);
//    maxPQIFCS point_dist;
   for (size_t p = 0; p < k; p++)
       point_dist.emplace(p, dists[p]);
    for (size_t p = k; p < npoints; p++) {
            point_dist.must_pop_emplace(p, dists[p]);
    }
        for (size_t l = 0; l < k; ++l) {
            closest_points[k - l - 1] = point_dist.data_[l+1].id;
            distances[k - l - 1] = point_dist.data_[l+1].distance * alpha * alpha;
        }
}