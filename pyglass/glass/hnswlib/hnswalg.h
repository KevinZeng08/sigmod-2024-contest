#pragma once

#include "hnswlib.h"
#include "visited_list_pool.h"
#include "glass/memory.hpp"
#include "glass/hash_table5.hpp"
#include <cassert>
#include <atomic>
#include <list>
#include <random>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <bitset>

namespace hnswlib {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;
template<typename dist_t>
struct Neighbor {
    int id;
    dist_t distance;

    Neighbor() = default;
    Neighbor(int id, dist_t distance) : id(id), distance(distance) {}
};

template <typename dist_t>
 struct LinearPool {
//     int last_plc=-1;
  LinearPool(int capacity, int = 0)
      : capacity_(capacity), data_(capacity_ + 1) {}

  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance >= dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }
  bool emplace(dist_t dist, int u) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
//        last_plc=cur_;
      cur_ = lo;
    }
//    else last_plc=cur_+1;
    return true;
  }
  void emplace_back(dist_t dist, int u) {
    data_[size_++] = {u, dist};
  }
//  std::pair<float, tableint> top() {
//    return std::make_pair(data_[cur_].distance, data_[cur_].id);
//  }
    inline dist_t top_cand(int efc){
      return size_>=efc?data_[efc - 1].distance:data_[size_ - 1].distance;
  }
    void finish_search(int efc) {
        size_=std::min(size_,efc);
        for(int i=0;i<size_;i++)data_[i].id=get_id(data_[i].id);
    }
    std::pair<dist_t, tableint> pop() {
        set_checked(data_[cur_].id);
        int pre = cur_;
    //    cur_=last_plc;
        while (cur_ < size_ && is_checked(data_[cur_].id)) {
        cur_++;
        }
    //    last_plc=cur_+1;
        return std::make_pair(data_[pre].distance,get_id(data_[pre].id));
    }
        std::pair<dist_t, tableint> heap_top() {
            return std::make_pair(data_[size_ - 1].distance, data_[size_ - 1].id);
        }
    void heap_pop() {
        size_--;
    }
 inline bool has_next() const { return cur_ < size_; }
 inline int id(int i) const { return get_id(data_[i].id); }
 inline int size() const { return size_; }
 inline int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  inline int get_id(int id) const { return id & kMask; }
  inline void set_checked(int &id) { id |= 1 << 31; }
  inline bool is_checked(int id) { return id >> 31 & 1; }

  int size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>, glass::align_alloc<Neighbor<dist_t>>> data_;
};
//struct maxPQFICS
//{
//    int size_ = 0;
//    int cur_ = 0;
//    int capacity_;
//    std::vector<Neighbor, glass::align_alloc<Neighbor>> data_;
//    explicit maxPQFICS(int n) : capacity_(n), data_(n + 5){
//    }
//    inline void clear() {
//        size_=0;
//    }
//    inline bool empty(){
//        return size_==0;
//    }
//    int find_bsearch(float dist) {
//        int lo = 1, hi = size_ + 1;
//        while (lo < hi) {
//        int mid = (lo + hi) / 2;
//        if (data_[mid].distance >= dist) {
//            hi = mid;
//        } else {
//            lo = mid + 1;
//        }
//        }
//        return lo;
//    }
//    void emplace(float dist,int id)
//    {
//        int lo = find_bsearch(dist);
//        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo + 1) * sizeof(Neighbor));
//        data_[lo] = {id, dist};
//        if (size_ < capacity_) {
//            size_++;
//        }
//    }
//    inline void emplace_back(float dist,int id) {
//        // data[++size_]=dists;
//        // id[size_]=ids;
//        data_[++size_] = {id, dist};
//    }
//    inline std::pair<float,int> top(){
//        return std::make_pair(data_[size_].distance, data_[size_].id);
//    }
//    inline void pop(){
//        size_--;
//    }
//    inline void pop_tail(){
//
//    }
//    inline int size(){
//        return size_;
//    }
//};


// SQ8U: SQ8 unsigned, asymmetric
enum QuantType { NONE = 0, SQ8 = 1, SQ8U = 2};
    template <typename dist_t, QuantType qtype = QuantType::NONE>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
        static const unsigned char DELETE_MARK = 0x01;

  static constexpr bool sq_enabled = qtype != QuantType::NONE;
  static constexpr bool sq_asymmetric = qtype == QuantType::SQ8U;
  static constexpr int code_size = 112;

        size_t max_elements_{0};
        mutable std::atomic<size_t> cur_element_count{
                0}; // current number of elements
        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        mutable std::atomic<size_t> num_deleted_{0}; // number of deleted elements
        size_t M_{0};
        size_t maxM_{0};
        size_t maxM0_{0};
        size_t ef_construction_{0};
        size_t ef_{0};

        double mult_{0.0}, revSize_{0.0};
        int maxlevel_{0};

        VisitedListPool *visited_list_pool_{nullptr};

        std::mutex global;
        std::vector<std::mutex> link_list_locks_;

        // Locks operations with element by label value
        mutable std::vector<std::mutex> label_op_locks_;

        tableint enterpoint_node_{0};

        size_t size_links_level0_{0};
        size_t offsetData_{0}, offsetSQData_{0}, offsetLevel0_{0}, label_offset_{0};

        char *data_level0_memory_{nullptr};
        char **linkLists_{nullptr};
        std::vector<int> element_levels_; // keeps level of each element

        size_t data_size_{0};

        DISTFUNC<dist_t> fstdistfunc_;
        DISTFUNC<int> fstdistfunc_sq_;
  void *dist_func_param_{nullptr};

        mutable std::mutex label_lookup_lock; // lock for label_lookup_
        // std::unordered_map<labeltype, tableint> label_lookup_;
        emhash5::HashMap<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        mutable std::atomic<long> metric_distance_computations{0};
        mutable std::atomic<long> metric_hops{0};

        bool allow_replace_deleted_ = false; // flag to replace deleted elements
        // (marked as deleted) during insertions

        std::mutex deleted_elements_lock; // lock for deleted_elements
        std::unordered_set<tableint>
                deleted_elements; // contains internal ids of deleted elements

  // symmetric quantization
  float alpha_ = 0.0f;
  char* sq_codes = nullptr;
  std::vector<float> mx, mi, dif;

  // prefetch
  size_t po = 5;

    void
    trainSQuant(const float* train_data, size_t ntrain) {
        if constexpr (sq_asymmetric) {
            size_t d = 100;
            size_t d_align = 112;
            for (int64_t i = 0; i < ntrain; ++i) {
                for (int64_t j = 0; j < d; ++j) {
                    mx[j] = std::max(mx[j], train_data[i * d + j]);
                    mi[j] = std::min(mi[j], train_data[i * d + j]);
                }
            }
            for (int64_t j = 0; j < d; ++j) {
                dif[j] = mx[j] - mi[j];
            } 
            for (int64_t j = d; j < d_align; ++j) {
                dif[j] = mx[j] = mi[j] = 0;
            }
        } else if constexpr (sq_enabled) {
            alpha_ = 0.0f;
            size_t dim = 100;
            for (size_t i = 0; i < ntrain; ++i) {
                const float* vec = train_data + i * dim;
                for (size_t j = 0; j < dim; ++j) {
                    alpha_ = std::max(alpha_, std::abs(vec[j]));
                }
            }
        }
 
    }

    // Symmetric
    void
    encodeSQuant(const float* from, int8_t* to) const {
        size_t dim = 100;
        for (size_t i = 0; i < dim; ++i) {
            float x = from[i] / alpha_;
            if (x > 1.0f) {
                x = 1.0f;
            }
            if (x < -1.0f) {
                x = -1.0f;
            }
            to[i] = std::round(x * 127.0f);
        }
    }

    // Asymmetric
    void encodeSQuantU(const float* from, uint8_t* to) const {
        size_t dim = 100;
        for (size_t i = 0; i < dim; ++i) {
            float x = (from[i] - mi[i]) / dif[i];
            if (x < 0) {
                x = 0.0;
            }
            if (x > 1.0) {
                x = 1.0;
            }
            uint8_t y = x * 255;
            to[i] = y;
        }
    }

  inline char*
    getSQDataByInternalId(tableint internal_id) const {
        if constexpr (sq_asymmetric) {
            return sq_codes + internal_id * code_size;
        } else if constexpr (sq_enabled) {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetSQData_);
        }
    }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location,
                        bool /**nmslib*/ = false, size_t max_elements = 0,
                        bool allow_replace_deleted = false)
                : allow_replace_deleted_(allow_replace_deleted) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
                        size_t ef_construction = 200, size_t random_seed = 100,
                        bool allow_replace_deleted = false)
                : link_list_locks_(max_elements),
                  label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
                  element_levels_(max_elements),
                  allow_replace_deleted_(allow_replace_deleted) {
            max_elements_ = max_elements;
            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            // fstdistfunc_ = s->get_dist_func();
            if constexpr (sq_enabled) {
      fstdistfunc_sq_ = s->get_dist_func_sq();
    }
    dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction, M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            // size_data_per_element_ =
            //         size_links_level0_ + data_size_ + sizeof(labeltype);
            size_data_per_element_ = size_links_level0_ + sizeof(labeltype); // no need to store original float vectors
            if constexpr (sq_asymmetric) {
                size_data_per_element_ += code_size * sizeof(uint8_t);
            }
            else if constexpr (sq_enabled) {
                size_data_per_element_ += code_size * sizeof(int8_t);
            }
            offsetData_ = size_links_level0_;
            // label_offset_ = size_links_level0_ + data_size_;
            label_offset_ = size_links_level0_;
            if constexpr (sq_enabled) {
                offsetSQData_ = label_offset_ + sizeof(labeltype);
            }
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *)glass::alloc2M(max_elements_ * size_data_per_element_);
                    // (char *)malloc(max_elements_ * size_data_per_element_);

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements,9);

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            // linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
            linkLists_ = (char**) glass::alloc2M(max_elements_ * sizeof(void *));
            size_links_per_element_ =
                    maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;

            label_lookup_.reserve(max_elements);
  }

        ~HierarchicalNSW() {
            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        struct CompareByFirst {
            constexpr bool
            operator()(std::pair<dist_t, tableint> const &a,
                       std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        inline void setEf(size_t ef) { ef_ = ef; }

        inline std::mutex &getLabelOpMutex(labeltype label) const {
            // calculate hash
            size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
            return label_op_locks_[lock_id];
        }

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,
                   (data_level0_memory_ + internal_id * size_data_per_element_ +
                    label_offset_),
                   sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ +
                    label_offset_),
                   &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *)(data_level0_memory_ +
                                 internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ +
                    offsetData_);
        }

        inline int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int)r;
        }

// Not in use!
inline float reduce_add_f32x16(__m512 x) {
  auto sumh =
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
  auto sumhh =
      _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
  d = 112;
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    // auto zz = _mm_loadu_epi8(y + i);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    auto d = _mm512_sub_ps(_mm512_mul_ps(xx, const_255), yy);
    sum = _mm512_fmadd_ps(d, d, sum);
  }
  return reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f);
    yy = yy * dif[i] + mi[i] * 255.0f;
    auto dif = x[i] * 255.0f - yy;
    sum += dif * dif;
  }
  return sum;
#endif
}
  inline dist_t
  calcDistance(const tableint id1, const tableint id2) {
    if constexpr (sq_asymmetric) {
        float dist = L2SqrSQ8_ext((float*)getDataByInternalId(id1), (uint8_t*)getSQDataByInternalId(id2), 
                112, mi.data(), dif.data());
        return dist;

    } else if constexpr (sq_enabled) {
        return fstdistfunc_sq_(getSQDataByInternalId(id1), getSQDataByInternalId(id2),
                             dist_func_param_);
    } else {
        return fstdistfunc_(getDataByInternalId(id1), getDataByInternalId(id2),
                            dist_func_param_);
    }
  }

  inline dist_t calcDistance(const void* data_point, const tableint id) {
    if constexpr (sq_asymmetric) {
      return L2SqrSQ8_ext((float*)data_point, (uint8_t*)getSQDataByInternalId(id), 
              112, mi.data(), dif.data());
    } else if constexpr (sq_enabled) {
      return fstdistfunc_sq_(data_point, getSQDataByInternalId(id),
                             dist_func_param_) * alpha_ * alpha_ / 127.0f / 127.0f;
    } else {
      return fstdistfunc_(data_point, getDataByInternalId(id),
                          dist_func_param_);
    }
  }

  void prefetchData(const tableint id) const {
    if constexpr (sq_enabled) {
        glass::mem_prefetch(getSQDataByInternalId(id), 2);
    //   _mm_prefetch(getSQDataByInternalId(id), _MM_HINT_T0);
    } else {
      _mm_prefetch(getDataByInternalId(id), _MM_HINT_T0);
    }
  }

        size_t getMaxElements() { return max_elements_; }

        size_t getCurrentElementCount() { return cur_element_count; }

        size_t getDeletedCount() { return num_deleted_; }

//        std::priority_queue<std::pair<dist_t, tableint>,
//                std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        LinearPool<dist_t> searchBaseLayer(tableint ep_id, tableint cur_c, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type *block_array = vl->changed;

            // maxPQFICS top_candidates(ef_construction_);
            LinearPool<dist_t> candidateSet(ef_construction_);
            // std::priority_queue<std::pair<dist_t, tableint>,
            //         std::vector<std::pair<dist_t, tableint>>,
            //         CompareByFirst>
            //  candidateSet;

            dist_t lowerBound;
            
            dist_t dist = calcDistance(cur_c, ep_id);
    //   top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(dist, ep_id);
            // candidateSet.emplace(-dist, ep_id);
            vl->set(ep_id);
            // while (!candidateSet.empty()) {
            while (candidateSet.has_next()) {
                // 当前还没有被遍历的最小的数据点和距离
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.pop();
                if (curr_el_pair.first > lowerBound &&
                    candidateSet.size() >= ef_construction_) {
                    candidateSet.size_ = ef_construction_;
                    break;
                }

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data; // = (int *)(linkList0_ + curNodeNum *
                // size_links_per_element0_);
                if (layer == 0) {
                    data = (int *)get_linklist0(curNodeNum);
                } else {
                    data = (int *)get_linklist(curNodeNum, layer);
                    //                    data = (int *) (linkLists_[curNodeNum] + (layer -
                    //                    1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint *)data);
                tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
                prefetchData(*datal);
                for (int i = 1; i <= po && i < size; ++i) {
    //                _mm_prefetch((char *)(visited_array + ((*(data + 1)))), _MM_HINT_T0);
    //                _mm_prefetch((char *)(visited_array + ((*(data + 1) + 64))), _MM_HINT_T0);
                    _mm_prefetch((char *)(visited_array + ((*(data + i))>>3)), _MM_HINT_T0);
                    _mm_prefetch((char *)(visited_array + ((*(data + i))>>3)+64), _MM_HINT_T0);
                    _mm_prefetch((char *)(block_array + ((*(data + i))>>10)), _MM_HINT_T0);
    //                _mm_prefetch((char *)(block_array + ((*(data + 1))>>10)+64), _MM_HINT_T0);
                    prefetchData(*(datal + i));
                }

#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    if (j + po + 1 < size) {
//                    _mm_prefetch((char *)(visited_array + ((*(datal + j + 1)))), _MM_HINT_T0);
                        _mm_prefetch((char *)(visited_array + ((*(datal + j + po + 1))>>3)), _MM_HINT_T0);
                        _mm_prefetch((char *)(block_array + ((*(datal + j + po + 1))>>10)), _MM_HINT_T0);
                        prefetchData(*(datal + j + po + 1));
                    }
#endif
//                    if (visited_array[candidate_id] == visited_array_tag)continue;
                    if (vl->get(candidate_id))continue;
//                    if((candidate_id>>3)>(max_elements_>>3))exit(555);
//                    visited_array[candidate_id] = visited_array_tag;
                        vl->set(candidate_id);
//                    cnt++;
//                    if(cnt<=10000)vec.emplace_back(candidate_id>>3);
                    // char *currObj1 = (getDataByInternalId(candidate_id));

                    // dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    dist_t dist1 = calcDistance(cur_c, candidate_id);
        // 结果集小于efc 或者 当前距离小于结果集的最大距离
        if (candidateSet.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(dist1, candidate_id);
                        lowerBound = candidateSet.top_cand(ef_construction_);
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

        //    return top_candidates;
        // 设置前efc个结果都没被check
            candidateSet.finish_search(ef_construction_);
//            candidateSet.size_ = std::min<size_t>(ef_construction_, candidateSet.size_);
            return candidateSet;
        }

        template <bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<dist_t, tableint>,
                std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                          BaseFilterFunctor *isIdAllowed = nullptr) const {
        }

        void getNeighborsByHeuristic2(LinearPool<dist_t> &top_candidates,const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }
            int cnt=top_candidates.size();
            Neighbor<dist_t>* neighbors = top_candidates.data_.data();
            top_candidates.size_ = 0;
            for(int i = 0;i < cnt;i++)
            {
                if (top_candidates.size() >= M)break;
                dist_t dist_to_query = neighbors[i].distance;
                bool good = true;
                for (int j = 0;j < top_candidates.size();j++) {
                       dist_t curdist = calcDistance(top_candidates.data_[j].id, neighbors[i].id);
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {//return_list is asc
                    top_candidates.emplace_back(neighbors[i].distance, neighbors[i].id);
                }
            }

//            for (std::pair<dist_t, tableint> curent_pair : return_list) {
//                top_candidates.emplace_back(curent_pair.first, curent_pair.second);
//            }
        }

        inline linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *)(data_level0_memory_ +
                                       internal_id * size_data_per_element_ +
                                       offsetLevel0_);
        }

        inline linklistsizeint *get_linklist0(tableint internal_id,
                                              char *data_level0_memory_) const {
            return (linklistsizeint *)(data_level0_memory_ +
                                       internal_id * size_data_per_element_ +
                                       offsetLevel0_);
        }

        inline linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *)(linkLists_[internal_id] +
                                       (level - 1) * size_links_per_element_);
        }

        inline linklistsizeint *get_linklist_at_level(tableint internal_id,
                                                      int level) const {
            return level == 0 ? get_linklist0(internal_id)
                              : get_linklist(internal_id, level);
        }

        tableint mutuallyConnectNewElement(
                const void *, tableint cur_c,
                LinearPool<dist_t> &top_candidates,
                int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            // 剪枝前top cand大小小于等于efc
            getNeighborsByHeuristic2(top_candidates, M_);

            Neighbor<dist_t>* selectedNeighbors=top_candidates.data_.data();
            tableint next_closest_entry_point = selectedNeighbors[0].id;
            int cnt=top_candidates.size();

            {
                // lock only during the update
                // because during the addition the lock for cur_c is already acquired
                std::unique_lock<std::mutex> lock(link_list_locks_[cur_c],
                                                  std::defer_lock);
                if (isUpdate) {
                    lock.lock();
                }
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                setListCount(ll_cur, cnt);
                tableint *data = (tableint *)(ll_cur + 1);
                for (size_t idx = 0; idx < cnt; idx++) {

                    data[idx] = selectedNeighbors[idx].id;
                }
            }

            for (size_t idx = 0; idx < cnt; idx++) {
                std::unique_lock<std::mutex> lock(
                        link_list_locks_[selectedNeighbors[idx].id]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx].id);
                else
                    ll_other = get_linklist(selectedNeighbors[idx].id, level);

                size_t sz_link_list_other = getListCount(ll_other);

                tableint *data = (tableint *)(ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of
                // `selectedNeighbors[idx]` then no need to modify any connections or run
                // the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        // dist_t d_max = fstdistfunc_(
                        //         getDataByInternalId(cur_c),
                        //         getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
                        dist_t d_max = calcDistance(cur_c, selectedNeighbors[idx].id);
                        // Heuristic:
//                        std::priority_queue<std::pair<dist_t, tableint>,
//                                std::vector<std::pair<dist_t, tableint>>,
//                                CompareByFirst>
                        LinearPool<dist_t> candidates(sz_link_list_other+1);
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                calcDistance(data[j], selectedNeighbors[idx].id),
                                    // fstdistfunc_(getDataByInternalId(data[j]),
                                    //              getDataByInternalId(selectedNeighbors[idx]),
                                    //              dist_func_param_),
                                    data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);
                        //todo:use memcpy to optimize
                        int indx = 0;
                        int cnt = candidates.size();
                        while (candidates.size() > 0) {
                            data[cnt - indx - 1] = candidates.heap_top().second;
                            candidates.heap_pop();
                            indx++;
                        }

                        setListCount(ll_other, cnt);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
                        getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        void resizeIndex(size_t new_max_elements) {

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements,9);

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *)realloc(
                    data_level0_memory_, new_max_elements * size_data_per_element_);
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new =
                    (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_,
                         cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize =
                        element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i]
                                               : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                       size_t max_elements_i = 0) {
            std::ifstream input(location, std::ios::binary);

            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:
            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize, input.cur);
                }
            }

            input.clear();
            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ =
                    maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements,9);

            linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *)malloc(linkListSize);
                    input.read(linkLists_[i], linkListSize);
                }
            }

//    for (size_t i = 0; i < cur_element_count; i++) {
//      if (isMarkedDeleted(i)) {
//        num_deleted_ += 1;
//        if (allow_replace_deleted_)
//          deleted_elements.insert(i);
//      }
//    }

            input.close();

            return;
        }

        template <typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            tableint internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataByInternalId(internalId);
            size_t dim = *((size_t *)dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *)data_ptrv;
            for (int i = 0; i < (int)dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        /*
         * Marks an element with the given label deleted, does NOT really change the
         * current graph.
         */
        void markDelete(labeltype label) {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            tableint internalId = search->second;
            lock_table.unlock();

            markDeletedInternal(internalId);
        }

        /*
         * Uses the last 16 bits of the memory for the linked list size to store the
         * mark, whereas maxM0_ has to be limited to the lower 16 bits, however, still
         * large enough in almost all cases.
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId)) {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
                if (allow_replace_deleted_) {
                    std::unique_lock<std::mutex> lock_deleted_elements(
                            deleted_elements_lock);
                    deleted_elements.insert(internalId);
                }
            }
        }

        /*
         * Removes the deleted mark of the node, does NOT really change the current
         * graph.
         *
         * Note: the method is not safe to use when replacement of deleted elements is
         * enabled, because elements marked as deleted can be completely removed by
         * addPoint
         */
        void unmarkDelete(labeltype label) {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            tableint internalId = search->second;
            lock_table.unlock();

            unmarkDeletedInternal(internalId);
        }

        /*
         * Remove the deleted mark of the node.
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId)) {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
                if (allow_replace_deleted_) {
                    std::unique_lock<std::mutex> lock_deleted_elements(
                            deleted_elements_lock);
                    deleted_elements.erase(internalId);
                }
            }
        }

        /*
         * Checks the first 16 bits of the memory to see if the element is marked
         * deleted.
         */
        inline bool isMarkedDeleted(tableint internalId) const {
            return false;
//    unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
//    if(*ll_cur & DELETE_MARK)std::cout<<"really?????"<<std::endl;
//    return *ll_cur & DELETE_MARK;
        }

        inline unsigned short int getListCount(linklistsizeint *ptr) const {
            return *((unsigned short int *)ptr);
        }

        inline void setListCount(linklistsizeint *ptr, unsigned short int size) const {
            *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
        }

        /*
         * Adds point. Updates the point if it is already in the index.
         * If replacement of deleted elements is enabled: replaces previously deleted
         * point if any, updating it with new point
         */
        void addPoint(const void *data_point, labeltype label,
                      bool replace_deleted = false) {

            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
            addPoint(data_point, label, -1);
            return;
        }




        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *)(data + 1);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        }

        tableint addPoint(const void *data_point, labeltype label, int level) {
            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                //todo:use
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);

//      auto search = label_lookup_.find(label);
//      if (search != label_lookup_.end()) {
//        tableint existingInternalId = search->second;
//        lock_table.unlock();
//
//        if (isMarkedDeleted(existingInternalId)) {
//          unmarkDeletedInternal(existingInternalId);
//        }
//        updatePoint(data_point, existingInternalId, 1.0);
//
//        return existingInternalId;
//      }

                cur_c = cur_element_count;
                cur_element_count++;
                // label_lookup_[label] = cur_c;
                label_lookup_.emplace_unique(label, cur_c);
            }

            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
//    if (level > 0)
//      curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
                   0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            // memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if constexpr (sq_asymmetric) {
        encodeSQuantU((const float*)data_point, (uint8_t*)getSQDataByInternalId(cur_c));
    } else if constexpr (sq_enabled) {
        encodeSQuant((const float*)data_point, (int8_t*)getSQDataByInternalId(cur_c));
    }

            if (curlevel) {
                linkLists_[cur_c] =
                        // (char *)malloc(size_links_per_element_ * curlevel + 1);
                        (char *) glass::alloc64B(size_links_per_element_ * curlevel + 1);
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) {
                if (curlevel < maxlevelcopy) {
                    // dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj),
                    //                               dist_func_param_);
        dist_t curdist = calcDistance(cur_c, currObj);
                    for (int level = maxlevelcopy; level > curlevel; level--) {
                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *)(data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                // dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand),
                                //                         dist_func_param_);
              dist_t d = calcDistance(cur_c, cand);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

//      bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    LinearPool<dist_t> top_candidates = searchBaseLayer(currObj, cur_c, level);

            // if (level == 0) {
            //     currObj = mutuallyConnectNewElement0(data_point, cur_c, top_candidates,
            //                                             level, false);
            // } {
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates,
                                                        level, false);
            // }
                }
            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        }

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k,
                  BaseFilterFunctor *isIdAllowed = nullptr) const {

        }

        void checkIntegrity() {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++) {
                for (int l = 0; l <= element_levels_[i]; l++) {
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *)(ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j = 0; j < size; j++) {
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if (cur_element_count > 1) {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++) {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked
                      << " connections\n";
        }
    };
} // namespace hnswlib



//todo:change here