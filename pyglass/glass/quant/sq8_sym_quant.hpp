#pragma once

#include "glass/common.hpp"
#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/simd/distance.hpp"
#include "glass/thread_pool.hpp"

#include <cmath>
#include <vector>

namespace glass {

template <Metric metric, int DIM = 0> struct SQ8SymmetricQuantizer {
  using data_type = int8_t;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  float alpha = 0.0f;
  float alpha_square = 0.0f;

  SQ8SymmetricQuantizer() = default;

  explicit SQ8SymmetricQuantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        alpha(0.0f) {}

  ~SQ8SymmetricQuantizer() { free(codes); }

  void train(const float *data, int n) {
    ctpl::thread_pool pool(32);
    constexpr size_t block_size = 256;
    size_t block_num = (n + block_size - 1) / block_size;

    for (size_t i = 0; i < n; ++i) {
        const float* vec = data + i * d;
        for (size_t j = 0; j < d; ++j) {
            alpha = std::max(alpha, std::abs(vec[j]));
        }
    }
    alpha_square = alpha * alpha;
    codes = (char *)alloc2M((size_t)n * code_size);

    for (size_t i = 0; i < block_num; ++i) {
      size_t start = i * block_size;
      size_t end = std::min((i + 1) * block_size, (size_t)n);
      pool.push([&, start, end, index = i](int id) {
        for (size_t j = start; j < end; ++j) {
          encode(data + j * d, get_data(j));
        }
      });
    }
    pool.stop(true);
    // for (int i = 0; i < n; ++i) {
    //   encode(data + i * d, get_data(i));
    // }
  }

  char *get_data(int u) const { return codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (size_t i = 0; i < d; ++i) {
        float x = from[i] / alpha;
        if (x > 1.0f) {
            x = 1.0f;
        }
        if (x < -1.0f) {
            x = -1.0f;
        }
        to[i] = std::round(x * 127.0f);
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float * /**q*/, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <typename Pool>
  void reorder_dist(const Pool &pool, const float * /**q*/, int *dst, float* dists, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
      dists[i] = pool.dist(i) * alpha_square;
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = int;
    constexpr static auto dist_func = L2SqrSQ8_sym;
    const SQ8SymmetricQuantizer &quant;
    int8_t *q;
    float alpha;
    Computer(const SQ8SymmetricQuantizer &quant, const float *query)
        : quant(quant), q((int8_t*)alloc64B(quant.code_size)), alpha(quant.alpha)
           {
      quant.encode(query, (char*)q);
    //   std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass