#pragma once

#include <algorithm>
#include <mutex>
#include <random>
#include <unordered_set>
#include <list>

namespace glass {

using LockGuard = std::lock_guard<std::mutex>;

inline void GenRandom(std::mt19937 &rng, int *addr, const int size,
                      const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() {
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
  }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return mt() % max; }

  /// between 0 and 1
  float rand_float() { return mt() / float(mt.max()); }

  double rand_double() { return mt() / double(mt.max()); }
};

template <typename key_t, typename value_t>
struct LRUCache {
  using key_value_pair_t = std::pair<key_t, value_t>;
  using list_iterator_t = typename std::list<key_value_pair_t>::iterator;

  LRUCache(size_t cap = kDefaultSize) : capacity(cap) {}

  void put(const key_t& key, const value_t& value) {
    std::unique_lock lock(mtx);
    auto it = map.find(key);
    list.push_front(key_value_pair_t(key, value));
    if (it != map.end()) {
      map.erase(it);
      list.erase(it->second);
    }
    map[key] = list.begin();
    if (map.size() > capacity) {
      auto last = list.end();
      last--;
      map.erase(last->first);
      list.pop_back();
    }
  }

  bool try_get(const key_t& key, value_t& value) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = map.find(key);
    if (it == map.end()) {
      return false;
    }
    // std::cout << "cache hit!" << std::endl;
    list.splice(list.begin(), list, it->second);
    value = it->second->second;
    return true;
  }

  std::list<key_value_pair_t> list;
  std::unordered_map<key_t, list_iterator_t> map;
  std::mutex mtx;
  size_t capacity;
  constexpr static size_t kDefaultSize = 10000;

};

constexpr inline uint64_t seed = 0xc70f6907UL;

inline uint64_t
hash_vec(const float* x, int d) {
  uint64_t h = seed;
  for (size_t i = 0; i < d; ++i) {
    h = h * 13331 + *(uint32_t*)(x + i);
  }
  return h;
}

} // namespace glass