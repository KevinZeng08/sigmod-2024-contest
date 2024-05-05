#include <iostream>
#include <vector>
#include <future>

#include "thread_pool.h"

int
main() {
    auto pool = ThreadPool::GetSearchPool();
    std::vector<std::future<void>> futures;
    uint32_t nq = 1000;
    std::vector<int> results(nq);
    for (int i = 0; i < nq; ++i) {
        futures.emplace_back(pool->push([&, i] {
            // compute intensive
            for (int j = 0; j < 10000000; ++j) {
                results[i] = i + 1;
            }
        }));
    }
    for (auto &&result : futures) {
        result.get();
    }
    for (int i = 0; i < nq; ++i) {
        std::cout << results[i] << std::endl;
    }
}