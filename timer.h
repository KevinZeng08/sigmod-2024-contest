#pragma once

#include <chrono>
#include <cmath>
#include <iostream>

template <typename T = std::chrono::nanoseconds>
class Timer {
    std::chrono::steady_clock::time_point last_point_;
    std::chrono::steady_clock::duration elapsed_time_{0};

public:
    Timer() = default;

    inline T getElapsedTime() {
        // auto time_end = std::chrono::steady_clock::now();
        // return std::chrono::duration_cast<T>(time_end - time_begin_);
        return std::chrono::duration_cast<T>(elapsed_time_);
    }

    inline void reset() {
        elapsed_time_ = std::chrono::steady_clock::duration::zero();
        last_point_ = std::chrono::steady_clock::now();
    }

    inline void start() {
        last_point_ = std::chrono::steady_clock::now();
    }


    inline void end() {
        const auto time_end = std::chrono::steady_clock::now();
        elapsed_time_ += time_end - last_point_;
    }


    inline double total() {
        return elapsed_time_.count();
    }
};
