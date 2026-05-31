#pragma once
// timer.h: wall-clock timer on steady_clock. Backend agnostic, the
// surrounding code synchronizes the device through a tensor readback
// before reading ms() so the measured span covers the GPU work.

#include <chrono>

struct Timer {
    std::chrono::steady_clock::time_point t;

    Timer() : t(std::chrono::steady_clock::now()) {}

    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    }

    void reset() { t = std::chrono::steady_clock::now(); }
};
