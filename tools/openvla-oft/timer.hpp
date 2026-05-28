#pragma once

#include <chrono>
#include <thread>

class Timer {
public:
  using Seconds = std::ratio<1>;
  using Milliseconds = std::milli;
  using Microseconds = std::micro;
  using Nanoseconds = std::nano;

  using s = Seconds;
  using ms = Milliseconds;
  using us = Microseconds;
  using ns = Nanoseconds;

public:
  Timer(const bool eager = false) {
    if (eager) {
      start();
    }
  }

  static double time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_epoch = now.time_since_epoch();
    return std::chrono::duration<double>(time_since_epoch).count();
  }

  // sleep
  template <typename Unit = Seconds, typename Rep>
  static void sleep(Rep duration) {
    std::this_thread::sleep_for(std::chrono::duration<Rep, Unit>(duration));
  }

  void start() { mStart = std::chrono::high_resolution_clock::now(); }

  template <typename Unit = Seconds> double elapsed() const {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, Unit>(now - mStart).count();
  }

  template <typename Unit = Seconds> double stop() {
    const auto now = std::chrono::high_resolution_clock::now();
    const double elapsed =
        std::chrono::duration<double, Unit>(now - mStart).count();
    mStart = now;
    return elapsed;
  }


private:
  std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
};