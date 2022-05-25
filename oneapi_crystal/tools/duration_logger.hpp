#ifndef ONEAPI_CRYSTAL_DURATION_LOGGER_HPP
#define ONEAPI_CRYSTAL_DURATION_LOGGER_HPP
#pragma once

#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace std::chrono;

/**
 * @brief Helper class to measure elapsed time
 *        Exploits the object life-cycle paradigm: 
 *         > samples time as object creation  
 *         > samples time at object destruction
 *         > prints the difference
 * 
 *       Usage (in a scope):
 *
 *       {
 *         DurationLogger dl{"foo"} // here time is measured
 *         // do stuff you want to
 *         // benchmark
 *         // ...
 *         // here the object is about to get
 *         // destroyed, thus triggering the destructor
 *       }
 */
class DurationLogger {

public:
    DurationLogger(const std::string &caller)
        : _start(high_resolution_clock::now())
        , _caller(caller) {}

    DurationLogger()
        : _start(high_resolution_clock::now()) {}
    
    DurationLogger(const DurationLogger &) = delete;

    DurationLogger& operator=(const DurationLogger &) = delete;

    ~DurationLogger() {
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - _start);
        double elapsed = static_cast<double>(duration.count()) / 1E6;

        std::cout << (!_caller.empty() ? "[" + _caller + "] " : "")
                  << "Elapsed time: "
                  << elapsed << " s\n";
    }

private:
    std::string _caller;
    time_point<
        high_resolution_clock, 
        duration<long, std::ratio<1, (long int)1E9>>
        > _start;
};



static void event_profiling(sycl::event &e, const std::string &msg) {
    auto start = 
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    
    auto end = 
        e.get_profiling_info<sycl::info::event_profiling::command_end>();
    
    double elapsed = (end - start) / 1E6;
    std::cout << msg << elapsed << " ms\n";
}

#endif 