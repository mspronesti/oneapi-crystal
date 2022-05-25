#ifndef ONEAPI_CRYSTAL_SYCL_QUEUE_HELPERS_HPP
#define ONEAPI_CRYSTAL_SYCL_QUEUE_HELPERS_HPP
#pragma once

#include <iostream>
#include <CL/sycl.hpp>

namespace crystal {

    class queue_tester;

    void queue_tester(sycl::queue &q) {
        q.submit([](sycl::handler &h){
            h.single_task<class queue_tester>([]() {});
        })
        .wait_and_throw();
    }

    /**
     * @brief Tries to acquire a queue from a given device
     *        or retrieves the host device
     * @tparam T            selector type
     * @param selector      selector attempted to use
     * @return sycl::queue  retrieved queue
     */
    inline sycl::queue try_get_queue_with_dev(const sycl::device &in_dev){
            // exception handler to be used inside 
        auto exception_handler = [](const sycl::exception_list &exceptions){
            for (std::exception_ptr const &e : exceptions) {
                try {
                    std::rethrow_exception(e);
                }
                catch (sycl::exception const &e) {
                    std::cout << "Caught asynchronous SYCL exception: " << e.what() << '\n';
                }
                catch (std::exception const &e) {
                    std::cout << "Caught asynchronous STL exception: " << e.what() << '\n';
                }
            }
        };
        
        sycl::device dev;
        sycl::queue q;

        try {
            dev = in_dev;
            q = sycl::queue(dev, exception_handler);

            try {
                // test queue is indeed working
                // otherwise catch the exception
                // thrown and fall back on default selector
                queue_tester(q);

            } catch(...){
                dev = sycl::device(sycl::host_selector());
                q = sycl::queue(dev, exception_handler);
                std::cerr << "[Warning] " << dev.get_info<sycl::info::device::name>()
                        << " found but not working! Fall back on "
                        << dev.get_info<sycl::info::device::name>() << '\n';
            }
            
        } catch (...) {
        dev = sycl::device(sycl::host_selector());
        q = sycl::queue(dev, exception_handler);

        std::cerr << "[Warning] Expected device not found! Fall back on: " 
                    << dev.get_info<sycl::info::device::name>() << '\n';
        } 

        return q;
    }    


    /**
     * @brief Tries to acquire a queue from a given selector
     *        or retrieves the host device
     * @tparam T            selector type
     * @param selector      selector attempted to use
     * @return sycl::queue  retrieved queue
     */
    template<
        typename T
        >
    inline sycl::queue try_get_queue(const T & selector) {
        // TODO: consider replacing this with a C++20 concept
        static_assert(
            std::is_base_of<sycl::device_selector, T>::value,
            "Chosen type is not a valid sycl device selector"
        );

        // attempt using the provided selector
        // to get a device
        sycl::device dev;
        try {
            dev = sycl::device(selector);
        } catch(...) {
            dev = sycl::device(sycl::host_selector());
            std::cerr << "[Warning] Expected device not found! Fall back on: " 
                    << dev.get_info<sycl::info::device::name>() << '\n';
        }

        return try_get_queue_with_dev(dev);
    }


    template <
        typename T
        >
    T *load_to_device (
        T *src, 
        unsigned int size, 
        sycl::queue &q
    ) 
    { 
        try {
            T* dest;
            dest = (T*)malloc_device(sizeof(T) * size, q);

            q.memcpy(dest, src, sizeof(T) * size).wait();
            return dest;
        }
        catch (sycl::exception const &exc) {
            std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
            std::exit(1);
        }
    }
    

} // namespace crystal 

#endif 