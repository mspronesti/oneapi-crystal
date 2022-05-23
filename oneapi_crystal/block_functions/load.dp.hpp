#ifndef ONEAPI_CRYSTAL_LOAD_HPP
#define ONEAPI_CRYSTAL_LOAD_HPP
#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace crystal {

    template <typename T, int block_threads, int items_per_thread>
    SYCL_EXTERNAL __dpct_inline__ void block_load_direct (
            const unsigned int tid,
            T *block_itr,
            T (&items)[items_per_thread]
    )
    {
        T* thread_itr = block_itr + tid;

        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
            items[item] = thread_itr[item * block_threads];
        }
    }

    template <typename T, int block_threads, int items_per_thread>
    SYCL_EXTERNAL __dpct_inline__ void block_load_direct (
            const unsigned int tid,
            T *block_itr,
            T (&items)[items_per_thread],
            int num_items
    ) 
    {
        T* thread_itr = block_itr + tid;

        #pragma unroll
        for (int item = 0; item < items_per_thread; item++) {
            if (tid + (item * block_threads) < num_items) {
                items[item] = thread_itr[item * block_threads];
            }
        }
    }

    template <typename T, int block_threads, int items_per_thread>
    __dpct_inline__ void block_load(
            T *inp,
            T (&items)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    ) 
    {
        T* block_itr = inp;

        if ((block_threads * items_per_thread) == num_items) {
            block_load_direct<T, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), block_itr, items);
        } else {
            block_load_direct<T, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), block_itr, items, num_items);
        }
    }
               
} // namespace crystal

#endif //ONEAPI_CRYSTAL_STORE_HPP