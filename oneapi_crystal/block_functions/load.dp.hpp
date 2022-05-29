#ifndef ONEAPI_CRYSTAL_LOAD_HPP
#define ONEAPI_CRYSTAL_LOAD_HPP
#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace crystal {

    template <typename T, int block_threads, int items_per_thread>
    SYCL_EXTERNAL __dpct_inline__ void load_direct (
            const unsigned int tid,
            T *block_itr,
            T (&items)[items_per_thread]
    )
    {
        T* thread_itr = block_itr + tid;

        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            items[i] = thread_itr[i * block_threads];
        }
    }

    template <typename T, int block_threads, int items_per_thread>
    SYCL_EXTERNAL __dpct_inline__ void load_direct (
            const unsigned int tid,
            T *block_itr,
            T (&items)[items_per_thread],
            int num_items
    ) 
    {
        T* thread_itr = block_itr + tid;

        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            if (tid + (i * block_threads) < num_items) {
                items[i] = thread_itr[i * block_threads];
            }
        }
    }

    template <typename T, int block_threads, int items_per_thread>
    __dpct_inline__ void load(
            T *inp,
            T (&items)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    ) 
    {
        T* block_itr = inp;

        if ((block_threads * items_per_thread) == num_items) {
            load_direct<T, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), block_itr, items);
        } else {
            load_direct<T, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), block_itr, items, num_items);
        }
    }
               
} // namespace crystal

#endif //ONEAPI_CRYSTAL_STORE_HPP