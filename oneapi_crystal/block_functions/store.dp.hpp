#ifndef ONEAPI_CRYSTAL_STORE_HPP
#define ONEAPI_CRYSTAL_STORE_HPP
#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace crystal {
  
    template < ypename T, int block_threads, int items_per_thread>
    __dpct_inline__ void block_store_direct (
            int tid, 
            T *block_itr,
            T (&items)[items_per_thread]
    ) 
    {
        // thread iterator
        T* thread_itr = block_itr + tid;

        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            thread_itr[i * block_threads] = items[i];
        }
    }

    
    template <typename T, int block_threads, int items_per_thread>
    __dpct_inline__ void block_store_direct(
            int tid, 
            T *block_itr,
            T (&items)[items_per_thread],
            int num_items 
    )
    {

        T* thread_itr = block_itr + tid;

        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            if (tid + (i * block_threads) < num_items)
                thread_itr[i * block_threads] = items[i];
        }
    }


    template <typename T, int block_threads, int items_per_thread>
    __dpct_inline__ void block_store (
            T *out,
            T (&items)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    ) 
    {
        T* block_itr = out;

        if ((block_threads * items_per_thread) == num_items) {
            block_store_direct<T, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), block_itr, items);
        } else {
            block_store_direct<T, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), block_itr, items, num_items);
        }
    }

} // namespace crystal

#endif //ONEAPI_CRYSTAL_STORE_HPP
