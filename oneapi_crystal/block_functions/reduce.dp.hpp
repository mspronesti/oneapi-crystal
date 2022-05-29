#ifndef ONEAPI_CRYSTAL_REDUCE_HPP
#define ONEAPI_CRYSTAL_REDUCE_HPP
#pragma once 

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace crystal {

 
    template <
            typename T,
            int block_threads,
            int items_per_thread
            >
    __dpct_inline__ T reduce (T item, T *shared, sycl::nd_item<1> item_ct1) 
    {

        item_ct1.barrier();

        /**
         * ADDED BY ME BUT NOT USED... TO CHECK CORRECTNESS
         * **/

        T val = item;

        const int warp_size = item_ct1.get_sub_group().dimensions;

        int lane = item_ct1.get_local_id(0) % warp_size;
        int wid = item_ct1.get_local_id(0) / warp_size;

        for (int offset = 16; offset > 0; offset /= 2) {
            //__shfl_down_sync(0xffffffff, val, offset);
            val += item_ct1.get_sub_group().template shuffle_down(val, offset);
        }
        if (lane == 0) {
            shared[wid] = val;
        }

        item_ct1.get_sub_group().barrier();
        // Load the sums into the first warp
        val = (item_ct1.get_local_id(0) < item_ct1.get_local_range().get(0) / warp_size) ?
                shared[lane] : 0;

        // Calculate reduce of sums
        if (wid == 0) {
            for (int offset = 16; offset > 0; offset /= 2)
                //__shfl_down_sync(0xffffffff, val, offset);
                val += item_ct1.get_sub_group().template shuffle_down(val, offset);
        }

        return val;
    }

    template < typename T, int block_threads, int items_per_thread >
    __dpct_inline__ T reduce(
            T (&items)[items_per_thread],
            T *shared,
            sycl::nd_item<1> item_ct1
    ) 
    {
        T thread_sum = 0;

        #pragma unroll
        for (int i = 0; i < items_per_thread; ++i) {
            thread_sum += items[i];
        }
        return reduce(thread_sum, shared, item_ct1);
    }   

} //namespace crystal

#endif //ONEAPI_CRYSTAL_REDUCE_HPP
