#ifndef ONEAPI_CRYSTAL_JOIN_DPP_HPP
#define ONEAPI_CRYSTAL_JOIN_DPP_HPP
#pragma once 

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <CL/sycl/accessor.hpp>

#define HASH(X,Y,Z) ((X-Z) % Y)

/**
 * @brief Sycl version of the atomiCAS function 
 *        natively existing in cuda
 *        Performs atomically the following comparison and
 *        assignment: 
 *             *addr = *addr == expected ?  desired : *addr;
 * @returns the old value at addr
 */
template <typename T, sycl::access::address_space 
          addressSpace = sycl::access::address_space::global_space>
T atomicCAS (
    T *addr, 
    T expected, 
    T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed
)
{
  // add a pair of parentheses to declare a variable
  sycl::atomic<T, addressSpace> obj((sycl::multi_ptr<T, addressSpace>(addr)));
  obj.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}


namespace crystal {
        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_direct_1 (
                int tid, 
                K (&items)[items_per_thread],
                int (&selection_flags)[items_per_thread], K *ht,
                int ht_len, K keys_min
        ) 
        {
                #pragma unroll
                for (int i = 0; i < items_per_thread; i++) {
                  if (selection_flags[i]) {
                      int hash = HASH(items[i], ht_len, keys_min);
                                
                      K slot = ht[hash];
                      selection_flags[i] = slot != 0 ? 1 : 0;
                   }
                }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_direct_1 (
                int tid,
                K (&items)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        ) 
        {
          #pragma unroll
          for (int i = 0; i < items_per_thread; i++) {
            if (tid + (i * block_threads) < num_items) {
                if (selection_flags[i]) {
                  int hash = HASH(items[i], ht_len, keys_min);

                  K slot = ht[hash];
                  selection_flags[i] = slot != 0 ? 1 : 0;
                }
            }
          }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_1 (
                K (&items)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len,
                K keys_min, 
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
                if ((block_threads * items_per_thread) == num_items) {
                        probe_direct_1<K, block_threads, items_per_thread>(
                                item_ct1.get_local_id(0), items, selection_flags, ht, ht_len, keys_min);
                } else {
                        probe_direct_1<K, block_threads, items_per_thread>(
                                item_ct1.get_local_id(0), items, selection_flags, ht, ht_len, keys_min,
                                num_items);
                }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_1(
                K (&items)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len,
                int num_items, sycl::nd_item<1> item_ct1
        ) 
        {
          probe_1<K, block_threads, items_per_thread>(
                items, selection_flags, ht, ht_len, 0,  num_items, item_ct1);
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_direct_2 (
                int tid, 
                K (&keys)[items_per_thread], 
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len,
                K keys_min
        ) 
        {
          #pragma unroll
          for (int i = 0; i < items_per_thread; i++) {
            if (selection_flags[i]) {
                int hash = HASH(keys[i], ht_len, keys_min);

                uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
                if (slot != 0) {
                   res[i] = (slot >> 32);
                } else {
                   selection_flags[i] = 0;
                }
            }
          }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_direct_2 (
                int tid, K (&items)[items_per_thread],
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        ) 
        {
           #pragma unroll
           for (int i = 0; i < items_per_thread; i++) {
                if (tid + (i * block_threads) < num_items) {
                   if (selection_flags[i]) {
                      int hash = HASH(items[i], ht_len, keys_min);

                       uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
                        if (slot != 0) {
                           res[i] = (slot >> 32);
                        } else {
                          selection_flags[i] = 0;
                        }
                   }
                }
           }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_2 (
                K (&keys)[items_per_thread], 
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len,
                K keys_min, 
                int num_items,
                sycl::nd_item<1> item_ct1
        ) 
        {
                if ((block_threads * items_per_thread) == num_items) {
                        probe_direct_2<K, V, block_threads, items_per_thread>(
                                item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                                keys_min);
                } else {
                        probe_direct_2<K, V, block_threads, items_per_thread>(
                                item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                                keys_min, num_items);
                }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_2(
                K (&keys)[items_per_thread], 
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len,
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
           probe_2<K, V, block_threads, items_per_thread>(
                keys, res, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void build_direct_selective_1 (
                int tid,
                K (&keys)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, 
                K keys_min
        ) 
        {
          #pragma unroll
          for (int i = 0; i < items_per_thread; i++) {
                if (selection_flags[i]) {
                  int hash = HASH(keys[i], ht_len, keys_min);
                  atomicCAS(&ht[hash], 0, keys[i]); 
                }       
           }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void build_direct_selective_1 (
                int tid, 
                K (&items)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        )
        {
          #pragma unroll
          for (int i = 0; i < items_per_thread; i++) {
             if (tid + (i * block_threads) < num_items) {
                if (selection_flags[i]) {
                   int hash = HASH(items[i], ht_len, keys_min);

                   atomicCAS(&ht[hash], 0, items[i]);
                }
              }
           }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void build_selective_1(
                K (&keys)[items_per_thread], 
                int (&selection_flags)[items_per_thread],
                K *ht, 
                int ht_len, 
                K keys_min, 
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {

          if ((block_threads * items_per_thread) == num_items) {
                build_direct_selective_1<K, block_threads, items_per_thread>(
                        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min);
          } else {
                build_direct_selective_1<K, block_threads, items_per_thread>(
                        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min,
                        num_items);
          }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void build_selective_1(
                K (&keys)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, 
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
          build_selective_1<K, block_threads, items_per_thread>(
                keys, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void build_direct_selective_2(
                int tid, 
                K (&keys)[items_per_thread], 
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len, 
                K keys_min
        ) 
        {
          #pragma unroll
          for (int i = 0; i < items_per_thread; i++) {
                if (selection_flags[i]) {
                   int hash = HASH(keys[i], ht_len, keys_min);

                   atomicCAS(&ht[hash << 1], 0, keys[i]);

                   ht[(hash << 1) + 1] = res[i];
                }
           }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void build_direct_selective_2(
                int tid, K (&keys)[items_per_thread],
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        ) 
        {
          #pragma unroll
          for (int i = 0; i < items_per_thread; i++) {
             if (tid + (i * block_threads) < num_items) {
                if (selection_flags[i]) {
                    int hash = HASH(keys[i], ht_len, keys_min);

                    atomicCAS(&ht[hash << 1], 0, keys[i]);
                    ht[(hash << 1) + 1] = res[i];
                }
              }
          }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void build_selective_2(
                K (&keys)[items_per_thread], 
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len, 
                K keys_min,
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
          if ((block_threads * items_per_thread) == num_items) {
                build_direct_selective_2<K, V, block_threads, items_per_thread>(
                        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                        keys_min);
          } else {
                build_direct_selective_2<K, V, block_threads, items_per_thread>(
                        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                        keys_min, num_items);
          }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void build_selective_2(
                K (&keys)[items_per_thread],
                V (&res)[items_per_thread],
                int (&selection_flags)[items_per_thread],
                K *ht,
                int ht_len,
                int num_items,
                sycl::nd_item<1> item_ct1
        ) 
        {
          build_selective_2<K, V, block_threads, items_per_thread>(
                keys, res, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }        
} // namespace crystal 

#endif //ONEAPI_CRYSTAL_JOIN_DPP_HPP