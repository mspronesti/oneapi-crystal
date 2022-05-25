#ifndef ONEAPI_CRYSTAL_JOIN_DPP_HPP
#define ONEAPI_CRYSTAL_JOIN_DPP_HPP
#pragma once 

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <CL/sycl/accessor.hpp>

#define HASH(X,Y,Z) ((X-Z) % Y)

template< typename T >
using global_atomic_ref = sycl::atomic_ref <
        T,
        sycl::detail::memory_order::relaxed,
        sycl::detail::memory_scope::device,
        sycl::access::address_space::global_space
        >;

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
                for (int item = 0; item < items_per_thread; item++) {
                        if (selection_flags[item]) {
                                int hash = HASH(items[item], ht_len, keys_min);
                                K slot = ht[hash];
                                selection_flags[item] = slot != 0 ? 1 : 0;
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
          for (int item = 0; item < items_per_thread; item++) {
            if (tid + (item * block_threads) < num_items) {
                if (selection_flags[item]) {
                  int hash = HASH(items[item], ht_len, keys_min);

                  K slot = ht[hash];
                  selection_flags[item] = slot != 0 ? 1 : 0;
                }
            }
          }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void BlockProbeAndPHT_1 (
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
        __dpct_inline__ void BlockProbeAndPHT_1(
                K (&items)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht, 
                int ht_len,
                int num_items, sycl::nd_item<1> item_ct1
        ) 
        {
          BlockProbeAndPHT_1<K, block_threads, items_per_thread>(
                items, selection_flags, ht, ht_len, 0, item_ct1, num_items);
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
          for (int item = 0; item < items_per_thread; item++) {
            if (selection_flags[item]) {
                int hash = HASH(keys[item], ht_len, keys_min);

                uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
                if (slot != 0) {
                        res[item] = (slot >> 32);
                } else {
                        selection_flags[item] = 0;
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
           for (int item = 0; item < items_per_thread; item++) {
                if (tid + (item * block_threads) < num_items) {
                   if (selection_flags[item]) {
                      int hash = HASH(items[item], ht_len, keys_min);

                       uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
                       selection_flags[item] = slot != 0 ? 1 : 0;
                   }
                }
           }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void probe_2(
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
                int num_items, sycl::nd_item<1> item_ct1
        ) 
        {
           probe_2<K, V, block_threads, items_per_thread>(
                keys, res, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void build_direct_selective(
                int tid, K (&keys)[items_per_thread],
                int (&selection_flags)[items_per_thread], 
                K *ht,
                int ht_len, K keys_min
        ) 
        {
          #pragma unroll
          for (int item = 0; item < items_per_thread; item++) {
                if (selection_flags[item]) {
                  int hash = HASH(keys[item], ht_len, keys_min);
                  global_atomic_ref<K>(ht[hash])
                        .compare_exchange_strong(keys[item], 0);
                }
           }
        }

        template <typename K, int block_threads, int items_per_thread>
        __dpct_inline__ void build_direct_selective(
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
          for (int item = 0; item < items_per_thread; item++) {
             if (tid + (item * block_threads) < num_items) {
                if (selection_flags[item]) {
                   int hash = HASH(items[item], ht_len, keys_min);

                    global_atomic_ref<K>(ht[hash]).compare_exchange_strong(items[item], 0);
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
                build_direct_selective<K, block_threads, items_per_thread>(
                        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min);
          } else {
                build_direct_selective<K, block_threads, items_per_thread>(
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
                keys, selection_flags, ht, ht_len, 0, item_ct1, num_items);
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void build_selective_direct_1(
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
          for (int item = 0; item < items_per_thread; item++) {
                if (selection_flags[item]) {
                int hash = HASH(keys[item], ht_len, keys_min);

                //K old = atomicCAS(&ht[hash << 1], 0, keys[item]);
                global_atomic_ref<K>(ht[hash << 1]).compare_exchange_strong(keys[item], 0);

                ht[(hash << 1) + 1] = res[item];
                }
           }
        }

        template <typename K, typename V, int block_threads, int items_per_thread>
        __dpct_inline__ void build_selective_direct_1(
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
          for (int item = 0; item < items_per_thread; item++) {
             if (tid + (item * block_threads) < num_items) {
                if (selection_flags[item]) {
                    int hash = HASH(keys[item], ht_len, keys_min);

                    global_atomic_ref<K>(ht[hash << 1]).compare_exchange_strong(keys[item], 0);
                    ht[(hash << 1) + 1] = res[item];
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
                build_selective_direct_1<K, V, block_threads, items_per_thread>(
                        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                        keys_min);
          } else {
                build_selective_direct_1<K, V, block_threads, items_per_thread>(
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