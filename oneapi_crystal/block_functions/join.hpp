#ifndef ONEAPI_CRYSTAL_JOIN_DPP_HPP
#define ONEAPI_CRYSTAL_JOIN_DPP_HPP
#pragma once 

#include <CL/sycl.hpp>
#include "oneapi_crystal/utils/atomic.hpp"

#define HASH(X,Y,Z) ((X-Z) % Y)



namespace crystal {
        template <typename K, int work_groups, int group_size>
        inline void probe_direct_1 (
                int tid, 
                K (&items)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                K keys_min
        ) 
        {
                #pragma unroll
                for (int i = 0; i < group_size; i++) {
                  if (selection_flags[i]) {
                      int hash = HASH(items[i], ht_len, keys_min);
                                
                      K slot = ht[hash];
                      selection_flags[i] = slot != 0 ? 1 : 0;
                   }
                }
        }

        template <typename K, int work_groups, int group_size>
        inline void probe_direct_1 (
                int tid,
                K (&items)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        ) 
        {
          #pragma unroll
          for (int i = 0; i < group_size; i++) {
            if (tid + (i * work_groups) < num_items) {
                if (selection_flags[i]) {
                  int hash = HASH(items[i], ht_len, keys_min);

                  K slot = ht[hash];
                  selection_flags[i] = slot != 0 ? 1 : 0;
                }
            }
          }
        }

        template <typename K, int work_groups, int group_size>
        inline void probe_1 (
                K (&items)[group_size],
                int (&selection_flags)[group_size], 
                K *ht, 
                int ht_len,
                K keys_min, 
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
                if ((work_groups * group_size) == num_items) {
                        probe_direct_1<K, work_groups, group_size>(
                                item_ct1.get_local_id(0), items, selection_flags, ht, ht_len, keys_min);
                } else {
                        probe_direct_1<K, work_groups, group_size>(
                                item_ct1.get_local_id(0), items, selection_flags, ht, ht_len, keys_min,
                                num_items);
                }
        }

        template <typename K, int work_groups, int group_size>
        inline void probe_1(
                K (&items)[group_size],
                int (&selection_flags)[group_size], 
                K *ht, 
                int ht_len,
                int num_items, sycl::nd_item<1> item_ct1
        ) 
        {
          probe_1<K, work_groups, group_size>(
                items, selection_flags, ht, ht_len, 0,  num_items, item_ct1);
        }

        template <typename K, typename V, int work_groups, int group_size>
        inline void probe_direct_2 (
                int tid, 
                K (&keys)[group_size], 
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len,
                K keys_min
        ) 
        {
          #pragma unroll
          for (int i = 0; i < group_size; i++) {
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

        template <typename K, typename V, int work_groups, int group_size>
        inline void probe_direct_2 (
                int tid, K (&items)[group_size],
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        ) 
        {
           #pragma unroll
           for (int i = 0; i < group_size; i++) {
                if (tid + (i * work_groups) < num_items) {
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

        template <typename K, typename V, int work_groups, int group_size>
        inline void probe_2 (
                K (&keys)[group_size], 
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht, 
                int ht_len,
                K keys_min, 
                int num_items,
                sycl::nd_item<1> item_ct1
        ) 
        {
                if ((work_groups * group_size) == num_items) {
                        probe_direct_2<K, V, work_groups, group_size>(
                                item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                                keys_min);
                } else {
                        probe_direct_2<K, V, work_groups, group_size>(
                                item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                                keys_min, num_items);
                }
        }

        template <typename K, typename V, int work_groups, int group_size>
        inline void probe_2(
                K (&keys)[group_size], 
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht, 
                int ht_len,
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
           probe_2<K, V, work_groups, group_size>(
                keys, res, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }

        template <typename K, int work_groups, int group_size>
        inline void build_direct_selective_1 (
                int tid,
                K (&keys)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                K keys_min
        ) 
        {
          #pragma unroll
          for (int i = 0; i < group_size; i++) {
                if (selection_flags[i]) {
                  int hash = HASH(keys[i], ht_len, keys_min);
                  atomicCAS(&ht[hash], 0, keys[i]); 
                }       
           }
        }

        template <typename K, int work_groups, int group_size>
        inline void build_direct_selective_1 (
                int tid, 
                K (&items)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        )
        {
          #pragma unroll
          for (int i = 0; i < group_size; i++) {
             if (tid + (i * work_groups) < num_items) {
                if (selection_flags[i]) {
                   int hash = HASH(items[i], ht_len, keys_min);

                   atomicCAS(&ht[hash], 0, items[i]);
                }
              }
           }
        }

        template <typename K, int work_groups, int group_size>
        inline void build_selective_1(
                K (&keys)[group_size], 
                int (&selection_flags)[group_size],
                K *ht, 
                int ht_len, 
                K keys_min, 
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {

          if ((work_groups * group_size) == num_items) {
                build_direct_selective_1<K, work_groups, group_size>(
                        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min);
          } else {
                build_direct_selective_1<K, work_groups, group_size>(
                        item_ct1.get_local_id(0), keys, selection_flags, ht, ht_len, keys_min,
                        num_items);
          }
        }

        template <typename K, int work_groups, int group_size>
        inline void build_selective_1(
                K (&keys)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
          build_selective_1<K, work_groups, group_size>(
                keys, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }

        template <typename K, typename V, int work_groups, int group_size>
        inline void build_direct_selective_2(
                int tid, 
                K (&keys)[group_size], 
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht, 
                int ht_len, 
                K keys_min
        ) 
        {
          #pragma unroll
          for (int i = 0; i < group_size; i++) {
                if (selection_flags[i]) {
                   int hash = HASH(keys[i], ht_len, keys_min);

                   atomicCAS(&ht[hash << 1], 0, keys[i]);

                   ht[(hash << 1) + 1] = res[i];
                }
           }
        }

        template <typename K, typename V, int work_groups, int group_size>
        inline void build_direct_selective_2(
                int tid, K (&keys)[group_size],
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht,
                int ht_len, 
                K keys_min, 
                int num_items
        ) 
        {
          #pragma unroll
          for (int i = 0; i < group_size; i++) {
             if (tid + (i * work_groups) < num_items) {
                if (selection_flags[i]) {
                    int hash = HASH(keys[i], ht_len, keys_min);

                    atomicCAS(&ht[hash << 1], 0, keys[i]);
                    ht[(hash << 1) + 1] = res[i];
                }
              }
          }
        }

        template <typename K, typename V, int work_groups, int group_size>
        inline void build_selective_2(
                K (&keys)[group_size], 
                V (&res)[group_size],
                int (&selection_flags)[group_size], 
                K *ht, 
                int ht_len, 
                K keys_min,
                int num_items, 
                sycl::nd_item<1> item_ct1
        ) 
        {
          if ((work_groups * group_size) == num_items) {
                build_direct_selective_2<K, V, work_groups, group_size>(
                        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                        keys_min);
          } else {
                build_direct_selective_2<K, V, work_groups, group_size>(
                        item_ct1.get_local_id(0), keys, res, selection_flags, ht, ht_len,
                        keys_min, num_items);
          }
        }

        template <typename K, typename V, int work_groups, int group_size>
        inline void build_selective_2(
                K (&keys)[group_size],
                V (&res)[group_size],
                int (&selection_flags)[group_size],
                K *ht,
                int ht_len,
                int num_items,
                sycl::nd_item<1> item_ct1
        ) 
        {
          build_selective_2<K, V, work_groups, group_size>(
                keys, res, selection_flags, ht, ht_len, 0, num_items, item_ct1);
        }        
} // namespace crystal 

#endif //ONEAPI_CRYSTAL_JOIN_DPP_HPP