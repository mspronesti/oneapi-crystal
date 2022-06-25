#include <CL/sycl.hpp>
#include <iostream>
#include <stdio.h>

#include <oneapi/mkl.hpp>
#include <oneapi_crystal/crystal.hpp>

#include "generator.h"
#include "../oneapi_crystal/utils/atomic.hpp"
#include "../oneapi_crystal/tools/queue_helpers.hpp"

#include <chrono>


#define TILE_SIZE (block_threads * items_per_thread)

#define NUM_BLOCK_THREAD 128
#define NUM_ITEM_PER_THREAD 4

using namespace crystal;
using namespace std;

// struct to trace time elapsed
// during operator execution
struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
};

template <int block_threads, int items_per_thread>
void build_kernel(
    int *dim_key, 
    int *dim_val, 
    int num_tuples, 
    int *hash_table, 
    int num_slots,
    sycl::nd_item<1> item_ct1
) 
{
  int items[items_per_thread];
  int items2[items_per_thread];
  int selection_flags[items_per_thread];

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  init_flags<block_threads, items_per_thread>(selection_flags);
  load<int, block_threads, items_per_thread>(dim_key + tile_offset, items, num_tile_items, item_ct1);
  load<int, block_threads, items_per_thread>(dim_val + tile_offset, items2, num_tile_items, item_ct1);
  build_selective_2<int, int, block_threads, items_per_thread>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items, item_ct1);
}

template<int block_threads, int items_per_thread>
void probe_kernel(
    int *fact_fkey, 
    int *fact_val, 
    int num_tuples, 
    int *hash_table, 
    int num_slots, 
    unsigned long long *res,
    sycl::nd_item<1> item_ct1
) 
{
  // Load a tile striped across threads
  int selection_flags[items_per_thread];
  int keys[items_per_thread];
  int vals[items_per_thread];
  int join_vals[items_per_thread];

  unsigned long long sum = 0;

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (num_tuples+ TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  init_flags<block_threads, items_per_thread>(selection_flags);
  load<int, block_threads, items_per_thread>(fact_fkey + tile_offset, keys, num_tile_items, item_ct1);
  load<int, block_threads, items_per_thread>(fact_val + tile_offset, vals, num_tile_items, item_ct1);

  probe_2<int, int, block_threads, items_per_thread>(keys, join_vals, selection_flags,
      hash_table, num_slots, num_tile_items, item_ct1);

  #pragma unroll
  for (int i = 0; i < items_per_thread; ++i)
  {
    if ((item_ct1.get_local_id(0) + (block_threads * i) < num_tile_items))
      if (selection_flags[i])
        sum += vals[i] * join_vals[i];
  }

  unsigned long long aggregate = 
        sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());

  if (item_ct1.get_local_id(0) == 0) {
      atomicAdd(*res, aggregate);
  }
}


TimeKeeper hash_join(
    sycl::queue &q,
    int *d_dim_key, 
    int *d_dim_val, 
    int *d_fact_fkey,
    int *d_fact_val, 
    int num_dim, 
    int num_fact
) 
{ 
  unsigned long long* res;
  int* hash_table = nullptr; 
  int num_slots = num_dim;
  float time_build, time_probe, time_memset;

  hash_table = (int*)malloc_device(sizeof(int)* 2 * num_dim, q);
  res = (unsigned long long*)malloc_device(sizeof(long long), q);
  
  int tile_items = NUM_BLOCK_THREAD * NUM_ITEM_PER_THREAD;

  chrono::high_resolution_clock::time_point st, mmset, build, finish;
  // begin time measurement
  st = chrono::high_resolution_clock::now();

  q.memset(hash_table, 0, num_slots * sizeof(int) * 2).wait();
  q.memset(res, 0, sizeof(long long)).wait();
  
  mmset = chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    size_t local_range_size = NUM_BLOCK_THREAD;
    size_t num_groups = static_cast<size_t>(num_dim + tile_items - 1) / tile_items;
    size_t global_range_size= local_range_size * num_groups;
    
    cgh.parallel_for<class build>(
        sycl::nd_range<1>(global_range_size, local_range_size),
          [=](sycl::nd_item<1> item_ct1) {
            build_kernel<NUM_BLOCK_THREAD, NUM_ITEM_PER_THREAD>(
              d_dim_key, d_dim_val, num_dim, hash_table, num_slots, item_ct1);
    });
  });
  build = chrono::high_resolution_clock::now();

  q.submit([&](sycl::handler &cgh) {
    size_t local_range_size = NUM_BLOCK_THREAD;
    size_t num_groups = static_cast<size_t>(num_fact + tile_items - 1) / tile_items;
    size_t global_range_size = local_range_size * num_groups;
    
    cgh.parallel_for<class probe>(
      sycl::nd_range<1>(global_range_size, local_range_size),
        [=](sycl::nd_item<1> item_ct1)  {
            probe_kernel<NUM_BLOCK_THREAD, NUM_ITEM_PER_THREAD>(
              d_fact_fkey, d_fact_val, num_fact, hash_table, num_slots, res, item_ct1);
    });
  }).wait();

  finish = chrono::high_resolution_clock::now();
  unsigned long long h_res;

  q.memcpy(&h_res, res, sizeof(long long)).wait();

  std::cout<<"JOIN RESULTS: "<< h_res << std::endl;

  sycl::free(hash_table, q);
  sycl::free(res, q);

  time_memset = std::chrono::duration<double>(mmset - st).count() * 1000. ;
  time_build = std::chrono::duration<double>(build - mmset).count() * 1000. ;
  time_probe = std::chrono::duration<double>(finish - build).count() * 1000. ;

  TimeKeeper t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset};
  return t;
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char **argv) 
{
  auto q = try_get_queue(sycl::default_selector{});

  std::cout<<"Running on "
          << q.get_device().get_info<sycl::info::device::name>()
          <<std::endl;


  int num_fact           = 256 * 1<<20;
  int num_dim            = 16 * 1<<20;
  int num_trials         = 3;

  // Initialize command line
  if(argc > 1) {
      num_dim = atoi(argv[1]);
  }

  int log2 = 0;
  int num_dim_dup = num_dim >> 1;
  while (num_dim_dup) {
    num_dim_dup >>= 1;
    log2 += 1;
  }

  // Allocate problem device arrays
  int *d_dim_key = nullptr;
  int *d_dim_val = nullptr;
  int *d_fact_fkey = nullptr;
  int *d_fact_val = nullptr;

  d_dim_key = (int*) malloc_device(sizeof(int) * num_dim, q);
  d_dim_val = (int*) malloc_device(sizeof(int) * num_dim, q);
  d_fact_fkey = (int*) malloc_device(sizeof(int) * num_fact, q);
  d_fact_val = (int*) malloc_device(sizeof(int) * num_fact, q);


  int *h_dim_key = nullptr;
  int *h_dim_val = nullptr;
  int *h_fact_fkey = nullptr;
  int *h_fact_val = nullptr;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  q.memcpy(d_dim_key, h_dim_key, sizeof(int) * num_dim).wait();
  q.memcpy(d_dim_val, h_dim_val, sizeof(int) * num_dim).wait();
  q.memcpy(d_fact_fkey, h_fact_fkey, sizeof(int) * num_fact).wait();
  q.memcpy(d_fact_val, h_fact_val, sizeof(int) * num_fact).wait();
  
  
  for (int j = 0; j < num_trials; j++) {
    TimeKeeper t = hash_join(q, d_dim_key, d_dim_val, d_fact_fkey, d_fact_val, num_dim, num_fact);
    cout<< "{"
        << "\"num_dim\":" << num_dim 
        << ",\"num_fact\":" << num_fact 
        << ",\"radix\":" << 0
        << ",\"time_partition_build\":" << 0
        << ",\"time_partition_probe\":" << 0
        << ",\"time_partition_total\":" << 0
        << ",\"time_build\":" << t.time_build << " ms"
        << ",\"time_probe\":" << t.time_probe << " ms"
        << ",\"time_extra\":" << t.time_extra << " ms"
        << ",\"time_join_total\":" << t.time_total << " ms"
        << "}" << endl;
  }

 
  sycl::free(d_dim_key, q);
  sycl::free(d_dim_val, q);
  sycl::free(d_fact_fkey, q);
  sycl::free(d_fact_val, q);


  return 0;
}
