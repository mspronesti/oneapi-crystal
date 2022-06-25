#include <CL/sycl.hpp>
#include <iostream>
#include <stdio.h>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include <oneapi_crystal/crystal.hpp>

#include "ssb_utils.h"
#include "../oneapi_crystal/tools/queue_helpers.hpp"
#include "../oneapi_crystal/utils/atomic.hpp"
#include <chrono>

#define TILE_SIZE (block_threads * items_per_thread)

using namespace std;
using namespace crystal;

template<int block_threads, int items_per_thread>
void probe(
    int* lo_orderdate, 
    int* lo_custkey, 
    int* lo_suppkey, 
    int* lo_revenue, 
    int lo_len,
    int* ht_s, 
    int s_len,
    int* ht_c, 
    int c_len,
    int* ht_d, 
    int d_len,
    int* res, 
    sycl::nd_item<1> item_ct1
) 
{
  // Load a segment of consecutive items that are blocked across threads
  int items[items_per_thread];
  int selection_flags[items_per_thread];
  int c_nation[items_per_thread];
  int s_nation[items_per_thread];
  int year[items_per_thread];
  int revenue[items_per_thread];

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
  }

  init_flags<block_threads, items_per_thread>(selection_flags);

  load<int, block_threads, items_per_thread>(lo_suppkey + tile_offset, items, num_tile_items, item_ct1);
  probe_2<int, int, block_threads, items_per_thread>(items, s_nation, selection_flags,
      ht_s, s_len, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(lo_custkey + tile_offset, items, num_tile_items, item_ct1);
  probe_2<int, int, block_threads, items_per_thread>(items, c_nation, selection_flags,
      ht_c, c_len, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(lo_orderdate + tile_offset, items, num_tile_items, item_ct1);
  probe_2<int, int, block_threads, items_per_thread>(items, year, selection_flags,
      ht_d, d_len, 19920101, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(lo_revenue + tile_offset, revenue, num_tile_items, item_ct1);

  #pragma unroll
  for (int ITEM = 0; ITEM < items_per_thread; ++ITEM) {
    if ((item_ct1.get_local_id(0) + (block_threads * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (s_nation[ITEM] * 250 * 7  + c_nation[ITEM] * 7 +  (year[ITEM] - 1992)) % ((1998-1992+1) * 250 * 250);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        res[hash * 4 + 2] = s_nation[ITEM];

        atomicAdd(res[hash * 4 + 3], revenue[ITEM]);
      }
    }
  }
}

template<int block_threads, int items_per_thread>
void build_hashtable_s(
    int *filter_col, 
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

  load<int, block_threads, items_per_thread>(filter_col + tile_offset, items, num_tile_items, item_ct1);
  predicate_eq<int, block_threads, items_per_thread>(items, 24, selection_flags, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(dim_key + tile_offset, items, num_tile_items, item_ct1);
  load<int, block_threads, items_per_thread>(dim_val + tile_offset, items2, num_tile_items, item_ct1);
  build_selective_2<int, int, block_threads, items_per_thread>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items, item_ct1);
}

template<int block_threads, int items_per_thread>
void build_hashtable_c(
    int *filter_col, 
    int *dim_key, 
    int* dim_val, 
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

  load<int, block_threads, items_per_thread>(filter_col + tile_offset, items, num_tile_items, item_ct1);
  predicate_eq<int, block_threads, items_per_thread>(items, 24, selection_flags, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(dim_key + tile_offset, items, num_tile_items, item_ct1);
  load<int, block_threads, items_per_thread>(dim_val + tile_offset, items2, num_tile_items, item_ct1);
  build_selective_2<int, int, block_threads, items_per_thread>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items, item_ct1);
}

template<int block_threads, int items_per_thread>
void build_hashtable_d(
    int *dim_key, 
    int *dim_val, 
    int num_tuples, 
    int *hash_table, 
    int num_slots, 
    int val_min,
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

  load<int, block_threads, items_per_thread>(dim_val + tile_offset, items, num_tile_items, item_ct1);
  predicate_gte<int, block_threads, items_per_thread>(items, 1992, selection_flags, num_tile_items, item_ct1);
  predicate_and_lte<int, block_threads, items_per_thread>(items, 1997, selection_flags, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(dim_key + tile_offset, items2, num_tile_items, item_ct1);
  build_selective_2<int, int, block_threads, items_per_thread>(items2, items, selection_flags, 
      hash_table, num_slots, 19920101, num_tile_items, item_ct1);
}

void runQuery(
    sycl::queue &q,
    int *lo_orderdate, 
    int *lo_custkey, 
    int *lo_suppkey,
    int *lo_revenue, 
    int lo_len, 
    int *d_datekey, 
    int *d_year,
    int d_len, 
    int *s_suppkey, 
    int *s_nation, 
    int *s_city, 
    int s_len,
    int *c_custkey,
    int *c_nation, 
    int *c_city, 
    int c_len
) 
{
 try {
    chrono::high_resolution_clock::time_point st, finish;
    st = chrono::high_resolution_clock::now();

    int *ht_d, *ht_c, *ht_s;
    int d_val_len = 19981230 - 19920101 + 1;

    ht_d = (int*)malloc_device(2 * d_val_len * sizeof(int), q);
    ht_c = (int*)malloc_device(2 * c_len * sizeof(int), q);
    ht_s = (int*)malloc_device(2 * s_len * sizeof(int), q);

    q.memset(ht_d, 0, 2 * d_val_len * sizeof(int)).wait();
    q.memset(ht_s, 0, 2 * s_len * sizeof(int)).wait();


    int tile_items = 128*4;
    
    int num_blocks_s = (s_len + tile_items - 1)/tile_items;
    q.submit([&](sycl::handler &h){

        h.parallel_for<class build_s>(sycl::nd_range<1>({static_cast<size_t>(num_blocks_s * 128)},{128}),
            [=](sycl::nd_item<1>  it) {
            build_hashtable_s<128,4>(s_nation, s_suppkey, s_city, s_len, ht_s,
                                    s_len, it);
        });
    });

    int num_blocks_c = (c_len + tile_items - 1)/tile_items;
    q.submit([&](sycl::handler &h){

        h.parallel_for<class build_c>(sycl::nd_range<1>({static_cast<size_t>(num_blocks_c * 128)},{128}),
            [=](sycl::nd_item<1>  it) {
            build_hashtable_c<128,4>(c_nation, c_custkey, c_city, c_len, ht_c,
                                    c_len, it);
        });
    });


    int d_val_min = 19920101;
    int num_blocks_d = (d_len + tile_items - 1)/tile_items;
    q.submit([&](sycl::handler &h){

        h.parallel_for<class build_d>(sycl::nd_range<1>({static_cast<size_t>(num_blocks_d * 128)}, {128}),
            [=](sycl::nd_item<1>  it) {
            build_hashtable_d<128,4>(d_datekey, d_year, d_len, ht_d, d_val_len,
                                    d_val_min, it);
        });
    });

    int *res;
    int res_size = ((1998-1992+1) * 250 * 250);
    int res_array_size = res_size * 4;

    res = (int*)malloc_device(res_array_size * sizeof(int), q);
    q.memset(res, 0, res_array_size * sizeof(int)).wait();

    int num_blocks_lo = (lo_len + tile_items - 1)/tile_items;
    // Run
    q.submit([&](sycl::handler &h){

        h.parallel_for<class Probe>(sycl::nd_range<1>({static_cast<size_t>(num_blocks_lo * 128)},{128}),
            [=](sycl::nd_item<1>  it) {
            probe<128,4>(lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len,
                        ht_s, s_len, ht_c, c_len, ht_d, d_val_len, res, it);
            });

    }).wait();

    int* h_res = new int[res_array_size];
    q.memcpy(h_res, res, res_array_size * sizeof(int)).wait();
    
    finish = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = finish - st;

    cout << "Result:" << endl;
    int res_count = 0;
    for (int i=0; i<res_size; i++) {
        if (h_res[4*i] != 0) {
        cout << h_res[4*i] << " " << h_res[4*i + 1] << " " << h_res[4*i + 2] << " " << h_res[4*i + 3] << endl;
        res_count += 1;
        }
    }

    cout << "Res Count: " << res_count << endl;
    cout << "Time Taken Total: " << diff.count() * 1000 << endl;

    delete[] h_res;
    sycl::free(res, q);
    sycl::free(ht_d, q);
    sycl::free(ht_c, q);
    sycl::free(ht_s, q);

  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
}
/**
 * Main
 */
int main(int argc, char **argv)
{
  auto q = try_get_queue(sycl::default_selector{});

  // device
  auto dev_name = q.get_device().get_info<sycl::info::device::name>();
  std::cout <<"Running on " << dev_name << '\n' ;
  int num_trials          = 3;

  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);

  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_nation = loadColumn<int>("s_nation", S_LEN);
  int *h_s_city = loadColumn<int>("s_city", S_LEN);

  int *h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *h_c_nation = loadColumn<int>("c_nation", C_LEN);
  int *h_c_city = loadColumn<int>("c_city", C_LEN);

  cout << "** LOADED DATA **" << endl;

  int *d_lo_orderdate = load_to_device<int>(h_lo_orderdate, LO_LEN, q);
  int *d_lo_custkey = load_to_device<int>(h_lo_custkey, LO_LEN, q);
  int *d_lo_suppkey = load_to_device<int>(h_lo_suppkey, LO_LEN, q);
  int *d_lo_revenue = load_to_device<int>(h_lo_revenue, LO_LEN, q);

  int *d_d_datekey = load_to_device<int>(h_d_datekey, D_LEN, q);
  int *d_d_year = load_to_device<int>(h_d_year, D_LEN, q);

  int *d_s_suppkey = load_to_device<int>(h_s_suppkey, S_LEN, q);
  int *d_s_nation = load_to_device<int>(h_s_nation, S_LEN, q);
  int *d_s_city = load_to_device<int>(h_s_city, S_LEN, q);

  int *d_c_custkey = load_to_device<int>(h_c_custkey, C_LEN, q);
  int *d_c_nation = load_to_device<int>(h_c_nation, C_LEN, q);
  int *d_c_city = load_to_device<int>(h_c_city, C_LEN, q);

  cout << "** LOADED DATA TO DEVICE: " << dev_name << "**" << endl;

  for (int t = 0; t < num_trials; t++) {
    runQuery(q,
        d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
        d_d_datekey, d_d_year, D_LEN,
        d_s_suppkey, d_s_nation, d_s_city, S_LEN,
        d_c_custkey, d_c_nation, d_c_city, C_LEN);
  }

  return 0;
}
