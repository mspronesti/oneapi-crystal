#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdio.h>

#include <oneapi/mkl.hpp>
#include <oneapi_crystal/crystal.hpp>

#include "generator.h"
#include "../oneapi_crystal/utils/atomic.hpp"
#include "../oneapi_crystal/tools/queue_helpers.hpp"

#include <chrono>

using namespace std;

#define TILE_SIZE (block_threads * items_per_thread)

#define NUM_BLOCK_THREAD 128
#define NUM_ITEM_PER_THREAD 4

using namespace crystal;
using namespace std;


template<int block_threads, int items_per_thread>
void project(
    float* in1, 
    float* in2, 
    float* out, 
    int num_items,
    sycl::nd_item<1> item_ct1
)
{
  float items[items_per_thread];
  float items2[items_per_thread];
  float res[items_per_thread];

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE; // group_id() + 128*4
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

    if (item_ct1.get_group(0) == num_tiles - 1) {
        num_tile_items = num_items - tile_offset; // 100 -
    }

  load<float, block_threads, items_per_thread>(in1 + tile_offset, items, num_tile_items, item_ct1);
  load<float, block_threads, items_per_thread>(in2 + tile_offset, items2, num_tile_items, item_ct1);

  #pragma unroll
  for (int i = 0; i < items_per_thread; i++) {
        if (item_ct1.get_local_id(0) + (i * block_threads) <
            num_tile_items) {
      res[i] = 2*items[i] + 3*items2[i];
    }
  }

  store<float, block_threads, items_per_thread>(out + tile_offset, res, num_tile_items, item_ct1);
}

template<int block_threads, int items_per_thread>
void project_sigmoid(
    float* in1, 
    float* in2, 
    float* out, 
    int num_items,
    sycl::nd_item<1> item_ct1
)
{
  float items[items_per_thread];
  float items2[items_per_thread];
  float res[items_per_thread];

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = num_items - tile_offset;
  }

  load<float, block_threads, items_per_thread>(in1 + tile_offset, items, num_tile_items, item_ct1);
  load<float, block_threads, items_per_thread>(in2 + tile_offset, items2, num_tile_items, item_ct1);

  #pragma unroll
  for (int i = 0; i < items_per_thread; i++) {
    if (item_ct1.get_local_id(0) + (i * block_threads) < num_tile_items) 
            res[i] =
                1.0f / (1.0f + sycl::exp(-2 * items[i] - 3 * items2[i]));
  }

  store<float, block_threads, items_per_thread>(out + tile_offset, res, num_tile_items, item_ct1);
}


float project_gpu(
    sycl::queue &q,
    float* in1, 
    float* in2, 
    float* out, 
    int num_items
) 
{
  int tile_items = 128*4;
  int num_blocks = (num_items + tile_items - 1)/tile_items;
    
  chrono::high_resolution_clock::time_point st, finish;
  // begin time measurement
  st = chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for(
          sycl::nd_range<1>(static_cast<size_t>(num_blocks*128), 128),
           [=](sycl::nd_item<1> item_ct1) {
               project<128, 4>(in1, in2, out, num_items, item_ct1);
           });
  }).wait();
  finish = chrono::high_resolution_clock::now();

  // time in ms
  return std::chrono::duration<float>(finish - st).count() * 1000.;
}

float project_sigmoid_gpu(
    sycl::queue &q, 
    float* in1, 
    float* in2, 
    float* out, 
    int num_items
)
{
  int tile_items = 128*4;
  int num_blocks = (num_items + tile_items - 1)/tile_items;
  
  chrono::high_resolution_clock::time_point st, finish;
  // begin time measurement
  st = chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>({static_cast<size_t>(num_blocks*128)},{128}),
        [=](sycl::nd_item<1> item_ct1) {
            project_sigmoid<128, 4>(in1, in1, out, num_items,
                                    item_ct1);
        });
  }).wait();
  finish = chrono::high_resolution_clock::now();

  // time in ms
  return std::chrono::duration<float>(finish - st).count() * 1000.;
}

/**
 * Main
 */
int main(int argc, char **argv)
{
  auto q = try_get_queue(sycl::default_selector{}); 
  oneapi::mkl::rng::uniform<float> distr_ct1;
  int num_items = 1 << 28;
  int num_trials          = 3;
  
  float *d_in1 = nullptr;
  d_in1 = (float*) malloc_device(num_items*sizeof(float), q.get_device(), q.get_context());

  float *d_in2 = nullptr;
  d_in2 = (float*) malloc_device(num_items*sizeof(float), q.get_device(), q.get_context());

  float  *d_out = nullptr;
  d_out = (float*) malloc_device(num_items*sizeof(float), q.get_device(), q.get_context());

  float  *d_out_sig = nullptr;
  d_out_sig = (float*) malloc_device(num_items*sizeof(float), q.get_device(), q.get_context());

  sycl::event start, stop;
  std::cout<<"Running on: "
           << q.get_device().get_info<sycl::info::device::name>();
   
    oneapi::mkl::rng::philox4x32x10 *generator;
    int seed = 0;
    generator =
        new oneapi::mkl::rng::philox4x32x10(dpct::get_default_queue(), seed);

    oneapi::mkl::rng::generate(distr_ct1, *generator, num_items, d_in1);
    oneapi::mkl::rng::generate(distr_ct1, *generator, num_items, d_in2);

  float time_proj_gpu;
  float time_proj_sigmoid_gpu;  

  for (int t = 0; t < num_trials; t++) {
    time_proj_gpu = project_gpu(q, d_in1, d_in2, d_out, num_items);
    time_proj_sigmoid_gpu = project_sigmoid_gpu(q,
        d_in1, d_in2
        , d_out_sig,
         num_items);

    std::cout<< "{"
        << "\"time_proj_gpu\":" << time_proj_gpu
        << ",\"time_proj_sigmoid_gpu\":" << time_proj_sigmoid_gpu
        << "}" << endl;
  }

  if (d_in1) sycl::free(d_in1, q); 
  if (d_in2) sycl::free(d_in2, q); 
  if (d_out) sycl::free(d_out, q); 
  if (d_out_sig) sycl::free(d_out_sig, q); 

  return 0;
}