#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <oneapi_crystal/crystal.hpp>

#include "ssb_utils.h"
#include "../oneapi_crystal/tools/queue_helpers.hpp"
#include "../oneapi_crystal/tools/duration_logger.hpp"

#define TILE_SIZE block_threads * items_per_thread

using namespace crystal;
using namespace std;

// query kernel
template<int block_threads, int items_per_thread>
void device_select_if(
    int* lo_orderdate, 
    int* lo_discount, 
    int* lo_quantity, 
    int* lo_extendedprice,
    int lo_num_entries, 
    unsigned long long* revenue, 
    sycl::nd_item<1> item_ct1
) 
{
  // Load a segment of consecutive items that are blocked across threads
  int items[items_per_thread];
  int selection_flags[items_per_thread];
  int items2[items_per_thread];

  long long sum = 0;

  int tile_offset = item_ct1.get_group(0) * TILE_SIZE;
  int num_tiles = (lo_num_entries + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (item_ct1.get_group(0) == num_tiles - 1) {
    num_tile_items = lo_num_entries - tile_offset;
  }

  load<int, block_threads, items_per_thread>(lo_orderdate + tile_offset, items, num_tile_items, item_ct1);
  predicate_gte<int, block_threads, items_per_thread>(items, 19940101, selection_flags, num_tile_items, item_ct1);
  predicate_and_lte<int, block_threads, items_per_thread>(items, 19940131, selection_flags, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(lo_quantity + tile_offset, items, num_tile_items, item_ct1);
  predicate_and_gte<int, block_threads, items_per_thread>(items, 26, selection_flags, num_tile_items, item_ct1);
  predicate_and_lte<int, block_threads, items_per_thread>(items, 35, selection_flags, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(lo_discount + tile_offset, items, num_tile_items, item_ct1);
  predicate_and_gte<int, block_threads, items_per_thread>(items, 4, selection_flags, num_tile_items, item_ct1);
  predicate_and_lte<int, block_threads, items_per_thread>(items, 6, selection_flags, num_tile_items, item_ct1);

  load<int, block_threads, items_per_thread>(lo_extendedprice + tile_offset, items2, num_tile_items, item_ct1);

  #pragma unroll
  for (int item = 0; item < items_per_thread; ++item)
  {
    if (item_ct1.get_local_id(0) + (block_threads * item) < num_tile_items)
      if (selection_flags[item])
        sum += items[item] * items2[item];
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  unsigned long long aggregate = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());

  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (item_ct1.get_local_id(0) == 0) {
    sycl::atomic<unsigned long long>(
        sycl::global_ptr<unsigned long long>(revenue))
        .fetch_add(aggregate);
  }
}

void run_query(
    sycl::queue &q,
    int *lo_orderdate, 
    int *lo_discount, 
    int *lo_quantity,
    int *lo_extendedprice, 
    int lo_num_entries
)
{
    try {
        unsigned long long revenue;
        unsigned long long* d_sum = nullptr;

        // measuring time taken
        {
            DurationLogger dl{"Query 12"}; // probe time

            d_sum = (unsigned long long*)malloc_device(sizeof(unsigned long long), q);
            
            q.memset(d_sum, 0, sizeof(long long)).wait();

                // Run ----------------------
                int tile_items = 128 * 4; // replace with a define!
                int num_blocks = (lo_num_entries + tile_items - 1)/tile_items;

                q.submit([&](sycl::handler &h){

                    h.parallel_for<class query_kernel>(sycl::nd_range<1>({static_cast<size_t>(num_blocks*128)},{128}),
                        [=](auto& it) {
                        device_select_if<128,4>(lo_orderdate,
                            lo_discount, lo_quantity, lo_extendedprice, lo_num_entries, d_sum, it);
                        });

                }).wait();
                // --------------------------

                        
            q.memcpy(&revenue, d_sum, sizeof(long long)).wait();
        } // probe time again when dl is destroyed

        std::cout << "Revenue: " << revenue << std::endl;

        sycl::free(d_sum, q);
    }
    catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    std::exit(1);
    }
}



int main(int argc, char** argv)
{
  dpct::device_ext &dev = dpct::get_current_device();
  auto q = try_get_queue_with_dev(dev);

  // device
  auto dev_name = q.get_device().get_info<sycl::info::device::name>();
  std::cout <<"Running on " << dev_name << '\n' ;
  
  // number of running trials
  int num_trials          = 3;

  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
  int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
  int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);

  cout << "** LOADED DATA **" << endl;

  int *d_lo_orderdate = load_to_device<int>(h_lo_orderdate, LO_LEN, q);
  int *d_lo_discount = load_to_device<int>(h_lo_discount, LO_LEN, q);
  int *d_lo_quantity = load_to_device<int>(h_lo_quantity, LO_LEN, q);
  int *d_lo_extendedprice = load_to_device<int>(h_lo_extendedprice, LO_LEN, q);
  int *d_d_datekey = load_to_device<int>(h_d_datekey, D_LEN, q);
  int *d_d_year = load_to_device<int>(h_d_year, D_LEN, q);

  cout << "** LOADED DATA TO DEVICE: " << dev_name << " **"<< endl;

  for (int t = 0; t < num_trials; t++) {
    run_query(q, d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice, LO_LEN);
  }

  return 0;
}
