#ifndef ONEAPI_CRYSTAL_PREDICATE_HPP
#define ONEAPI_CRYSTAL_PREDICATE_HPP
#pragma once

#include <CL/sycl.hpp>
          
namespace crystal {

    template < int block_threads, int items_per_thread>
    inline void init_flags(int (&selection_flags)[items_per_thread]) 
    {
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            selection_flags[i] = 1;
        }
    }


    template < 
        typename T, 
        typename SelectOp, 
        int block_threads, 
        int items_per_thread
        >
    inline void predicate_direct(
            int tid,
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread]
    ) 
    {
        
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            selection_flags[i] = select_op(items[i]);
        }
    }

    template <
        typename T,
        typename SelectOp,
        unsigned int block_threads,
        unsigned int items_per_thread
        >
    inline void predicate_direct (
        int tid, 
        T (&items)[items_per_thread], 
        SelectOp select_op,
        int (&selection_flags)[items_per_thread], 
        int num_items
    ) 
    {
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            if (tid + (i * block_threads) < num_items) {
                selection_flags[i] = select_op(items[i]);
            }
        }
    }

    template<
            typename T,
            typename SelectOp,
            int block_threads,
            int items_per_thread
            >
    inline void predicate (
        T (&items)[items_per_thread], 
        SelectOp select_op,
        int (&selection_flags)[items_per_thread],
        int num_items, 
        sycl::nd_item<1> item_ct1
    ) 
    {
        if ((block_threads * items_per_thread) == num_items) {
            predicate_direct<T, SelectOp, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), items, select_op, selection_flags);
        } else {
            predicate_direct<T, SelectOp, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), items, select_op, selection_flags, num_items);
        }
    }

    template<
            typename T,
            typename SelectOp,
            int block_threads,
            int items_per_thread
            >
    inline void predicate_and_direct (
            int tid,
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread]
    ) 
    {
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            selection_flags[i] = selection_flags[i] && select_op(items[i]);
        }
    }

    template<
            typename T,
            typename SelectOp,
            unsigned int block_threads,
            unsigned int items_per_thread
            >
    inline void predicate_and_direct (
            int tid,
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread],
            int num_items
    ) 
    {
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            if (tid + (i * block_threads) < num_items) {
                selection_flags[i] = selection_flags[i] && select_op(items[i]);
            }
        }
    }

    template<
            typename T,
            typename SelectOp,
            int block_threads,
            int items_per_thread
            >
    inline void predicate_and (
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    ) 
    {
        if ((block_threads * items_per_thread) == num_items) {
            predicate_and_direct <T, SelectOp, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), items, select_op, selection_flags);
        } else {
            predicate_and_direct <T, SelectOp, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), items, select_op, selection_flags, num_items);
        }
    }

    template<
            typename T,
            typename SelectOp,
            int block_threads,
            int items_per_thread
            >
    inline void predicate_or_direct (
            int tid,
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread]
    ) 
    {
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            selection_flags[i] = selection_flags[i] || select_op(items[i]);
        }
    }

    template<
            typename T,
            typename SelectOp,
            int block_threads,
            int items_per_thread
            >
    inline void predicate_or_direct(
            int tid,
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread],
            int num_items 
    ) 
    {
        #pragma unroll
        for (int i = 0; i < items_per_thread; i++) {
            if (tid + (i * block_threads) < num_items) {
                selection_flags[i] = selection_flags[i] || select_op(items[i]);
            }
        }
    }

    template<
            typename T,
            typename SelectOp,
            int block_threads,
            int items_per_thread
            >
    inline void predicate_or (
            T (&items)[items_per_thread],
            SelectOp select_op,
            int (&selection_flags)[items_per_thread],
            int num_items, 
            sycl::nd_item<1> item_ct1 
    ) 
    {

        if ((block_threads * items_per_thread) == num_items) {
            predicate_or_direct<T, SelectOp, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), items, select_op, selection_flags);
        } else {
            predicate_or_direct<T, SelectOp, block_threads, items_per_thread>(
                    item_ct1.get_local_id(0), items, select_op, selection_flags, num_items);
        }
    }


    template<typename T>
    struct LessThan {
        T compare;

        inline LessThan(T compare) : compare(compare) {}

        inline bool operator()(const T &a) const {
            return (a < compare);
        }
    };

    template<typename T>
    struct GreaterThan {
        T compare;

        inline GreaterThan(T compare) : compare(compare) {}

        inline bool operator()(const T &a) const {
            return (a > compare);
        }
    };

    template<typename T>
    struct LessThanEq {
        T compare;

        inline LessThanEq(T compare) : compare(compare) {}

        inline bool operator()(const T &a) const {
            return (a <= compare);
        }
    };

    template<typename T>
    struct GreaterThanEq {
        T compare;

        inline GreaterThanEq(T compare) : compare(compare) {}

        inline bool operator()(const T &a) const {
            return (a >= compare);
        }
    };

    template<typename T>
    struct Eq {
        T compare;

        inline Eq(T compare) : compare(compare) {}

        inline bool operator()(const T &a) const {
            return (a == compare);
        }
    };

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_lt(
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items, sycl::nd_item<1> item_ct1 ){
        LessThan<T> select_op(compare);
        predicate<T, LessThan<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_and_lt (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items, sycl::nd_item<1> item_ct1) {
        LessThan<T> select_op(compare);
        predicate_and<T, LessThan<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_gt (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1) {
        GreaterThan<T> select_op(compare);
        predicate<T, GreaterThan<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_and_gt (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1) {
        GreaterThan<T> select_op(compare);
        predicate_and<T, GreaterThan<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_lte (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    )
    {
        LessThanEq<T> select_op(compare);
        predicate<T, LessThanEq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_and_lte (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    )
    {
        LessThanEq<T> select_op(compare);
        predicate_and<T, LessThanEq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_gte(
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items, 
            sycl::nd_item<1> item_ct1
    ) 
    {
        GreaterThanEq<T> select_op(compare);
        predicate<T, GreaterThanEq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }


    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_and_gte (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items, 
            sycl::nd_item<1> item_ct1
    ) 
    {
        GreaterThanEq<T> select_op(compare);
        
        predicate_and<T, GreaterThanEq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_eq (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items, 
            sycl::nd_item<1> item_ct1
    ) 
    {
        Eq<T> select_op(compare);
        predicate<T, Eq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }


    template < typename T, int block_threads, int items_per_thread>
    inline void predicate_and_eq (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1
    ) 
    {
        Eq<T> select_op(compare);
        
        predicate_and<T, Eq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }

    template < 
        typename T, 
        int block_threads, 
        int items_per_thread
        >
    inline void predicate_or_eq (
            T (&items)[items_per_thread],
            T compare,
            int (&selection_flags)[items_per_thread],
            int num_items,
            sycl::nd_item<1> item_ct1 
    ) 
    {
        Eq<T> select_op(compare);

        predicate_or<T, Eq<T>, block_threads, items_per_thread>(
                items, select_op, selection_flags, num_items, item_ct1);
    }


} // namespace crystal

#endif //ONEAPI_CRYSTAL_PREDICATE_HPP
