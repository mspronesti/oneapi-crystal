#ifndef ONEAPI_CRYSTAL_SYCL_UTILS
#define ONEAPI_CRYSTAL_SYCL_UTILS
#pragma once

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

/**
 * @brief Sycl version of the atomiAdd function 
 *        natively existing in cuda
 *        Performs an atomic addition.

 * @returns the addition result
 */
template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed, 
     MemoryScope, sycl::access::address_space::global_space> ref(val);
  return ref.fetch_add(delta);
}

/**
 * @brief Sycl version of the atomiAdd function 
 *        natively existing in cuda, but in the local space.
 *        Performs an atomic addition.

 * @returns the addition result
 */
template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::work_group>
static inline T atomicAddLocal(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed, 
     MemoryScope, sycl::access::address_space::local_space> ref(val);
  return ref.fetch_add(delta);
}


#endif // ONEAPI_CRYSTAL_SYCL_UTILS