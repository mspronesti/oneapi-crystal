set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fsycl")

# include_directories(./)
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
        MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY AND MKL_SYCL_LIBRARY)
    set (MKL_FIND_QUIETLY TRUE)
endif()

set (src
    crystal.hpp
    block_functions/join.hpp
    block_functions/load.hpp
    block_functions/predicate.hpp
    block_functions/store.hpp
)

if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
        MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY AND MKL_SYCL_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()


set(SYCL_LIB "mkl_sycl")
set(INT_LIB "mkl_intel_ilp64")
set(SEQ_LIB "mkl_sequential")
set(THR_LIB "mkl_intel_thread")
set(COR_LIB "mkl_core")

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{MKLROOT}/include)


set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES mkl_sycl mkl_intel_ilp64 mkl_sequential mkl_core sycl pthread)

set (OPENAPI_CRYSTAL_INCLUDE_LIBS ../oneapi_crystal)

if(NOT TARGET oneapi_crystal)
    add_library(oneapi_crystal STATIC ${src})
endif()

set_target_properties(oneapi_crystal PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(oneapi_crystal
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        INTERFACE $<INSTALL_INTERFACE:include>
        PUBLIC ${OPENAPI_CRYSTAL_INCLUDE_LIBS}
        )

target_link_libraries(oneapi_crystal ${MKL_LIBRARIES})

install(DIRECTORY ./
        DESTINATION include/oneapi_crystal
        FILES_MATCHING
        PATTERN "*.hpp"
        )

export(TARGETS oneapi_crystal FILE oneapi_crystal-config.cmake)
set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE oneapi_crystal)
