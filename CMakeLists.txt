cmake_minimum_required (VERSION 3.4)
project (oneAPI-crystal)

if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set using Release with Debug Info")
    set ( CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
          STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
          FORCE
        )
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory (oneapi_crystal)

# Set this to 'ON' to compile the queries
option(BUILD_QUERIES "Build the queries" OFF)
if(BUILD_QUERIES)
    add_subdirectory(queries)
endif()
unset (BUILD_QUERIES CACHE)

# Set this to 'ON' to compile the operators
option(BUILD_OPERATORS "Build the operators" OFF)
if(BUILD_OPERATORS)
    add_subdirectory(operators)
endif()
unset (BUILD_OPERATORS CACHE)