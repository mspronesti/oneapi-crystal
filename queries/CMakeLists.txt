cmake_minimum_required(VERSION 2.8 )

#############################
# use this macro to add examples
# like this:
# add_example(<cpp_filename_without_extension>)
##############################
macro(add_query query)
    add_executable(${query} ${query}.dp.cpp)
    # just compile an application using 
    # oneapi_crystal like this
    target_link_libraries(${query} oneapi_crystal)
endmacro()

add_query(q11)
add_query(q12)