cmake_minimum_required(VERSION 3.10)

#############################################
# use this macro to add queries
# like this:
# add_query(<cpp_filename_without_extension>)
#############################################
macro(add_query query)
    add_executable(${query} ${query}.dp.cpp)
    # just compile an application using 
    # oneapi_crystal like this
    target_link_libraries(${query} oneapi_crystal)
endmacro()

add_query(q11)
add_query(q12)
add_query(q21)
add_query(q22)
add_query(q31)
add_query(q32)
add_query(q33)
add_query(q34)
add_query(q41)
add_query(q42)
add_query(q43)
