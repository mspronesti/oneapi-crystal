cmake_minimum_required(VERSION 3.10)

#################################################
# use this macro to add operators
# like this:
# add_operator(<cpp_filename_without_extension>)
#################################################
macro(add_operator op)
    add_executable(${op} ${op}.cpp)
    # just compile an application using 
    # oneapi_crystal like this
    target_link_libraries(${op} oneapi_crystal)
endmacro()

add_operator(join)
add_operator(project)