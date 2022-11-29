cmake_minimum_required(VERSION 3.14...3.99)
project(backend VERSION 1.0.0 LANGUAGES CXX)

#set(NUMCPP_NO_USE_BOOST ON)

add_subdirectory(NumCpp)
add_executable(backend main.cpp)
target_link_libraries(backend PRIVATE NumCpp)
target_include_directories(backend PUBLIC NumCpp)

