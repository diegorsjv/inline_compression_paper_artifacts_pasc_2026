cmake_minimum_required(VERSION 3.19)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message("CUDA_HOME is: $ENV{CUDA_HOME}")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_COMPILER nvcc)
set (CMAKE_CXX_STANDARD 17) 
set (CMAKE_CXX_STANDARD_REQUIRED ON) 
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")
set (CMAKE_CUDA_ARCHITECTURES 80)

add_executable(model_no_compression_exec no_compression_comm.cpp)

target_include_directories(model_no_compression_exec PRIVATE $ENV{OPENMPI_HOME}/include $ENV{CUDA_HOME}/include) 
target_link_directories(model_no_compression_exec PRIVATE $ENV{OPENMPI_HOME}/lib $ENV{CUDA_HOME}/lib64)
target_link_libraries(model_no_compression_exec PUBLIC mpi cudart)
