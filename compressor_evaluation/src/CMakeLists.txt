cmake_minimum_required(VERSION 3.19)

project(compressor_comparison CUDA CXX)
find_package(zfp REQUIRED)
find_package(mgard REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

message("CUDA_HOME is: $ENV{CUDA_HOME}")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_COMPILER nvcc)
set (CMAKE_CXX_STANDARD 17) 
set (CMAKE_CXX_STANDARD_REQUIRED ON) 
set (CMAKE_CUDA_ARCHITECTURES 80)
file(GLOB COMPRESSORS zfp_compression_multi_stream.h)

add_executable(compressor_comparison_cuszp_benchmark_3d_compression_default_stream_exec cuszp_benchmark_3d_compression_default_stream.cpp error_metrics.cu)
add_executable(compressor_comparison_zfp_benchmark_3d_compression_default_stream_exec ${COMPRESSORS} zfp_benchmark_3d_compression_default_stream.cpp error_metrics.cu)

target_include_directories(compressor_comparison_cuszp_benchmark_3d_compression_default_stream_exec PRIVATE ${CUSZP_INCLUDE_DIR} $ENV{OPENMPI_HOME}/include $ENV{CUDA_HOME}/include $ENV{NCCL_HOME}/include) 
target_link_directories(compressor_comparison_cuszp_benchmark_3d_compression_default_stream_exec PRIVATE ${CUSZP_LIBRARY_DIR} $ENV{OPENMPI_HOME}/lib $ENV{CUDA_HOME}/lib64 $ENV{NCCL_HOME}/lib)
target_link_libraries(compressor_comparison_cuszp_benchmark_3d_compression_default_stream_exec PUBLIC cuSZp mpi cudart nccl dl)

target_compile_definitions(compressor_comparison_zfp_benchmark_3d_compression_default_stream_exec PRIVATE ZFP_WITH_CUDA)
target_include_directories(compressor_comparison_zfp_benchmark_3d_compression_default_stream_exec PRIVATE ${ZFP_INCLUDE_DIR} $ENV{OPENMPI_HOME}/include $ENV{CUDA_HOME}/include) 
target_link_directories(compressor_comparison_zfp_benchmark_3d_compression_default_stream_exec PRIVATE $ENV{OPENMPI_HOME}/lib $ENV{CUDA_HOME}/lib64)
target_link_libraries(compressor_comparison_zfp_benchmark_3d_compression_default_stream_exec PUBLIC zfp::zfp mpi cudart dl)


add_executable(compressor_comparison_mgard_benchmark_3d_compression_default_stream_exec mgard_benchmark_3d_compression_default_stream.cu error_metrics.cu)
target_link_libraries(compressor_comparison_mgard_benchmark_3d_compression_default_stream_exec PUBLIC mgard::mgard)
