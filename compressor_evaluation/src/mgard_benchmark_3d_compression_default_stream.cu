#include <iostream>
#include <cstring> 
#include <sstream> 
#include <random>
#include <chrono>
#include <fstream>
#include "error_metrics.h"
#include "mgard/compress_x.hpp"



#define CUDA_CALL(call)                                                         \
{                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
}

void parse_command_line(int argc, char* argv[], size_t& nx0, size_t& nky0 , size_t& nz0, size_t& nv0, size_t& nw0, \
                       size_t& n_spec, double& error_bound, int& num_streams){

    if(argc<8){
        std::cout << "Usage: ./app -nx0 <v> -nky0 <v> -nz0 <v> -nv0 <v> -nw0 <v> -n_spec <v> -cr <v>\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (argc > 1) {
        // Iterate over the command line arguments
        for (int i = 1; i < argc; i++) {
            // Check for "-fr" flag
            if (strcmp(argv[i], "-nx0") == 0) {
                nx0 = std::stoull(argv[i+1]);
            }
            else if (strcmp(argv[i], "-nky0") == 0) {
                nky0 = std::stoull(argv[i+1]);
            }
            else if(strcmp(argv[i], "-nz0") == 0){
                nz0 = std::stoull(argv[i+1]);
            }
            else if(strcmp(argv[i], "-nv0") == 0){
                nv0 = std::stoull(argv[i+1]);
            }
            else if(strcmp(argv[i], "-nw0") == 0){
                nw0 = std::stoull(argv[i+1]);
            }
            else if(strcmp(argv[i], "-n_spec") == 0){
                n_spec = std::stoull(argv[i+1]);
            }
            else if(strcmp(argv[i], "-eb") == 0){
                error_bound = std::stod(argv[i+1]);
            }
            else if(strcmp(argv[i], "-ns") == 0){
                num_streams = std::atoi(argv[i+1]);
            }
        }
    }
}

void initializeArrayHost6D(double* array, size_t nx0, size_t nky0, size_t nz0, size_t nv0, size_t nw0, size_t n_spec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (size_t l = 0; l < n_spec; ++l) {
        for (size_t k = 0; k < nw0; ++k) {
            for (size_t j = 0; j < nv0; ++j) {
                for (size_t i = 0; i < nz0; ++i) {
                    for (size_t h = 0; h < nky0; ++h) {
                        for (size_t g = 0; g < nx0; ++g) {
                            // Calculate the linear index using Fortran ordering
                            int index = (g + nx0 * (h + nky0 * (i + nz0 * (j + nv0 * (k + nw0 * l)))));
                            // Initialize the element
                            array[index] = dis(gen);
                        }
                    }
                }
            }
        }
    }
}

void writeArrayToFileLinearized(double* array, size_t array_size, const std::string& filename) {
    std::ofstream outputFile(filename); // Open file with specified filename
    if(outputFile.is_open()){
        for(size_t i=0; i<array_size; i++){
            outputFile << array[i] << std::endl;
        }
        outputFile.close();
    }

}


int main(int argc, char *argv[]){
    size_t nx0, nky0, nz0, nv0, nw0, n_spec; 
    double errorBound;
    size_t cmpSize = 0;
    
    int numStreams;
    parse_command_line(argc, argv, nx0, nky0 , nz0, nv0, nw0, n_spec, errorBound, numStreams);
    std::cout << "Dimensions: " << nx0 << "," << nky0 << "," << nz0 << "," << nv0 << "," << nw0 << "," << n_spec << ", eb: " << errorBound << ", num_streams: " <<numStreams << std::endl;

    size_t totalSize = nx0 * nky0 * nz0 * nv0 * nw0 * n_spec;
    mgard_x::SIZE n1 = nx0; 
    mgard_x::SIZE n2 = nky0; 
    mgard_x::SIZE n3  = nz0*nv0*nw0*n_spec; 
    std::vector<mgard_x::SIZE> shape{n1, n2, n3};
    
    int device_id;
    int num_devices;    
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(0);
    cudaGetDevice(&device_id);

    double* h_array_p_original = new double[totalSize];
    double* h_array_p_decompressed = new double[totalSize];
    
    std::cout << "Initializing data on host....." << std::endl;
    initializeArrayHost6D(h_array_p_original, nx0, nky0, nz0, nv0, nw0, n_spec);
    std::cout << "Initialization complete" << std::endl;
    std::cout << "Total amount of bytes in initial array: " << totalSize*sizeof(double) << std::endl;
    
    double* d_array_original_p_original;
    double* d_array_original_p_decompressed;
    cudaMalloc((void**)&d_array_original_p_original, totalSize * sizeof(double));
    cudaMalloc((void**)&d_array_original_p_decompressed, totalSize * sizeof(double));
    void* decompressed_ptr = static_cast<void*>(d_array_original_p_decompressed);

    //std::ostringstream filename;
    //std::ostringstream filename2;
    //filename << "mgard_benchmark_3d_default_stream/p_original_data.txt";
    //writeArrayToFileLinearized(h_array_p_original, totalSize, filename.str());
    
    cudaMemcpy(d_array_original_p_original, h_array_p_original, totalSize * sizeof(double), cudaMemcpyHostToDevice);

    mgard_x::Config config;
    config.lossless = mgard_x::lossless_type::Huffman;
    config.dev_type = mgard_x::device_type::CUDA;

    void *s_compressed_data;
    cudaMalloc((void**)&s_compressed_data, totalSize * sizeof(double)+1e6);
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // Timing variables
    double total_time = 0.0;
    double compression_time = 0.0;
    double decompression_time = 0.0;
    int iterations = 200;
    std::chrono::time_point<std::chrono::high_resolution_clock> iteration_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> compression_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> decompression_start;
    for(int i=0; i<iterations; i++){
        iteration_start = std::chrono::high_resolution_clock::now();
        compression_start = std::chrono::high_resolution_clock::now();
        mgard_x::compress(3, mgard_x::data_type::Double, shape, errorBound, 0,
                  mgard_x::error_bound_type::ABS, d_array_original_p_original,
                  s_compressed_data, cmpSize, config, true);
        auto compression_end = std::chrono::high_resolution_clock::now();
        decompression_start = std::chrono::high_resolution_clock::now();
        mgard_x::decompress(s_compressed_data, cmpSize,decompressed_ptr, config,true);
        auto decompression_end = std::chrono::high_resolution_clock::now();
        auto iteration_end = std::chrono::high_resolution_clock::now();
        double iteration_duration = std::chrono::duration<double>(iteration_end - iteration_start).count();
        double compression_duration = std::chrono::duration<double>(compression_end - compression_start).count();
        double decompression_duration = std::chrono::duration<double>(decompression_end - decompression_start).count();
        if(i>=100){
            total_time += iteration_duration;
            compression_time += compression_duration;
            decompression_time += decompression_duration;
        }
    }
    cudaMemcpy(h_array_p_decompressed, d_array_original_p_decompressed, totalSize * sizeof(double), cudaMemcpyDeviceToHost);
    //std::ostringstream filename3;
    //filename3 << "mgard_benchmark_3d_default_stream/p_decompressed_data.txt";
    //writeArrayToFileLinearized(h_array_p_decompressed, totalSize, filename3.str());

    double avg_total_time = total_time / 100;
    double avg_compression_time = compression_time / 100;
    double avg_decompression_time = decompression_time / 100;
    std::cout << "\nAverage iteration time: " << avg_total_time << " seconds" << std::endl;
    std::cout << "\nAverage compression time: " << avg_compression_time << " seconds" << std::endl;
    std::cout << "\nAverage decompression time: " << avg_decompression_time << " seconds" << std::endl;
  
    std::cout << "Compressed data size is: " << cmpSize << std::endl; 
    std::cout << "Compression ratio: "
          << static_cast<double>( (totalSize * sizeof(double)) / static_cast<double>(cmpSize) )
          << std::endl;
    double mse = compute_mse_gpu(d_array_original_p_original, d_array_original_p_decompressed, totalSize);
    double rel_l2_error = compute_relative_l2_error_gpu(d_array_original_p_original,d_array_original_p_decompressed, totalSize);
    double max_err = compute_max_error_gpu(d_array_original_p_original, d_array_original_p_decompressed, totalSize);

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "L2 Error: " << rel_l2_error << std::endl;
    std::cout << "Max Absolute Error: " << max_err << std::endl;


    // Free memory on host and device
    delete[] h_array_p_original;
    delete[] h_array_p_decompressed;
    cudaFree(d_array_original_p_original);
    cudaFree(d_array_original_p_decompressed);
    cudaFree(s_compressed_data);
    return 0;
}
