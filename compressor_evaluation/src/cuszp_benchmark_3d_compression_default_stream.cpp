#include <iostream>
#include <cstring> 
#include <sstream> 
#include <random>
#include <chrono>
#include <fstream>
#include "error_metrics.h"
#include "cuSZp.h"
#include "nvtx3/nvToolsExt.h"


#define CUDA_CALL(call)                                                         \
{                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
}

#define NCCL_CALL(call, processID)                                              \
{                                                                           \
        ncclResult_t err = call;                                                \
        if (err != ncclSuccess) {                                               \
            fprintf(stderr, "NCCL error (Rank %d) %s:%d: '%s'\n", processID,__FILE__, __LINE__,     \
                    ncclGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
}

void parse_command_line(int argc, char* argv[], int& nx0, int& nky0 , int& nz0, int& nv0, int& nw0, \
                       int& n_spec, float& error_bound, int& num_streams){

    if(argc<8){
        std::cout << "Usage: ./app -nx0 <v> -nky0 <v> -nz0 <v> -nv0 <v> -nw0 <v> -n_spec <v> -cr <v>\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (argc > 1) {
        // Iterate over the command line arguments
        for (int i = 1; i < argc; i++) {
            // Check for "-fr" flag
            if (strcmp(argv[i], "-nx0") == 0) {
                nx0 = std::atoi(argv[i+1]);
            }
            else if (strcmp(argv[i], "-nky0") == 0) {
                nky0 = std::atoi(argv[i+1]);
            }
            else if(strcmp(argv[i], "-nz0") == 0){
                nz0 = std::atoi(argv[i+1]);
            }
            else if(strcmp(argv[i], "-nv0") == 0){
                nv0 = std::atoi(argv[i+1]);
            }
            else if(strcmp(argv[i], "-nw0") == 0){
                nw0 = std::atoi(argv[i+1]);
            }
            else if(strcmp(argv[i], "-n_spec") == 0){
                n_spec = std::atoi(argv[i+1]);
            }
            else if(strcmp(argv[i], "-eb") == 0){
                error_bound = std::atof(argv[i+1]);
            }
            else if(strcmp(argv[i], "-ns") == 0){
                num_streams = std::atoi(argv[i+1]);
            }
        }
    }
}

void initializeArrayHost6D(double* array, int nx0, int nky0, int nz0, int nv0, int nw0, int n_spec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_real_distribution<double> dis_img(2.0, 3.0);

    for (int l = 0; l < n_spec; ++l) {
        for (int k = 0; k < nw0; ++k) {
            for (int j = 0; j < nv0; ++j) {
                for (int i = 0; i < nz0; ++i) {
                    for (int h = 0; h < nky0; ++h) {
                        for (int g = 0; g < nx0; ++g) {
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

void writeArrayToFileLinearized(double* array, int array_size, const std::string& filename) {
    std::ofstream outputFile(filename); // Open file with specified filename
    if(outputFile.is_open()){
        for(int i=0; i<array_size; i++){
            outputFile << array[i] << std::endl;
        }
        outputFile.close();
    }

}


int main(int argc, char *argv[]){
    nvtxRangePushA("main");
    int nx0, nky0, nz0, nv0, nw0, n_spec; 
    float errorBound;
    size_t cmpSize = 0;
    int* n_dimensions = new int(3);
    
    int numStreams;
    parse_command_line(argc, argv, nx0, nky0 , nz0, nv0, nw0, n_spec, errorBound, numStreams);
    std::cout << "Dimensions: " << nx0 << "," << nky0 << "," << nz0 << "," << nv0 << "," << nw0 << "," << n_spec << ", eb: " << errorBound << ", num_streams: " <<numStreams << std::endl;

    int totalSize = nx0 * nky0 * nz0 * nv0 * nw0 * n_spec;
    n_dimensions[0] = nx0; 
    n_dimensions[1] = nky0; 
    n_dimensions[2] = nz0*nv0*nw0*n_spec; 
    uint3 dims_cuszp = {n_dimensions[0], n_dimensions[1], n_dimensions[2]};
    
    int device_id;
    int num_devices;    
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(0);
    cudaGetDevice(&device_id);

    double* h_array_p_original = new double[totalSize];
    double* h_array_p_decompressed = new double[totalSize];
    
    std::cout << "Initializing data on host....." << std::endl;
    nvtxRangePushA("initialize_host_arrays");
    initializeArrayHost6D(h_array_p_original, nx0, nky0, nz0, nv0, nw0, n_spec);
    //initializeArrayHost6D(h_array_p_decompressed, nx0, nky0, nz0, nv0, nw0, n_spec);
    nvtxRangePop();
    std::cout << "Initialization complete" << std::endl;
    std::cout << "Total amount of bytes in initial array: " << totalSize*sizeof(double) << std::endl;
    
    double* d_array_original_p_original;
    double* d_array_original_p_decompressed;
    cudaMalloc((void**)&d_array_original_p_original, totalSize * sizeof(double));
    cudaMalloc((void**)&d_array_original_p_decompressed, totalSize * sizeof(double));

    //std::ostringstream filename;
    //std::ostringstream filename2;
    //filename << "cuszp_benchmark_3d_default_stream/p_original_data.txt";
    //writeArrayToFileLinearized(h_array_p_original, totalSize, filename.str());
    
    nvtxRangePushA("transfer_array_h_to_d");
    cudaMemcpy(d_array_original_p_original, h_array_p_original, totalSize * sizeof(double), cudaMemcpyHostToDevice);
    nvtxRangePop(); 

    unsigned char* s_compressed_data;
    cudaMalloc((void**)&s_compressed_data, totalSize * sizeof(double));
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
    nvtxRangePushA("t_loop");
    for(int i=0; i<iterations; i++){
        iteration_start = std::chrono::high_resolution_clock::now();
        nvtxRangePushA("iteration");
        compression_start = std::chrono::high_resolution_clock::now();
        cuSZp_compress_3D_plain_f64(d_array_original_p_original, s_compressed_data, totalSize, &cmpSize, dims_cuszp, errorBound, 0);
        auto compression_end = std::chrono::high_resolution_clock::now();
        decompression_start = std::chrono::high_resolution_clock::now();
        cuSZp_decompress_3D_plain_f64(d_array_original_p_decompressed, s_compressed_data, totalSize, cmpSize, dims_cuszp, errorBound, 0);
        auto decompression_end = std::chrono::high_resolution_clock::now();
        nvtxRangePop();
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
    //nvtxRangePop();
    //nvtxRangePushA("transferArrays_d_to_h");
    //cudaMemcpy(h_array_p_decompressed, d_array_original_p_decompressed, totalSize * sizeof(double), cudaMemcpyDeviceToHost);
    //nvtxRangePop();
    //std::ostringstream filename3;
    //filename3 << "cuszp_benchmark_3d_default_stream/p_decompressed_data.txt";
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
    nvtxRangePop();
    return 0;
}
