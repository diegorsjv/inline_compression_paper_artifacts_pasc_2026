/* ZFPCompressor class declaration and definition:
This source file contains the GENE-specific wrapper for ZFP compression.
Through the creation of a ZFPCompressor object, the user can call on different
compression/decompression functions of the ZFP library through a simplified interface,
avoiding bloating of the application specific code.*/

#ifndef ZFP_COMPRESSION
#define ZFP_COMPRESSION

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include "cuda_runtime.h"
#include "mpi.h"
#include "zfp.h"
#include <sys/stat.h>

enum BackendEnum{
    CPU_BACKEND,
    GPU_BACKEND
};

bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}


void writeArrayToFile(double* d_array, int numElements, int counter, int rank, const std::string& buf_type) {
    // Allocate memory on the host
    double* h_array = new double[numElements];

    // Copy data from device (GPU) to host (CPU)
    cudaMemcpy(h_array, d_array, numElements * sizeof(double), cudaMemcpyDeviceToHost);

    // Generate the filename
    std::string filename = "rank_" + std::to_string(rank) + "_" + buf_type + "_iteration_" + std::to_string(counter) + ".txt";

    // Open the file for writing
    std::ofstream outFile(filename);

    // Check if the file is opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        delete[] h_array;
        return;
    }

    // Write each element of the array to the file, one element per line
    for (int i = 0; i < numElements; ++i) {
        outFile << h_array[i] << std::endl;
    }

    // Close the file
    outFile.close();

    // Free host memory
    delete[] h_array;
}


void print_data_as_bin(int file_id, double* data, size_t size){
    double* data_host;
    size_t bytes = size*sizeof(double);
    std::cout << "print_data_as_bin size: " << size << std::endl;
    cudaMallocHost((void**)&data_host, bytes);
    cudaMemcpy( data_host, data, bytes, cudaMemcpyDeviceToHost);
    std::ostringstream filenameStream;
    filenameStream << "rank_0_exchange_v_data_" << file_id << ".bin";
    if (fileExists(filenameStream.str())) {
        std::cout << "File " << filenameStream.str() << " already exists. Skipping write." << std::endl;
        return;
    }
    std::ofstream outputFile(filenameStream.str(), std::ios::binary);
     if (!outputFile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return;
    }
    std::cout << "Bytes written: " << bytes << std::endl;
    outputFile.write(reinterpret_cast<const char*>(data_host), bytes);
    outputFile.close();
    cudaFreeHost(data_host);
}


template <typename T>
class ZFPCompressor{
    private:
        zfp_field* meta_data;
        zfp_stream* zfp;
        double user_mode_value;
        size_t compressed_size;
        BackendEnum backend_flag;
    public:
        ZFPCompressor(T* data_to_compress, const int n_dimensions, const int* dimensions, const double user_value, BackendEnum b_flag);
        ~ZFPCompressor();
        void ZFPCompressorSetDataPointer(T* data_to_compress);
        void ZFPCompressorSetStrides(ptrdiff_t x_s);
        void ZFPCompressorSetStrides(ptrdiff_t x_s, ptrdiff_t y_s);
        void ZFPCompressorSetStrides(ptrdiff_t x_s, ptrdiff_t y_s, ptrdiff_t z_s);
        void ZFPCompressorSetStrides(ptrdiff_t x_s, ptrdiff_t y_s, ptrdiff_t z_s, ptrdiff_t w_s);
        void ZFPCompressorSetCudaStream(cudaStream_t custream);
        void print_data(T* data, size_t size);
        unsigned char* get_compressed_buffer();
        void get_compressed_buffer(unsigned char* compressed_data_buffer);
        size_t get_compressed_size();
        size_t estimate_compressed_buffer_size();
        // TODO: bring back the setstride functions to add strides to the zfp_field.
        void compress_fixed_rate(unsigned char* compressed_buffer, size_t buffer_size);
        T* decompress_fixed_rate(unsigned char* compressed_data, const size_t buffer_size);
};

template <typename T>
inline ZFPCompressor<T> :: ZFPCompressor(T* data_to_compress, const int n_dimensions, const int* dimensions, const double user_value, BackendEnum b_flag):\
                            user_mode_value(user_value), compressed_size(0), backend_flag(b_flag)
{
    switch(n_dimensions){
        case(1):
            meta_data = zfp_field_1d(data_to_compress, zfp_type_double, dimensions[0]);
            zfp = zfp_stream_open(nullptr);
            break;
        case(2):
            meta_data = zfp_field_2d(data_to_compress, zfp_type_double, dimensions[0], dimensions[1]);
            zfp = zfp_stream_open(nullptr);
            break;
        case(3):
            meta_data = zfp_field_3d(data_to_compress, zfp_type_double, dimensions[0], dimensions[1], dimensions[2]);
            zfp = zfp_stream_open(nullptr);
            break;
        case(4):
            meta_data = zfp_field_4d(data_to_compress, zfp_type_double, dimensions[0], dimensions[1], dimensions[2],\
                                     dimensions[3]);
            zfp = zfp_stream_open(nullptr);
            break;
        default:
            std::cout << "["<< __FILE__ <<"]: Invalid array dimensionality" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    //std::cout << "["<<__FILE__<<"] Created ZFPCompressor for " << n_dimensions << "D data" << std::endl;
}

template <typename T>
inline ZFPCompressor<T> :: ~ZFPCompressor(){
    zfp_field_free(meta_data);
}

template <typename T>
inline void ZFPCompressor<T> :: ZFPCompressorSetDataPointer(T* data_to_compress){
    zfp_field_set_pointer(meta_data, data_to_compress);
}

template <typename T>
inline void ZFPCompressor<T> :: ZFPCompressorSetStrides(ptrdiff_t x_s){
    zfp_field_set_stride_1d(meta_data, x_s);
}

template <typename T>
inline void ZFPCompressor<T> :: ZFPCompressorSetStrides(ptrdiff_t x_s, ptrdiff_t y_s){
    zfp_field_set_stride_2d(meta_data, x_s, y_s);
}

template <typename T>
inline void ZFPCompressor<T> :: ZFPCompressorSetStrides(ptrdiff_t x_s, ptrdiff_t y_s, ptrdiff_t z_s){
    zfp_field_set_stride_3d(meta_data, x_s, y_s, z_s);
}

template <typename T>
inline void ZFPCompressor<T> :: ZFPCompressorSetStrides(ptrdiff_t x_s, ptrdiff_t y_s, ptrdiff_t z_s, ptrdiff_t w_s){
    zfp_field_set_stride_4d(meta_data, x_s, y_s, z_s, w_s);
}

template <typename T>
inline void ZFPCompressor<T> :: ZFPCompressorSetCudaStream(cudaStream_t custream){
   zfp_stream_set_cuda_stream(zfp, custream);
}

template <typename T>
inline size_t ZFPCompressor<T> :: get_compressed_size(){
    return compressed_size;
}

template <typename T>
inline size_t ZFPCompressor<T> :: estimate_compressed_buffer_size(){
    size_t buffer_size;
    zfp_stream_set_rate(zfp, user_mode_value, zfp_field_type(meta_data), zfp_field_dimensionality(meta_data), zfp_false);
    buffer_size = zfp_stream_maximum_size(zfp, meta_data);
    return buffer_size;
}

template <typename T>
inline void ZFPCompressor<T> :: print_data(T* data, size_t size){
    printf("Size of array is: %zu\n", size);
    double* data_host;
    cudaMallocHost((void**)&data_host, size*sizeof(double));
    cudaMemcpy( data_host, data, size*sizeof(double), cudaMemcpyDeviceToHost);

    std::ofstream outFile("rank_0_exchange_v_zfp_before.txt", std::ios::trunc);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open the file." << std::endl;
    }
    for (int i = 0; i < size; i+=2) {
        outFile << data_host[i] << '\t' << data_host[i+1] << '\n';
    }
    outFile.close();
}

template <typename T>
inline void ZFPCompressor<T>::compress_fixed_rate(unsigned char* compressed_buffer, size_t buffer_size){
    int num_threads=8;
    assert(buffer_size > 0);

    zfp_stream_set_rate(zfp, user_mode_value, zfp_field_type(meta_data), zfp_field_dimensionality(meta_data), zfp_false);

    //std::cout << "Starting compress_fixed_rate" << std::endl;
    bitstream* stream = stream_open(compressed_buffer, buffer_size);
    //std::cout << "setting bit stream" << std::endl;
    zfp_stream_set_bit_stream(zfp, stream);
    //std::cout << "zfp_rewind in progress" << std::endl;
    zfp_stream_rewind(zfp);
    //zfp_compress returns the resulting byte offset within the bit stream
    //this equals the number of bytes of compressed storage IF the stream
    //was rewound before the call to compress.
    switch(backend_flag){
        case CPU_BACKEND:
            if(zfp_stream_set_execution(zfp,zfp_exec_omp)){
                zfp_stream_set_omp_threads(zfp, num_threads);
                //std::cout << "[COMPRESSOR] Calling ZFP compress" << std::endl;
                compressed_size = zfp_compress(zfp, meta_data);
                assert(compressed_size > 0);
                //std::cout << "[COMPRESSOR] Compressed bytes: "<< compressed_size << std::endl;
            }
            break;
        case GPU_BACKEND:
            //std::cout << "[COMPRESSOR] Setting execution policy: CUDA" << std::endl;
            zfp_stream_set_execution(zfp, zfp_exec_cuda);
            // std::cout << "[COMPRESSOR] Calling ZFP compress" << std::endl;
            compressed_size = zfp_compress(zfp, meta_data);
            assert(compressed_size > 0);
            //std::cout << "[COMPRESSOR] Compressed bytes: "<< compressed_size << std::endl;
            break;
    }
    zfp_stream_close(zfp);
    stream_close(stream);
}

template <typename T>
inline T* ZFPCompressor<T>::decompress_fixed_rate(unsigned char* compressed_data, const size_t buffer_size){
    //std::cout << "[DECOMPRESSOR] Calling ZFP decompress" << std::endl;
    zfp_stream_set_rate(zfp, user_mode_value, zfp_field_type(meta_data), zfp_field_dimensionality(meta_data), zfp_false);

    const size_t local_buffersize = zfp_stream_maximum_size(zfp, meta_data);
    //std::cout << "[DECOMPRESSOR] local_buffersize = " << local_buffersize << std::endl;
    assert(buffer_size == local_buffersize);

    bitstream* stream = stream_open(compressed_data, buffer_size); //This was previous and worked
    //bitstream* stream = stream_open(compressed_data, local_buffersize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    switch(backend_flag){
        case CPU_BACKEND:
            if(zfp_decompress(zfp, meta_data) == 0){
                std::cerr << "Decompression failed" << std::endl;
            }
            break;
        case GPU_BACKEND:
            // std::cout << "[DECOMPRESSOR] Setting execution policy: CUDA" << std::endl;
            zfp_stream_set_execution(zfp, zfp_exec_cuda);
            if(zfp_decompress(zfp, meta_data) == 0){
                std::cerr << "Decompression failed" << std::endl;
            }
            break;
    }
    zfp_stream_close(zfp);
    stream_close(stream);
    return static_cast<T*>(meta_data->data);
}


#endif
