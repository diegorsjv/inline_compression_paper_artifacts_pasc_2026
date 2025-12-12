#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring> 
#include <sstream> 
#include "mpi.h"
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include <fstream>

void parse_command_line(int argc, char* argv[], int& nx0, int& nky0 , int& nz0, int& nv0, int& nw0, \
                       int& n_spec){

    if(argc<8){
        std::cout << "Usage: ./app -nx0 <v> -nky0 <v> -nz0 <v> -nv0 <v> -nw0 <v> -n_spec <v> -cr <v>\n" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
        }
    }
}

void initializeArrayHost6D(double* array, int nx0, int nky0, int nz0, int nv0, int nw0, int n_spec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int l = 0; l < n_spec; ++l) {
        for (int k = 0; k < nw0; ++k) {
            for (int j = 0; j < nv0; ++j) {
                for (int i = 0; i < nz0; ++i) {
                    for (int h = 0; h < nky0; ++h) {
                        for (int g = 0; g < nx0; ++g) {
                            // Calculate the linear index using Fortran ordering
                            int index = g + nx0 * (h + nky0 * (i + nz0 * (j + nv0 * (k + nw0 * l))));

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
    MPI_Init(&argc, &argv);
    int nx0, nky0, nz0, nv0, nw0, n_spec; 
    int* n_dimensions = new int(3);
    
    parse_command_line(argc, argv, nx0, nky0, nz0, nv0, nw0, \
                       n_spec);

    std::cout << "Dimensions: " << nx0 << "," << nky0 << "," << nz0 << "," << nv0 << "," << nw0 << "," << n_spec << std::endl;
    n_dimensions[0] = nx0; 
    n_dimensions[1] = nky0; 
    n_dimensions[2] = nz0;
    int totalSize = nx0 * nky0 * nz0 * nv0 * nw0 * n_spec;

    std::cout << "Size in bytes = " << totalSize*sizeof(double) << std::endl;
    int my_rank, size;
    int tag = 1;
    MPI_Status status;

    int device_id;
    int num_devices;    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    std::string processor_name_str(processor_name, name_len);
    int couple = (my_rank+size/2)%size;
    int is_first_half_rank = (my_rank < size/2) ? 0 : MPI_UNDEFINED;
    MPI_Comm first_half_ranks;
    MPI_Comm_split(MPI_COMM_WORLD, is_first_half_rank, my_rank, &first_half_ranks);


    double* h_array = new double[totalSize];
    MPI_Request reqs_send = MPI_REQUEST_NULL;
    MPI_Request reqs_recv = MPI_REQUEST_NULL;

    if(is_first_half_rank == 0){
        std::cout << "Initializing data on host....." << std::endl;
        initializeArrayHost6D(h_array, nx0, nky0, nz0, nv0, nw0, n_spec);
        std::cout << "Initialization complete" << std::endl;
        //writeArrayToFile(h_array, nx0, nky0, nz0, nv0, nw0, n_spec, "initial_array.txt");
    }

    // Rank 0 sends initialized host data to the first half of ranks
    //MPI_Bcast(h_array, totalSize, MPI_DOUBLE, 0, first_half_ranks);

    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(my_rank%num_devices);
    cudaGetDevice(&device_id);
    std::cout << "Rank "<< my_rank <<"Device ID: " << device_id << " on node: " << processor_name_str << std::endl;

    double* d_array_original;
    cudaMalloc((void**)&d_array_original, totalSize * sizeof(double));
    
    if(is_first_half_rank == 0){
        //std::ostringstream filename;
        //filename << "no_compression_results/no_compression_initial_array_" << my_rank << ".txt";
        //writeArrayToFileLinearized(h_array, totalSize, filename.str());
        cudaMemcpy(d_array_original, h_array, totalSize * sizeof(double), cudaMemcpyHostToDevice);

    }

    MPI_Barrier(MPI_COMM_WORLD);
   // Timing variables
    double total_time = 0.0;
    double compression_time = 0.0;
    double decompression_time = 0.0;
    int iterations = 200;
    std::chrono::time_point<std::chrono::high_resolution_clock> iteration_start;
    
    for(int i=0; i<iterations; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank==0){
            iteration_start = std::chrono::high_resolution_clock::now();
        }
        if(is_first_half_rank == 0){
            MPI_Send(d_array_original, totalSize, MPI_DOUBLE, couple, tag, MPI_COMM_WORLD);
        }else{
            MPI_Recv(d_array_original, totalSize, MPI_DOUBLE, couple, tag, MPI_COMM_WORLD, &status);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(my_rank==0){
            auto iteration_end = std::chrono::high_resolution_clock::now();
            double iteration_duration = std::chrono::duration<double>(iteration_end - iteration_start).count();
            if(i>=100){
                total_time += iteration_duration;
            }
        }
    }
    //if(is_first_half_rank != 0){
    //    cudaMemcpy(h_array, d_array_original, totalSize * sizeof(double), cudaMemcpyDeviceToHost);
	//    std::ostringstream filename;
    //	filename << "no_compression_results/no_compression_decompressed_array_" << my_rank << ".txt";
    //    writeArrayToFileLinearized(h_array, totalSize, filename.str());
    //}

    
    if(my_rank==0){
        double avg_total_time = total_time / 100;
        std::cout << "\nAverage iteration time: " << avg_total_time << " seconds" << std::endl;
    }

    // Free memory on host and device
    delete[] h_array;
    cudaFree(d_array_original);
    // cudaFree(d_array_decompressed);

    MPI_Finalize();
    return 0;
}
