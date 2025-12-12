# Directory structure
- **data**: stores the raw output data and postprocessed csv files gathered from the execution of the different compressor evaluation executables. This is the data
		used in the paper plots.  Also the SLURM submission scripts used to carry out the experiments are stored here.
- **src**: contains the source code for the different compressor evaluations we carried out. 

# Compiling

To compile the compressor evaluation executables you will need to have an installation for each of the evaluated compressors:
cuSZp v3, MGARD-X and ZFP with support for multi-stream execution. 

We provide a CMakeLists.txt file as build mechanism. To build the codes follow the next steps:

```
mkdir build
cd build
cmake .. -DZFP_DIR=</path/to/zfp/build> -DMGARD_DIR=</path/to/MGARD/build> -DcuSZp_DIR=</path/to/cuSZp3/build>
```
# Build environment 

We used the following evironment variables on our test system: Raven

Currently Loaded Modulefiles:
 1) gcc/12   2) cuda/12.1   3) openmpi/4.1   4) openmpi_gpu/4.1
