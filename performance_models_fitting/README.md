# Directory structure
- **data**: stores the raw output data and postprocessed csv files gathered from the execution of the uncompressed MPI multipair exchange
used to fit the performance model in the paper.  Also the SLURM submission scripts used to carry out the experiments are stored here.
- **src**: contains the source code for the MPI multipair uncompressed exchange benchmark. . 

# Compiling

To compile the uncompressed multipair exchange benchmark we provide a CMakeLists.txt file as build mechanism. 
To build the code follow the next steps:

```
cd src
mkdir build
cd build
cmake .. 
```
# Build environment 

We used the following evironment variables on our test system: Raven

Currently Loaded Modulefiles:
 1) gcc/12   2) cuda/12.1   3) openmpi/4.1   4) openmpi_gpu/4.1
