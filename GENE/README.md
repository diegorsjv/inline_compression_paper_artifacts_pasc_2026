# Directory structure
- **accuracy-evaluation**: stores the raw output data for the GENE accuracy evaluation plot in the paper (Figure 7). We also provide the 
parameters file for the case we use to evaluate accuracy both on the uncompressed baseline run and the compressed executions. There is one directory per configuration
including the baseline uncompressed execution. Inside each subdirectory we store the .out file created after execution. We also provide
the output parameters.dat which summarize the input problem and configurations for the run. The data used in the plots
is taken from the nrg.dat file. In particular, we plot columns 2(u\_par) and 7 (Q\_es) for each timestep.    
- **performance-evaluation**: stores the raw output data for the GENE performance evaluation experiments in the paper (Figure 8). To time the differen code sections 
in GENE the HT tool was used during execution (USE\_PERFLIB=HT). Uncompressed baseline runs with MPI and NCCL contain the parameters file used and the output 
of the HT tool inside each ht\_out directory. All experiments where done using 8 GPU nodes (4 GPUs per node) in Raven for a total of 32 processes. One perfout.txt per
process is stored inside the ht\_out directories. 
    - For compressed runs each directory is named based on the ZFP fixed-rate value used. Inside each directory is the parameters file used for GENE and the perfout.txt files
    with timing results for each process.   
- **src**: the GENE fusion code is freely accessible through an application in the (project webpage)[https://genecode.org/] and accepting the user agreement. As our baseline for GENE experiments in this paper,
we use GENE release 3.2 - alpha 0 (commit hash: 079d681).
    - compression_code_changes.txt:  Our compressed implementation is still not part of the main GENE source code and is thus not openly available. This text files shows all the relevant code changes, relative to the 
    baseline (GENE release 3.2 - alpha 0 commit hash: 079d681) necessary for our overlapped ZFP compression and NCCL communication scheme described in the paper. 
    - zfp_compression_multi_stream.h: Additionally, we include the wrapper file referenced in the gpu_exchange.cxx file. This creates a compressor object around ZFP's high level API to hide compression details from 
    application code. This should provide interested developers sufficient input to replicate our work. 

