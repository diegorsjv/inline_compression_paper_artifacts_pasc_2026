# Directory structure
- **accuracy-evaluation**: we provide the post-processed output for the different configurations of compression and baseline run. The input bsl6DConfig.json file
    used for all experiments of accuracy evaluation is provided.
- **performance-evaluation**: BSL6D relies on the SimpleKernelTimer from kokkos-tools to measure timings of different relevant code regions. In this directory we store
    the timing outputs (.dat files) for the different tagged regions in BSL6D, including the target 'HaloMPIExchangeDistributionFunction' region. In each directory one output file is reported
    per process (8 processes total in our target performance scenario). The kp_reader tool from the same tool-set can be used to read in the different provided .dat files. 

# Access to the compress halo exchange implemention in BSL6D

Our ZFP-based compressed halo exchange discussed in the paper can be found under the BSL6D project.
In particular [branch 119-data-compressed-halo-exchange](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d/-/tree/0297099652e69b6336e02732125bfcc58b482ef9/) includes 
the full implementation. 
