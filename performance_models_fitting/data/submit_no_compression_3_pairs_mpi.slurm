#!/bin/bash -l
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J no_compression_3_pairs_mpi
#
# Standard output and error:
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --constraint="gpu&^gpu-bw"
# --- uncomment to use 2 GPUs on a shared node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=18
#SBATCH --mem=500000
#SBATCH --time=02:30:00



module purge
module load  gcc/12 cuda/12.1 openmpi/4.1 openmpi_gpu/4.1 nccl/2.18.3

for i in {1..10}
do
   echo "no compression 4 pairs - 0.75 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 8 -nz0 8 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 1.5 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 8 -nz0 16 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 3 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 16 -nz0 16 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 4.5 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 16 -nz0 24 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 6.75 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 24 -nz0 24 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 9  MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 24 -nz0 32 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 10.125  MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 144 -nky0 24 -nz0 24 -nv0 2 -nw0 8 -n_spec 1
done


for i in {1..10}
do
   echo "no compression 4 pairs - 12 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 96 -nky0 32 -nz0 32 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs - 13.5 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 24 -nz0 24 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "no compression 4 pairs 18 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 32 -nz0 24 -nv0 2 -nw0 8 -n_spec 1
done

for i in {1..10}
do
   echo "20.25 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 36 -nz0 24 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "23.625 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 36 -nz0 28 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "26.25 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 40 -nz0 28 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "30 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 40 -nz0 32 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "33 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 44 -nz0 32 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "37.125 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 44 -nz0 36 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "41.25 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 44 -nz0 40 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "45 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 48 -nz0 40 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "49.5 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 48 -nz0 44 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "54 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 48 -nz0 48 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "63 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 56 -nz0 48 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "108 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 96 -nz0 48 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "216 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 96 -nz0 96 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "432 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 96 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "468 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 104 -nv0 2 -nw0 8 -n_spec 1 
done

for i in {1..10}
do
   echo "540 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 120 -nv0 2 -nw0 8 -n_spec 1 
done

# 700 MB
for i in {1..10}
do
   echo "700 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 152 -nv0 2 -nw0 8 -n_spec 1
done

# 950 MB
for i in {1..10}
do
   echo "950 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 200 -nv0 2 -nw0 8 -n_spec 1
done

# 1200 MB
for i in {1..10}
do
   echo "1200 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 256 -nv0 2 -nw0 8 -n_spec 1
done

# 1500 MB
for i in {1..10}
do
   echo "1500 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 320 -nv0 2 -nw0 8 -n_spec 1
done

# 1800 MB
for i in {1..10}
do
   echo "1800 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 384 -nv0 2 -nw0 8 -n_spec 1
done

# 2000 MB
for i in {1..10}
do
   echo "2000 MB $i"
   srun -n 6 ./model_no_compression_exec -nx0 192 -nky0 192 -nz0 448 -nv0 2 -nw0 8 -n_spec 1
done

