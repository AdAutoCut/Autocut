GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
export OMP_NUM_THREADS=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
#export MASTER_ADDR='10.116.144.152'
export MASTER_ADDR='localhost'
# The port for communication
export MASTER_PORT=8516
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} infer.py \
        --config-file ./configs/exp_outer.yaml
