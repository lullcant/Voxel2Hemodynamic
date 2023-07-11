CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch\
 --nproc_per_node=1 --master_port 29412 predict.py\
  --log_dir 20230310_test_20000_500\
  --num_point 20000 --block_size 500\
  --split test --batch_size 1 --features 5


