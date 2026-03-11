PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py \
  +experiments_longbench_few_shot=evaluate_llama_compress_zeroshot_samsum \
  models.target_token=2115 \
  models.split_size=256
