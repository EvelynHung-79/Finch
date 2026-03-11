PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py \
  +experiments_longbench_code=evaluate_llama_compress_zeroshot_lcc \
  models.target_token=646 \
  models.split_size=256 \
  models.condition="question" \
  models.normalize=True
