PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py \
  +experiments_longbench_multifieldqa=evaluate_llama_compress_zeroshot_qa_multifield \
  models.target_token=1783 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True 
  # trainers.evaluation_config.max_eval_samples=3
