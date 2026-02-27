PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py \
  +experiments_longbench_qasper=evaluate_llama_compress_zeroshot_qa_qasper \
  models.target_token=768 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True 
  # trainers.evaluation_config.max_eval_samples=3