PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py +experiments_longbench_summarization=evaluate_llama_compress_zeroshot_qmsum \
  models.target_token=3365 \
  models.split_size=256 \
  models.condition="question" \
  models.normalize=True
