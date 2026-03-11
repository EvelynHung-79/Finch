PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py \
  +experiments_longbench_musique=evaluate_llama_compress_zeroshot_qa_musique \
  models.target_token=1505 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True
