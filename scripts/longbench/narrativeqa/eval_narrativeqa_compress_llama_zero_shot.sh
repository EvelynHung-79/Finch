PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py +experiments_longbench_narrativeqa=evaluate_llama_compress_zeroshot_qa_narrativeqa \
  models.target_token=512 \
  models.split_size=256 \
  models.condition="question" \
  models.normalize=True

# PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
#   conf/accelerate/single_gpu.yaml \
#   src/context_compression/run.py --multirun +experiments_longbench_narrativeqa=evaluate_llama_compress_zeroshot_qa_narrativeqa \
#   models.target_token=512 \
#   models.split_size=64,256,1024,2048 \
#   models.condition="question" \
#   models.normalize=True