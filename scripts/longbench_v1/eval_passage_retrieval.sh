PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v1=evaluate_passage_retrieval \
  models.compression_rate=0.3199 \
  models.target_token=4191 \
  models.split_size=256
