PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v1=evaluate_trec \
  models.compression_rate=0.3481 \
  models.target_token=2608 \
  models.split_size=256 \
  models.condition="question" \
  models.normalize=True
