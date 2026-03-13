PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py +experiments_longbench_v1=evaluate_hotpotqa \
  models.compression_rate=0.5001 \
  models.target_token=4320 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True
