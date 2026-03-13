PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v1=evaluate_samsum \
  models.compression_rate=0.338 \
  models.target_token=3266 \
  models.split_size=256
