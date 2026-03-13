PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v1=evaluate_triviaqa \
  models.compression_rate=0.325 \
  models.target_token=3844 \
  models.split_size=128
