PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py +experiments_longbench_v1=evaluate_multinews \
  models.compression_rate=0.4259 \
  models.target_token=1374 \
  models.split_size=256 \
  models.condition="question" \
  models.normalize=True
