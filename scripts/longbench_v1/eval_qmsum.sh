PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py +experiments_longbench_v1=evaluate_qmsum \
  models.compression_rate=0.317 \
  models.target_token=4629 \
  models.split_size=256 \
  models.condition="question" \
  models.normalize=True
