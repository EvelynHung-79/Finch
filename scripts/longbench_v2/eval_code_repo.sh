PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v2=evaluate_code_repo \
  models.target_token=32609 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True \
  models.pin_header=True \
  models.per_head_vote=True
