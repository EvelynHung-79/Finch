PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v1=evaluate_musiqueqa \
  models.target_token=5113 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True \
  models.pin_header=True \
  models.per_head_vote=True \
  trainers.evaluation_config.max_eval_samples=20
