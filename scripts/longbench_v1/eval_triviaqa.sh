# FINCH (original)
PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/run.py \
  +experiments_longbench_v1=evaluate_triviaqa \
  models.target_token=3844 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True \
  models.pin_header=True \
  models.per_head_vote=True

# FullKV (target_token=200000, repetition_penalty=1.0 in qa_lm_predictor.yaml)
# PYTHONPATH=. venv/bin/python3 -m accelerate.commands.launch --config_file \
#   conf/accelerate/single_gpu.yaml \
#   src/run.py \
#   +experiments_longbench_v1=evaluate_triviaqa \
#   models.target_token=200000 \
#   models.split_size=512 \
#   models.condition="question" \
#   models.normalize=True \
#   models.pin_header=True \
#   models.per_head_vote=True
