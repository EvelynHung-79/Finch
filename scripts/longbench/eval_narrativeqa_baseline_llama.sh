export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH}
export PROJECT_ROOT=$(pwd)
export LOGS_ROOT=$(pwd)/logs
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

PYTHONPATH=. python3 -m accelerate.commands.launch --config_file \
  conf/accelerate/single_gpu.yaml \
  src/context_compression/run.py \
  +experiments_longbench_narrativeqa=evaluate_llama_compress_zeroshot_qa_narrativeqa \
  models.target_token=128000 \
  models.split_size=512 \
  models.condition="question" \
  models.normalize=True \
  trainers.evaluation_config.max_eval_samples=10 \
  custom_datasets.test.data_config.context_max_length=16384
