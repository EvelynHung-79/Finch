#!/bin/sh
cd "$(dirname "$0")/.."
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH}
export LOGS_ROOT=$(pwd)/logs
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

# Single-doc
# sh ./scripts/longbench_v1/eval_narrativeqa.sh
# sh ./scripts/longbench_v1/eval_qasper.sh
# sh ./scripts/longbench_v1/eval_multifieldqa.sh


# Multi-doc
# sh ./scripts/longbench_v1/eval_wikimqa.sh
# sh ./scripts/longbench_v1/eval_hotpotqa.sh
# sh ./scripts/longbench_v1/eval_musiqueqa.sh

# Summarization
# sh ./scripts/longbench_v1/eval_govreport.sh
# sh ./scripts/longbench_v1/eval_qmsum.sh
# sh ./scripts/longbench_v1/eval_multinews.sh

# Few-Shot Learn
sh ./scripts/longbench_v1/eval_trec.sh
sh ./scripts/longbench_v1/eval_samsum.sh
sh ./scripts/longbench_v1/eval_triviaqa.sh

# Synthetic
sh ./scripts/longbench_v1/eval_passage_count.sh
sh ./scripts/longbench_v1/eval_passage_retrieval.sh

# Code
sh ./scripts/longbench_v1/eval_lcc.sh
sh ./scripts/longbench_v1/eval_repobench.sh

# nohup bash ./scripts/run_everything.sh > run.log 2>&1 &