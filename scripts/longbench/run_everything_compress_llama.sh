#!/bin/sh
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH}
export PROJECT_ROOT=$(pwd)
export LOGS_ROOT=$(pwd)/logs
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

# Single-doc
sh ./scripts/longbench/eval_narrativeqa_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_qasper_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_multifieldqa_compress_llama_zero_shot.sh

# Multi-doc
# sh ./scripts/longbench/eval_wikimqa_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_hotpotqa_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_musiqueqa_compress_llama_zero_shot.sh

# Summarization
# sh ./scripts/longbench/eval_longbench_summarization_llama_govreport.sh
# sh ./scripts/longbench/eval_longbench_summarization_llama_qmsum.sh
# sh ./scripts/longbench/eval_longbench_summarization_llama_multinews.sh

# Few-Shot Learn
# sh ./scripts/longbench/eval_trec_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_samsum_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_triviaqa_compress_llama_zero_shot.sh

# Synthetic
# sh ./scripts/longbench/eval_passage_count_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_passage_retrieval_compress_llama_zero_shot.sh

# Code
# sh ./scripts/longbench/eval_lcc_compress_llama_zero_shot.sh
# sh ./scripts/longbench/eval_repobench_compress_llama_zero_shot.sh
