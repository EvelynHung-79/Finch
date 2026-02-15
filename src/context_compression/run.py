import logging
import os
import pathlib
import warnings
import datasets
import hydra
import omegaconf
import transformers
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

# --- 1. 手動設定 PROJECT_ROOT 與 過濾 Log ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 強制過濾無意義的警告訊息
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("hydra").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)

# 2. 屏蔽 Hugging Face 相關日誌
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# 3. 屏蔽環境與核心警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

# --- 2. 核心邏輯保持不變 ---
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: trainable_params += param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param}")

def run(cfg: omegaconf.DictConfig):
    accelerator = Accelerator(log_with="wandb")
    
    # 設定標準 Python Logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # 實例化 Tokenizer 與 Model
    logger.info(f"Loading Tokenizer...")
    tokenizer = hydra.utils.instantiate(cfg.tokenizers, _recursive_=False)

    if tokenizer.pad_token is None:
        # Llama 3 預設沒有 pad_token，通常設為 eos_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Setting pad_token to eos_token: {tokenizer.pad_token}")
    
    logger.info(f"Loading Model: {cfg.models['_target_']}")
    model = hydra.utils.instantiate(cfg.models)
    
    if model.dtype != torch.bfloat16:
        print("Switching model to bfloat16")
        model = model.to(torch.bfloat16)
    
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to match tokenizer length: {len(tokenizer)}")

    # 實例化 Trainer 並執行測試
    trainer = hydra.utils.instantiate(cfg.trainers, _recursive_=False)
    
    if cfg.trainers.mode in ["train_eval", "eval"]:
        ds_eval_obj = hydra.utils.instantiate(cfg.custom_datasets.test, tokenizer=tokenizer, model=model, _recursive_=False)
        predictor_config = cfg.predictors
        logger.info("Starting Evaluation!")
        trainer.evaluate(accelerator=accelerator, model=model, tokenizer=tokenizer, ds_eval_obj=ds_eval_obj, predictor_config=predictor_config)

@hydra.main(config_path="../../conf", config_name="default", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()