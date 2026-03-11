### Setting up the environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Remove last commit
```bash
git reset --soft HEAD~1
git push --force-with-lease
```

### Run Code in the background
```bash
nohup bash ./scripts/longbench/run_everything_compress_llama.sh > run.log 2>&1 &
```

### What considers a fair evaluation?
以下是公平比較的設定建議：

1. 設定相同的「文本長度」(max_length)

作法：兩邊的 max_length 都設定為夠大的值（例如 32K 或乾脆用 LongBench 預設的截斷長度）。

目的：確保 FINCH 和 LiveKVQuant 都是「讀取一模一樣的完整文章」。不要在 Dataset 階段就把文章截斷。

2. 設定相同的「記憶體預算」(Compression Ratio / Target Tokens)

作法：這是你們真正的比較基準。如果 FINCH 設定 target_token = 3000，代表它壓縮後只留 3000 個 token 的 KV Cache。你的 LiveKVQuant 也必須設定讓系統最終只保留 3000 個 token（或等價的壓縮比例）。

目的：比較在**「同樣的記憶體預算下」**，誰保留的資訊更好（F1 Score 較高）。

3. 尊重兩者演算法的「運作特性」(如 FINCH 的 split_size)

FINCH 原本的設計就是靠 split_size (例如 512) 一小段一小段處理，來避免一次算超長 Attention 矩陣造成的記憶體爆炸（Peak Memory）。

比較方式：保留 FINCH 的 split_size 讓它跑。你的 LiveKVQuant 照你的邏輯跑。

預期觀察：

Max Memory Peak：FINCH 因為有切塊，Peak 可能較低；你的方法如果是一次吃進去再壓縮，Peak 可能較高。

Latency：FINCH 切塊循序處理會拖慢速度（Prefill 時間長）；你的方法如果是平行處理，Latency 可能大勝 FINCH。

F1 Score：比較誰的淘汰機制比較聰明。

## Best records:
Narrativeqa: 4096 (max_length)
Qasper: 8192 (max_length)
Multifieldqa_en: 2048 (max_length)
