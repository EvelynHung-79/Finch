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

## Target Token Per Task
- Compression Rate in LiveKVQUantP is assuming n_warmup = 1
- target-token will be set by task
| 任務 (Task)        | 平均長度 (N) | Compression rate in LiveKVQuantP  | target_token (2048)| target_token (4096)|
|--------------------|-------------|-----------------------------------|--------------------|--------------------|
| NarrativeQA        | 18,405      | 0.2921                            | 598                | 1197
| Qasper             | 3,619       | 0.3749                            | 768                | 1536
| MultiFieldQA       | 4,559       | 0.3536                            | 724                | 1449
| HotpotQA           | 9,149       | 0.3126                            | 640                | 1281
| 2WikiMultihopQA    | 4,885       | 0.3482                            | 713                | 1426
| Musique            | 7,798       | 0.3197                            | 655                | 1309
| GovReport          | 8,169       | 0.3175                            | 650                | 1301
| QMSum              | 10,546      | 0.3072                            | 629                | 1258
| MultiNews          | 2,113       | 0.4483                            | 918                | 1836
| SAMSum             | 6,258       | 0.3314                            | 679                | 1358
| TriviaQA           | 8,015       | 0.3184                            | 652                | 1304
| TREC               | 5,176       | 0.3439                            | 704                | 1409
| Passage Retrieval  | 9,288       | 0.3120                            | 639                | 1278
| Passage Counting   | 11,141      | 0.3053                            | 625                | 1251
| RepoBench-P        | 5,622       | 0.3382                            | 693                | 1385
| LCC                | 1,235       | 0.5737                            | 1175               | 2350


### Explanation to ...
target_token：壓縮後的目標 Token 數量（預期保留長度）。這代表在經過 FINCH 壓縮機制後，模型 KV Cache 最終要保留下來的總 token 上限。例如設定為 3000，系統就會根據注意力分數（或相似度），從長文本中挑選出最重要的約 3000 個 token 來讓模型生成答案。

max_length：輸入文本的最大截斷長度。在資料集進行分詞（Tokenization）時，單一輸入序列（包含 System Prompt、Context 及 Question）允許的最大長度。這通常受限於模型本身支援的 Context Window 或是使用者指定的上限。

doc_stride：文本分塊時的重疊步長。當原始長度超過 max_length 導致文本必須被切割成多個片段（Chunks）輸入時，doc_stride 定義了相鄰兩個片段之間要「重疊」幾個 token。這樣可以避免句子剛好在切割處被截斷而喪失上下文語意。

split_size：壓縮過程中的運算分塊大小（Chunk Size）。在執行上下文壓縮時，為了避免記憶體爆炸，程式不會一次把超長文本全部丟進去算，而是將文本以 split_size（例如 512 個 token）為單位切分成一小段一小段（Segments）。模型會分批處理這些段落，計算注意力分數並篩選出重要的 token。

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
