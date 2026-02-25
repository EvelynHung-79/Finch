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

### Max_length per Task
Single-doc
任務: narrativeqa          | 最高 Token 數: 65287
任務: qasper               | 最高 Token 數: 21118
任務: multifieldqa_en      | 最高 Token 數: 14960

Multi-doc
任務: hotpotqa             | 最高 Token 數: 16341
任務: 2wikimqa             | 最高 Token 數: 16331
任務: musique              | 最高 Token 數: 16348

Summarization
任務: gov_report           | 最高 Token 數: 51392
任務: qmsum                | 最高 Token 數: 30389
任務: multi_news           | 最高 Token 數: 13935

Few-Shot Learn
任務: samsum               | 最高 Token 數: 17974
任務: trec                 | 最高 Token 數: 11378
任務: triviaqa             | 最高 Token 數: 23299

Synthetic
任務: passage_count        | 最高 Token 數: 28965
任務: passage_retrieval_en | 最高 Token 數: 15144

Code
任務: lcc                  | 最高 Token 數: 30150
任務: repobench-p          | 最高 Token 數: 39125

## Target Token Per Task
Single-doc                  1 Warmup Chunk   2 Warmup Chunk
任務: narrativeqa          |  5119          |  5500          |
任務: qasper               |  1312          |  1692          |
任務: multifieldqa_en      |  1554          |  1934          |

Multi-doc
任務: hotpotqa             |  2736          |  3116          |
任務: 2wikimqa             |  1638          |  2018          |
任務: musique              |  2388          |  2768          |

Summarization
任務: gov_report           |  2484          |  2864          |
任務: qmsum                |  3096          |  3476          |
任務: multi_news           |  924           |  1304          |

Few-Shot Learn
任務: samsum               |  1992          |  2372          |
任務: trec                 |  2444          |  2824          |
任務: triviaqa             |  1713          |  2093          |

Synthetic
任務: passage_count        |  2772          |  3152          |
任務: passage_retrieval_en |  3249          |  3629          |

Code
任務: lcc                  |  1828          |  2208          |
任務: repobench-p          |  698           |  1078          |

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