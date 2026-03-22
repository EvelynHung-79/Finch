---
name: finch_project_status
description: FINCH KV cache 壓縮專案在 Llama 3.1 上的實驗進度與結論
type: project
---

## 專案目標
將 FINCH（prompt-guided KV cache 壓縮）移植到 Llama 3.1 8B-Instruct，在 LongBench narrativeqa 上提升 F1 分數。

## 比較框架（重要）
- **FullKV** = 真正的 baseline（不壓縮，上限）
- **FINCH (per_head_vote=False)** = 原始方法實作結果（不是 baseline，是「我們的方法的起點」）
- **FINCH (per_head_vote=True)** = 改進後的方法
- 目標是縮小 FINCH vs FullKV 的 gap

## 核心方法限制（不能改）
FINCH 的核心是 chunk-wise, prompt-guided token dropping，這是論文的定義。
**不能改成「一次性全局壓縮取 top-k」**，因為那樣就不是 FINCH 了。
在框架內能改的只有：head selection 方式、condition、normalize 等超參數。

## 已確認的根本問題
Llama 3.1 vs Llama 2 分數差，三個根本原因：
1. GQA（8 KV heads, 32 Q heads）→ attention selection 信號比 Llama 2（32 KV heads）弱
2. RoPE theta=500000 → attention 分佈更均勻 → top-k 更難選到真正重要的 token
3. 迭代壓縮複合誤差（narrativeqa 約 58 輪 chunk）

## 實驗結果（narrativeqa, 200 samples）

| 實驗 | split_size | target_token | 備註 | F1 |
|------|-----------|-------------|------|-----|
| Baseline | 512 | 9137 (fixed) | per_head_vote=False | 12.62 |
| A | 2048 | 9137 (fixed) | | 12.81 |
| B | 4096 | 9137 (fixed) | | 9.15 |
| C | 512 | per-sample (rate=0.3063) | | 8.25 |
| D | 2048 | per-sample (rate=0.3063) | | 9.71 |
| E | 4096 | per-sample (rate=0.3063) | 27 OOM | 5.22 |
| **per_head_vote** | 512 | 9137 (fixed) | vote_r=k | **13.70** ← 最佳 |
| vote_r=k/2 | 512 | 9137 (fixed) | vote_r=4568 | 11.61（變差）|

**結論：** per_head_vote 是唯一有實質改善的方向（+1.08）。split_size 和 adaptive target 無效。vote_r=k/2 更差，代表 Llama 3.1 attention 太分散，嚴格投票反而丟掉重要 token，vote_r=k（最寬鬆）最佳。

## Per-head vote 設計
- 把 32 Q heads reshape 成 8 KV groups（每組 4 Q heads）
- 每個 KV group 獨立選 top-r KV positions（vote_r 目前 = k，最寬鬆）
- 用 vote count 取代 uniform sum 作為 token 重要性分數
- 實作：`src/models/modeling_llama.py` 的 `make_attn_hook`，`per_head_vote=True` 分支

## 實驗設計原則
- **target_token 不動（固定 9137）**：需要在同一基準下比較自己的方法 vs 原始 FINCH，不能改 target_token

## Multi-doc QA condition 實驗（2026-03-18）

**實驗**：2wikimqa 改 `condition=all`（原為 `condition=question`）

| condition | F1 |
|-----------|-----|
| `question`（v3 baseline） | 32.89 |
| `all`（實驗） | 24.37 |

**結論**：condition=all 反而更差（-8.5 分）。
原因：Llama 3.1 attention 本來就比 Llama 2 更平坦，condition=all 把 context self-attention 納入後偏向保留高頻詞，稀釋了 question attention 的聚焦信號，兩跳 evidence 都抓不到。
**condition=question 仍是最佳設定，不需動 question_prompt。**

Multi-hop 任務的低分是 FINCH 方法的固有限制：single-query compression signal 無法同時保留兩個語義獨立的 evidence，調參無法根本改善。

## 下一步候選
- vote_r 方向已確認無效，不再調整
- condition=all 已驗證無效（2wikimqa -8.5 分），不再嘗試
- 跑其他 LongBench 任務驗證 per_head_vote 泛化性（qasper, multifieldqa_en）

## 重要程式碼改動記錄
- `src/models/modeling_llama.py`：hook 改用 try/finally；output_attentions=False；新增 per_head_vote 邏輯
- `src/predictors/language_modeling_predictor.py`：OOM 後 hooks 殭屍問題已修
- per-sample adaptive target（compression_rate）已測試並移除（效果差）

## 三 task 完整比較

| Task | target_token | FullKV | FINCH原始 | FINCH+per_head | Gap縮小 |
|------|-------------|--------|----------|---------------|--------|
| narrativeqa | 9137 | 26.53 | 12.62 | **13.70** | 1.08 / 13.91 gap → 12.83 |
| qasper | 2132 | 46.94 | 35.17 | 34.77（未跑per_head） | - |
| multifieldqa_en | 5192 | 48.15 | 35.72 | **39.87** | 4.15 / 12.43 gap → 8.28 |

multifieldqa_en gap 縮小 33%，因為 chunk 數少（~13 輪），per_head_vote 效果能積累。
narrativeqa gap 只縮小 7.8%，因為 58 輪迭代複合誤差是框架本身的特性，無法在不改核心方法的前提下消除。

## 結論：當前上限
在 chunk-wise FINCH 框架內，per_head_vote 是目前能做的最有效改進。
narrativeqa 的大 gap 來自迭代複合誤差，這是 FINCH 方法本身在長文件上的固有限制，不是超參數問題。

## 當前 eval script 設定（narrativeqa）
- target_token=9137, split_size=512, condition=question, normalize=True, pin_header=True
- per_head_vote=True, vote_r=null（等於 k）

## Completion Mode 修正（2026-03-15）
trec / samsum / triviaqa / lcc / repobench-p 是 completion-style 任務（prompt 結尾為答案標籤 `Type:`/`Summary:`/`Answer:`/`Next line of code:`），模型應直接補全，不應套 chat template。
**根因**：`generate_input()` 原本一律加 `<|eot_id|><|assistant|>` header，導致這些任務輸出格式混亂（幻覺 few-shot context 內容、輸出數字序列等）。
**修正**：新增 `completion_mode: True` yaml 欄位，`llama31_dataset_custom.py` 讀取後在 `generate_input()` 跳過 chat template headers。
**已更新 yaml**：trec / samsum / triviaqa / lcc / repobench-p。
**待重跑**：上述五個任務。
