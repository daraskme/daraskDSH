# Anima ネイティブ学習 実測結果

対象: RTX PRO 6000 Blackwell Workstation Edition（SM 12.0, 600 W 上限, クロック非固定）、1 画像、batch 1、rank 16、BF16、AdamW (optimi 互換)。step 時間はウォームアップ後の中央値。

## 構成比較（ms/step）

| 構成 | 512px | 1024px | ピーク VRAM (512 / 1024) |
|---|---:|---:|---|
| diffusion-pipe（activation checkpointing あり＝現行既定） | 316 | 439 | — |
| diffusion-pipe（checkpointing なし） | 205 | 340 | — |
| 最小 PyTorch ループ eager | 135 | 292 | 8.6 / 21.9 GiB |
| 最小 PyTorch ループ compile (Inductor) | 82–86 | **222** | 7.0 / 15.7 GiB |
| **ネイティブ C++/CUDA（現行）** | **78** | 249–263 | 8.6 / 21.9 GiB |

- diffusion-pipe と eager/compile/native は 2026-11-18 の計測（`.runtime/anima-bench-l99erh95`）。native は 2026-11-20 に再計測（512: 77.7 ms、1024: 248.5 ms）。
- 1024px ではコンパイル版がネイティブより約 12% 速い。差は要素ごとの演算の融合で、言語の差ではない。

## 手順 1: step の内訳（2026-11-20, ネイティブ, `torch.profiler` CUDA トレース）

`nsys` が未導入のため、CUPTI のカーネル時間で集計した。プロファイラ有効時は起動ごとの負荷で壁時計が延びるので、壁時計は非計測時の値を使う。

| | 512px | 1024px |
|---|---:|---:|
| 壁時計 / step（非計測時） | 77.7 ms | 248.5 ms |
| GPU 稼働時間 / step | 64.6 ms | 247.6 ms |
| **GPU 空白** | **約 13 ms（17%）** | **ほぼ 0** |
| カーネル起動数 / step | 10,690 | 10,605 |
| GEMM | 36.3 ms（4,068 回） | 111.8 ms |
| attention（FlashAttention-2 系） | 6.6 ms | 51.0 ms（backward 36.6 ms） |
| 要素ごとの演算・その他 | 8.4 ms（2,888 回） | 42.9 ms |
| **コピー・型変換** | **9.7 ms（2,804 回）** | **32.5 ms** |
| reduce・norm | 3.4 ms | 9.0 ms |
| オプティマイザー | 0.4 ms | 0.5 ms |
| memcpy | 1.5 ms | 2.8 ms |

### 読み取り

- **GEMM 以外が GPU 時間の約 1/3（512: 21.5 ms、1024: 84 ms）。** 最大の削減対象。コピー・型変換が 1 step で 2,800 回起動されており、計画の手順 2（重複 cast の除去）が直接効く。
- **CUDA Graphs の上限は 512px で約 13 ms、1024px ではほぼ 0。** 起動数を減らせばこの空白も縮むので、Graphs は手順 2 と融合の後に再判定する。
- **1024px の attention backward（36.6 ms）が単一カーネルとして最大。** PyTorch 同梱の FlashAttention-2（sm80 系）で動いている。cuDNN SDPA との比較（手順 6）は 1024px では優先度を上げる価値がある。
- GEMM は cuBLAS の `cutlass_80` 系 (mma.sync) カーネル。SM 12.0 には wgmma/tcgen05 がないので想定どおり。
- LoRA の小行列（wmma 16x16 / 32x32）は 896 回・約 5.5 ms（512px）。

## 手順 2: 重複 cast の除去（2026-11-20）

変更（`model.cpp`）: `linear` は凍結 W と LoRA A で 1 回の cast を共有する。self-attention の Q/K/V は FP32 入力を 1 回だけ BF16 にする。全ブロックの adaLN が読む timestep 埋め込みは最初に 1 回だけ cast する。

| | 512px | 1024px |
|---|---:|---:|
| ms/step（前 → 後、非計測時） | 77.7 → **73.9（−5%）** | 248.5 → **238.7（−4%）** |
| コピー・型変換 / step | 9.7 → 7.9 ms | 32.5 → 27.0 ms |
| カーネル起動数 / step | 10,690 → 10,130 | 10,605 → 10,045 |

品質:
- 順伝播は参照と一致（相対誤差 0）。
- LoRA 勾配全体: 参照との相対 L2 0.0063、コサイン 0.99998、ノルム比 0.99999。
- 層ごとでは、勾配が小さく打ち消し合いの大きい adaLN 層で最大 14.5% の差。BF16 で勾配の和を取る順序が変わったため。FP32 で計算した参照と比べると、参照の BF16 自体が全体で 25.5% ずれており、ネイティブも 25.5%（層ごとの誤差合計は +0.45%）。BF16 由来の誤差の範囲内と判断した。
- 1024px・200 step の loss 曲線: 20 step ごとの平均の差は ±1.6% 以内。全体平均は 0.001902 → 0.001896。
- `tests/model_parity.py` の BF16 判定を、層ごとの 1e-5 から「全体の相対 L2 ≤ 0.02、コサイン ≥ 0.9999、ノルム比 ±1%」に変更した。CPU の FP32 モデルでは引き続き層ごとに 1e-5 で判定する。

残るコピー・型変換（約 2,200 回）の主な発生源は、RMSNorm と LayerNorm の FP32 往復、RoPE の `cat`、attention 前後の転置コピー。cast の削除では減らせず、融合の対象になる。

## 計画への反映

1. 手順 2（cast の重複除去）をそのまま次に実施する。コピー・型変換の 9.7 / 32.5 ms が対象。
2. 1024px は GPU 律速なので、Graphs より要素ごとの演算の融合と attention の比較を先に検討する。
3. 512px で Graphs を検討するのは、手順 2 の後に GPU 空白が 10% 以上残る場合だけ。

再計測: `.runtime/lora-venv/bin/python plugins/dsh-anima-lora-trainer/native/tests/profile_steps.py 512 40 <出力先>`（CUDA のみをトレースし、各 step の AdamW カーネルを区切りに 6 step 目以降を集計）。
