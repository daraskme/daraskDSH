# MiniMax H3 動画生成のネイティブ化・高速化計画

方針: **速度 ≥ 品質**。最小の変更で最大の短縮を取る。新しい抽象化や設定項目は、測定で必要と分かるまで追加しない。完成後は ComfyUI 経由の旧経路（ワークフロー `MiniMax H3 - Eros beta7 + VBVR Pro V2`）を削除する。

## 現状（2026-11-19 実測、ComfyUI v0.45.0、RTX PRO 6000 Blackwell 96 GB）

- 構成:
  - DiT: `10Eros_Max_h3_8stepTURBO_beta7`（BF16、50 ブロック、hidden 5376、56 ヘッド × 128、SwiGLU 14336、約 19.3B パラメータ、38.5 GB）
  - LoRA: `H3_VBVR_Pro-V2`（強度 1）
  - サンプリング: SigmaShift 10/3、BlockSparseAttention（SLA、keep 10%、comfy_kitchen のカーネル）、res_multistep 8 step
  - テキストエンコーダー: Qwen3-VL 32B int8 convrot（26 GB）
  - 動画 VAE（fp16、5 GB）と音声 VAE
- 時間:

| 条件 | 時間 |
|---|---:|
| 初回（モデルのロード込み） 640×384×39 | 40.6 s |
| 2 回目以降 640×384×39（新しいプロンプトでも +0.1 s） | 4.0–4.6 s |
| 2 回目以降 1280×704×121（約 27k token） | 48.6–49.6 s |
| 　うちサンプリング 8 step | 約 38 s（4.8 s/step） |
| 　うち VAE デコードと mp4 化 | 約 10.3 s |

- トークン数（Astra の指摘で訂正）: 121 フレームは 124 フレームに切り上がり、潜在の時間長は 37。映像だけで 37×22×40 = 32,560 行、音声が 414 行、さらにテキスト分がある。
- 概算（未実測）: 1 forward で QKV・出力投影・MLP が約 1.27 PFLOP、dense の attention が約 1.56 PFLOP。SLA の keep 10% は「残すキーブロックの割合」であり、attention の時間が 10% になるわけではない（対角、条件 KV、音声の dense 行、選択とパックの処理が残る）。Q/K の RMSNorm と RoPE はすでに融合されている。
- 常駐時の VRAM は約 71 GB。LoRA 学習（約 17 GB）や Gemma（約 20 GB）と同じ GPU を使う。
- P-H3 の画像生成は、同じ DiT クラスの音声なし版（`minimax_h3_image`）。ここで作る実行系を次の P-H3 作業でそのまま使う。

## 進め方（Astra のレビューを反映。1 段ずつ測り、速くならなければ捨てる）

レビュー全文: [h3-astra-review.md](h3-astra-review.md)

1. **既存のグラフのまま内訳を測る** — 単体の計測スクリプトは作らない。実際のグラフで `PackedLayout.seq_len`、sparse producer が発火しているか、カーネル時間の内訳（GEMM・SLA・その他）を記録する。同じ生成済み latent で、映像のデコード（ViT3D）・音声のデコード・MP4 保存を分けて測る。速度用のシードは、事前に決めた異なる 3 つを使う（ノードのキャッシュで速く見えるのを避ける）。
2. **FP8 を正しい形で試す** — 同じ BF16 ベースに強度 1 の LoRA を高い精度で一度だけ反映し、BF16 版と、そこから量子化した FP8 版を比べる（`fp8_e4m3fn_fast` は scale 1 で活性を clamp するだけなので、この実験の代わりにならない）。確認するのは、FP8 GEMM が実際に使われているか、変換込みの秒数、LoRA が効いているか、映像と音声。Qwen の int8 convrot と SLA 内部の int8 は変えない。PSNR を採否の基準にはしない。
3. **デコードと保存** — 1 の内訳で保存が重ければ、同等画質のハードウェアエンコード（NVENC）を使う。映像のデコードが重ければ、ViT の演算を見る。
4. **常駐の採否** — 統一ロックや新しいキューは先に作らない。既存の排他の下で「H3 → 他のジョブ → H3」を一往復だけ測り、引き渡し時の実 VRAM と、復帰後に MP4 ができるまでの時間で決める（常駐 71 GB + Gemma 20 GB では余裕が約 5 GB しかない）。
5. **実行系** — 下の「実行系の形」のとおり、完全ネイティブで作る。2〜4 の測定は、移植後のネイティブ実行系で行う。

削除した項目: C++ への全面移植、融合カーネルの追加（Q/K の正規化と RoPE は融合済み）、`channels_last → cuDNN → BF16` という VAE の試行の並び（デコーダーは ViT3D）、「初回 15 s」の目標、P-H3 との共通化（P-H3 の計画で改めて扱う）。

## 実行系の形（決定: 完全ネイティブ）

ユーザーの決定により、ComfyUI のコードを実行時に一切使わない。DSH プラグイン `dsh-minimax-h3` に、C++/LibTorch と CUDA の実行系（Anima の学習エンジンと同じ構成）を作る。

| 部品 | 中身 | 出典・方針 |
|---|---|---|
| テキストエンコーダー | Qwen3-VL 32B のテキスト部分。第 50 層の hidden states を出す | int8 convrot の重みを層ごとに BF16 へ戻して計算する（プロンプトは短いので速度への影響は小さい） |
| DiT | 50 ブロックの single-stream。LoRA はロード時に重みへ反映する | ComfyUI の `comfy/ldm/minimax/model.py` を仕様として C++ に書き直す |
| SLA | sparse attention（sol_attn） | comfy-kitchen（Apache-2.0、GitHub で公開）の CUDA ソースを取り込んでビルドする |
| サンプラー | res_multistep 8 step、SigmaShift 10/3、AV の時間格子と条件付け | `nodes_minimax_h3.py` と `sampling.py` を仕様にする |
| 映像 VAE デコーダー（ViT3D）と音声 VAE デコーダー | — | `vae.py` と `audio_vae.py` を仕様にする |
| MP4 の作成 | 映像と音声の多重化 | ffmpeg（NVENC が使えれば使う） |

- **ライセンス**: ComfyUI（GPL-3.0）のコードを C++ に書き直したものは派生物にあたる。私的に使う分には問題ないが、配布する場合は GPL が適用される。comfy-kitchen は Apache-2.0。

### 移植の手順（Astra の第 2 回レビューを反映）

1. **既存グラフで計測し、基準のテンソルを採る** — 短い尺の 1 件で、AV の noise、条件、9 個の sigma、中間の latent（初回・2 回目・最後の step）、最終の AV latent、圧縮前の RGB、音量補正後の 32 kHz ステレオ波形を保存する。以降の部品は、まずこの値を直接入れて一致を確かめる。
2. **SLA を最初に確かめる（最大の危険）** — comfy-kitchen を v0.2.41 に固定し、Sol が使う翻訳単位とヘッダーだけを LibTorch の拡張としてビルドする（CUDA 12.8 以上）。SM 12.0 で、実際の形状の 1 ブロックを LibTorch の CUDA stream 上で動かす。ここで動かなければ計画を見直す。
3. **テキスト条件** — 次の 3 点で順に一致させる。
   - token ID 列（完全一致。T2V のタグはすべて 1）
   - Qwen の第 50 層の正規化前の出力（5120 次元）
   - `condition_proj` と LoRA を反映した token refiner（2 層、8 投影）の出力（5376 次元）

   int8 convrot は、行ごとの scale と group 256 の逆回転で BF16 に戻す。まず 1 投影で復元を確認し、旧経路の int8 との第 50 層の差と、新しいプロンプトでの時間を測る。許容できなければ int8 のまま計算する。条件は 1 回の生成につき一度だけ作る。
4. **DiT** — LoRA の 208 投影（ブロック 200 + refiner 8）をロード時に反映する。一致は「1 ブロック → 1 forward の AV 両方の出力」の順で確かめる。
5. **SLA の本番経路** — `sol_attn_chunked` と同じ動作にする。
   - `[text | audio | video]` の並び
   - 映像が始まる前までのブロックの KV を常に残す
   - 音声を含む query ブロックは dense で計算する
   - QKV は 4,096 行ずつ処理する
   - 各層の最初の step だけ統計を取り、以後は前の step の `kmean` と `vscale` を使う

   同じブロックの初回と次の step で、sink の範囲・統計・出力を比べる。
6. **サンプラー** — 仕様は `model_sampling.py:338–357`、`model_base.py:2230–2252`、`minimax/model.py:673–714`。
   - 映像の shift は 10、音声の shift は 3 で別々。サンプラーは映像側の sigma で両方を進める。
   - `res_multistep` は、最初と最後の step が Euler、中間が前の denoised を使う 2 次の更新。
   - noise は CPU の同じ生成器から、映像 → 音声の順に作る。

   一致は「9 個の sigma」「中間 step の映像と音声それぞれの DiT 入力・速度・denoised」で確かめる。直接入れた値で一致してから、シードからの通しの生成でも確かめる。
7. **VAE と MP4** — 映像は dully fp16 の `decoder.transformer_blocks` に加えて、latent の逆正規化、空間のタイル、時間方向の重なりと切り詰め。音声は `max(1, 5×std)` で割る正規化と、映像の尺に合わせた末尾の切り詰め。圧縮前の RGB と波形で比べてから、124 フレーム・24 fps の MP4 を確認する。
8. **FP8**（BF16 で音声付き MP4 が一致した直後、UI より前）— 同じマージ済みの重みから、DiT の投影だけを FP8 にする。出力は BF16 のまま SLA に渡す（producer は BF16 入力が前提）。採否は「SLA が引き続き使われていること」と「変換込みの総時間」で決める。
9. **DSH への組み込み** — チャットツール、パネル、LoRA 学習と共通のキューと GPU 排他。常駐するかどうかは「H3 → 他のジョブ → H3」の一往復を測って決める。
10. **旧経路の削除**

移植しないもの: 画像版の PDD サンプラー、負条件の forward、VSA、汎用のパッチ機構、vision と参照画像の入力、P-H3 との共通化、モデル全体の FP32 対照、VAE エンコーダー、MP4 全フレームの PSNR。

## 品質の確認

- 固定の 3 プロンプト × 固定シードで、旧経路（ComfyUI、BF16）と新経路の動画を並べて目視する。各フレームの PSNR も記録し、音声も聞いて比べる。
- FP8 は、目視で破綻（顔・手の崩れ、ちらつき、音声のノイズ）がなければ採用する。

## 計測

- 640×384×39 と 1280×704×121 の 2 条件。それぞれ初回と 2 回目以降、各 3 回。
- 「依頼から mp4 ができるまでの時間」を主な指標にする。

## 完了条件

- 1280×704×121（124 フレーム）の 2 回目以降について、デコードと保存を含む総時間の目標を、計測と FP8 の結果を見てから確定する（暫定 35 s 以下。35 s に届くには、FP8 の対象部分が元の時間で 28 s 以上必要）。
- DSH のチャットとサイドバーから、ネイティブの実行系で H3 動画を生成できる。部品ごとの一致確認と、旧経路と比べた速度・映像・音声の確認が済んだら、ComfyUI の H3 ワークフローを削除する。

## 進捗（2026-11-19）

### 手順 1: 既存グラフでの計測と基準テンソル（完了）

- 計測ツール: `plugins/dsh-minimax-h3/tools/capture_reference.py`。ComfyUI 本体（`main.py`）を 2 つ目のインスタンス（ポート 8189）として起動し、`execution` が読み込まれた直後にフックを入れる。本番の ComfyUI はそのまま。
- 基準テンソル（640×384×39、seed 42、プロンプトは `prompt.txt`）: `.runtime/h3-native/reference-small/reference.safetensors`（47 個）。
  - token ID（33 個）、Qwen の出力（[1, 33, 5120]、FP32）、refiner の入力と出力（[33, 5376]）
  - sigma 9 個
  - DiT の入出力（step 0・1・7。映像 [1,24,12,24,40]、音声 [1,32,2,65]、context [1,33,5376]、text_token_tags）
  - 最終の AV latent
  - 圧縮前の RGB（[39,384,640,3]）と音声波形（[1,2,52000]）
  - noise は個別に採れなかったが、sigma₀ = 1 のため step 0 の DiT 入力 `dit.call0.x` が noise そのものになる。
- 1280×704×121（124 フレーム）、2 回目以降の内訳（`.runtime/h3-native/large-timings`）:

| 区間 | 時間 |
|---|---:|
| サンプリング 8 step | 39.0 s（1 step 4.53–4.76 s） |
| 　1 step の GEMM | 3.53 s（77%、cuBLAS `nvjet_sm120`、約 360 TFLOPS） |
| 　SLA（`sol_exact` と `sol_producer`） | 0.59 s（13%） |
| 　要素ごとの演算・コピー・LayerNorm | 約 0.5 s（10%） |
| 映像のデコード | 8.3 s |
| MP4 の保存 | 1.9 s |
| 音声のデコード | 0.07 s |

- GEMM はすでに BF16 のピーク近くで動いている。残る大きなレバーは FP8（GEMM）と映像デコード。

### 手順 2: SLA を SM 12.0 のネイティブで動かす（完了・成功）

- comfy-kitchen v0.2.41（commit cbc44a5、Apache-2.0）から、Sol の 7 つの `.cu` と `sol_layout.cuh` とその依存ヘッダー 5 個を、ライセンス表記ごと `plugins/dsh-minimax-h3/native/kitchen/vendor/` に取り込んだ（無改変）。
- ツールチェーン:
  - H3 の実行系は、ComfyUI と同じ PyTorch 2.15 + CUDA 13.2 にそろえる。cu132 の cuBLAS が `nvjet_sm120` の GEMM を使うため（Anima で使っている 2.9.1 + cu128 では `cutlass_80` になる）。
  - CUDA 12.8 の nvcc は GCC 15 の標準ライブラリを解釈できないため、PyPI の `nvidia-cuda-nvcc`・`nvidia-nvvm`・`nvidia-cuda-crt` 13.2.78 を `.runtime/h3-native/toolchain` に入れて使う（バージョンを揃えないと PTX のバージョンが合わない）。
- 構成: カーネルは `libdsh_kitchen.so`（C ABI、`sm_120a`、`native/build_kitchen.py` でビルド）にまとめ、`kitchen/kitchen_ops.cpp` が `torch.ops.dsh_kitchen.*` として LibTorch の現在の CUDA stream で起動する。
- 確認（`native/kitchen/test_sol_parity.py`）: 実形状（T = 33 + 414 + 32,560 = 33,007、56 ヘッド × 128、4,096 行ずつ、keep 10%、条件と音声の sink）で、同じ `sol_attn_chunked` ドライバーを wheel の `_C` と自前ビルドに差し替えて実行した。

| | out / kmean / vscale | wheel | 自前ビルド |
|---|---|---:|---:|
| step 1（統計の初回取得） | ビット一致 | 35.9 ms | 35.2 ms |
| step 2（前 step の統計を使用） | ビット一致 | 32.4 ms | 27.6 ms |

- 次は手順 3（テキスト条件）。Python のドライバー部分（統計、top-k の閾値、RoPE のパッキング）は、DiT の移植と一緒に C++ へ移す。

### 手順 3: テキスト条件（完了）

実装は `plugins/dsh-minimax-h3/native/te.cpp`（Qwen3-VL 32B の第 50 層まで）と `condition.cpp`（condition_proj、token refiner、LoRA のマージ）。確認は `native/test_text_parity.py`。

| 照合点 | 結果 |
|---|---|
| token ID（33 個） | 完全一致（`tokenizers` + 書き出した `qwen3vl_tokenizer.json`。Rust 版も同じ実装を使う） |
| Qwen 第 50 層の正規化前の出力（[33, 5120]、FP32） | 相対 L2 5.5e-7 |
| condition_proj + LoRA 反映済み refiner（[33, 5376]、BF16） | 同じ入力からビット一致 |
| 通し（ネイティブの Qwen → refiner） | 相対 L2 0.5%（Qwen 出力の FP32 の微差が、condition_proj の BF16 化で丸めの境界をまたぐため。BF16 の丸め誤差と同じ大きさ） |

分かったこと（実装の前提を修正した点）:
- **ComfyUI のテキストエンコーダーは活性を int8 に量子化していない。** int8 convrot の重みを読み込みのたびに戻し（W = (q × scale) H）、FP32 で計算している。最初に作った int8 活性の経路は、出力が ComfyUI から 2.2% ずれ、FP32 で重みを戻した場合とは 2e-7 で一致した。そのため int8 活性の経路は削除した。
- 重みを毎回回転させると 1 回 554 ms かかる。H が対称なので x Wᵀ = (x H)(q × scale)ᵀ と書き換え、33 行の活性だけを回転させた（約 0.2–0.37 s。1 回の生成で 1 度だけなので、生成全体の 1% 未満）。専用の int8 カーネルを作れば約 20 ms まで縮められるが、効果が小さいので見送る。
- **LoRA のマージ方法**: ComfyUI（dynamic VRAM）は、`w + (strength × alpha / rank) × (up @ down)` を、`up @ down` も含めてすべて重みの dtype（BF16）で計算している。FP32 や FP16 で計算した場合とはビット単位で一致せず、BF16 で計算した場合だけビット一致した（refiner 内部の各 linear の重みを実際に記録して確かめた）。DiT の 200 投影にも同じ方法を使う。
- condition_proj の入力は BF16 にキャストしてから計算する（ビット一致を確認）。

### 手順 4: DiT 本体（完了・dense attention でビット一致）

- 実装: `native/dit.cpp`（T2V。バッチ 1、keyframe・参照・mask なし）。対象は、packed layout（[text | audio | video]、float64 の位置）、3D RoPE テーブル（128 次元のうち先頭 96 次元を回転）、曲線形の AdaLN（`adaln_t_table` [1025, 8] を線形補間し、FP32 で投影）、セグメントごとの in-place 変調とゲート、SDPA、SwiGLU、FP32 の最終層、`forward` 側での音声スケジュールの換算（audio_scale = 10/3）。LoRA は 208 投影（ブロック 200 + refiner 8）を、ロード時に in-place でマージする（BF16、ビット一致の方式）。
- 基準: BlockSparseAttention ノードを外したグラフで ComfyUI を実行して採った（`.runtime/h3-native/reference-small-dense`。ブロック 0 の入出力、t_emb、RoPE テーブル、変調セグメントも含む）。確認は `native/test_dit_parity.py`。

| 照合点 | 結果 |
|---|---|
| ブロック 0（同じ入力から） | ビット一致 |
| forward 全体の映像・音声の出力（step 0・1・7） | ビット一致 |

- 分かったこと: per-head RMSNorm と RoPE を ATen で書くと 1.6e-4 ずれる（RMSNorm の出力を BF16 に丸めてから回転するため）。comfy-kitchen の融合カーネル `rms_rope.cu` を vendor（`kitchen/vendor/` に、依存ヘッダーと一緒に無改変で取り込み）し、ComfyUI と同じように qkv バッファ上で in-place に呼ぶと、attention の出力がビット一致した。
- 速度: 計測中にユーザーの Anima LoRA 学習が同じ GPU で動いていた（使用率 100%、600 W、2.4 GHz に制限）ため、比較できていない。単体の GEMM も 164 TFLOPS しか出ていなかった。GPU が空いているときに測り直す。
