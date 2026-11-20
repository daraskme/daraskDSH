# P-H3 image（画像生成・編集）のネイティブ化計画

方針: **速度 ≥ 品質**。H3 動画で作った実行系（`plugins/media-shared/native/`）をそのまま使い、足りない部品だけを足す。新しい抽象化は作らない。完成後は ComfyUI のワークフロー `P-H3-Image Text to Image (int8)` と `P-H3-Image Edit (int8)` を削除する。

## 構成（チェックポイントから確認、2026-11-19）

| 部品 | ファイル | 中身 |
|---|---|---|
| DiT | `p-h3-image_int8_convrot`（21 GB） | H3 と同じ 50 ブロック・hidden 5376・音声入出力つき。ブロックの 4 投影（qkv・out・fc1・fc2）が int8 convrot（行ごとの scale、group 256 の回転）。refiner と小さな層は BF16/FP32 |
| テキストエンコーダー | `p-h3-image_text_encoder_int8_convrot`（5 GB） | Qwen3-VL 4B（36 層、hidden 2560、vision 塔と deepstack つき）+ `text_connector`（第 13・19・25・31・36 層の hidden を標準化 → 12800→5120 の射影 → 先頭トークンを sink に置換） |
| プロンプト拡張 | `p-h3-image_prompt_upsampler_int8_convrot`（9.7 GB） | Qwen3.5-9B 系（hidden 4096、32 層、線形注意（gated delta）と通常注意のハイブリッド、MTP ヘッド、vision 塔）。ワークフローでは greedy・思考モード・最大 2048 トークン（編集は 3072）。スイッチで外せる |
| VAE | `p-h3-image_vae_int8_convrot`（3 GB） | 3D CNN エンコーダー（編集の参照画像用）+ ViT デコーダー（36 層、int8 convrot の投影、1 フレーム） |

- サンプリング: Euler、手動 sigma 8 段（1.0 … 0.528718, 0）、SigmaShift 1/1、BasicGuider（負条件なし）。
- latent: 1 フレームの AV latent（映像 [1, 24, 1, H/16, W/16] + 音声 2 フレーム）。既定の大きさは 2912×2912（約 8.5 MP、映像 8,281 トークン）。
- ComfyUI の int8 は W8A8: 活性を行ごとに動的に int8 化し（convrot の回転を融合）、comfy-kitchen の int8 GEMM で計算する。VAE デコーダーの注意も int8（`int8_attention`）。

## 進め方（1 段ずつ ComfyUI の基準と照合する。Astra のレビューを反映）

レビュー全文: [ph3-image-astra-review.md](ph3-image-astra-review.md)

1. **基準を採る** — H3 と同じ `capture_reference.py` で T2I（拡張なし、1024×1024 と 2912×2912）と編集（**非正方形**の参照 1 枚）を 1 件ずつ実行し、token ID、text_connector の出力、DiT 直前の context と token_tags、FP32 の noise、DiT の入出力（step 0・1・7）、最終 latent、保存直前の RGB、各ノードの時間を保存する。あわせて DiT・テキスト・VAE の各 int8 層が実際に通る経路（入出力の dtype、scale・回転・bias の順、注意のバックエンド。2912 角で何が選ばれたか）を記録する。速度はテンソル保存を外して測る。
2. **int8 の線形層** — 1 で実際に通ると分かった comfy-kitchen v0.2.41 の int8 カーネルだけを、Sol と同じく `kitchen/vendor/` に取り込み、`libdsh_kitchen.so` に足す。実形状の 1 投影でビット一致を確認する（テキストエンコーダーは H3 のように活性 FP32 の可能性があるので、1 の記録に従う）。
3. **DiT** — `dit.cpp` の 4 投影を「BF16 か int8 か」で切り替えるだけにする（重みの dtype で判定）。1 ブロック → 1 forward の順に照合。画像は 1 フレームなので SLA は使わない（ComfyUI のワークフローにもない）。
4. **テキスト条件（T2I）** — Qwen3-VL 4B のテキスト部分と text_connector（5120 次元）、続けて画像版 DiT の condition_proj と refiner（既存の `condition.cpp`）。token ID → 5 層の hidden → connector の出力 → **DiT 直前の context（5376 次元）と token_tags** の順に照合する。`te.cpp` は 4B の形状と 5 層の取り出しに合わせるだけにする。
5. **サンプラー** — Euler と手動 sigma 9 値、shift 1/1、audio_scale 1、SLA なし。画像の 2 形状（映像 [1, 24, 1, H/16, W/16]、音声 [1, 32, 2, 2]）を直接 `initial_noise` に渡す（動画用の `latent_shapes` は長さ 5 に切り上げるので使わない）。`sampler.py` に関数を 1 つ足し、FP32 の noise と全 8 回の Euler を照合する。
6. **VAE デコーダー** — 1 で確認した画像版の経路（latent の逆正規化、タイルの範囲とバッチ数、画素の逆正規化）どおりに `vae.py` の既存関数をつなぎ、int8 投影と int8 注意を足す。2912 角は 33,124 パッチになるのでタイル経路を必ず確認する。同じ最終 latent から保存直前の RGB までをビット一致で確認する。
7. **ここで T2I を DSH に出す**（H3 と同じパネル・キュー）。速度を ComfyUI と比べる。
8. **編集** — vision 塔（Qwen3-VL の ViT と deepstack）と VAE エンコーダー（3D CNN）は、入口の前処理（縮小・寸法の丸め）を別々に照合する（共通化しない）。DiT は参照 1 枚に限り、実グラフの並び・位置・時刻行を既存の forward に反映し、生成対象の行だけを出力ヘッドに渡す。
9. **プロンプト拡張は移植しない**（Astra の推奨。9.7 GB のロードと最大 2048〜3072 トークンの逐次生成が丸ごと不要になる）。ユーザーのプロンプトをそのまま使い、ワークフローの拡張 off と同じ条件にする。GGUF 化して llama.cpp で動かす案と C++ 移植案は見送る。
10. **旧経路の削除** — ComfyUI を起動せずに T2I と編集の画像が出てから行う。

移植しないもの: プロンプト拡張（Qwen3.5 系 LLM）、画像版 PDD サンプラー、負条件、LoRA（現状のワークフローにない）、ControlNet、P-H3 以外の画像モデル（MiniMax-H3-Image）。

## 進捗（2026-11-19）

T2I（プロンプト拡張なし）は手順 1〜7 まで完了し、DSH の左サイドバー「P-H3 画像」から使える。編集（手順 8）と旧経路の削除（手順 10）は未着手。

- 基準: `.runtime/h3-native/ph3-reference-1024`（1024×1024、seed 42、`capture_reference.py`）。照合は `native/test_ph3_parity.py`。
- **int8 の線形層**: ComfyUI の DiT は W8A8（活性を行ごとに動的 int8 化、convrot の回転と SwiGLU を量子化カーネルに融合、CUTLASS の int8 GEMM で dequant を融合）。comfy-kitchen の `int8_linear.cu` と `cutlass_gemm_int8.cu` を vendor し、CUTLASS（comfy-kitchen が固定する d4b4b49、ヘッダーのみ）を `.runtime/h3-native/cutlass` に取得してビルドした。`dsh_kitchen.int8_linear` は M > 2 の融合経路だけを持つ。
- **DiT**: `dit.cpp` の 4 投影を重みの dtype で切り替えるだけで共用。もう 1 点、ComfyUI はファイルの FP32 の Linear 重み（AdaLN、patch 埋め込み、出力ヘッド）を読み込み時に BF16 に丸めてから FP32 で計算し、バッファ（`adaln_t_table`、`rope.inv_freq`）は FP32 のまま使う。同じ丸めを `ph3.load_dit()` で行い、ブロック 0、step 0・1・7 の映像・音声出力がビット一致。
- **テキスト条件**: `te.cpp` を形状を重みから読むように一般化（H3 の 32B と共用）。int8 の埋め込みは BF16 に戻す（ComfyUI と同じ）。5 層の hidden は相対 7e-7、text_connector の出力は 6e-8（H3 と同じ FP32 の計算順の差）。captured の connector 出力からの DiT context はビット一致。通しでは H3 と同じく condition_proj の BF16 化で 1 ulp ずれ、context が 0.5% 違う。
- **サンプラー**: Euler と手動 sigma 9 値、画像の 2 形状を直接作る（Astra の指摘 3）。noise、step 7 の DiT 入力、最終 latent がビット一致。
- **VAE**: ComfyUI は投影も注意も int8（注意は sage 系の専用カーネル）。1 枚に 1 回だけなので、int8 の重みを読み込み時に FP16 へ戻し、H3 の `vae.py`（ViT とタイル）をそのまま使った。同じ最終 latent から PSNR 60.5 dB（見た目は同じ）。
- 速度（GPU が空いた状態、1 枚あたり。ネイティブはジョブごとに読み込み約 12 s）:

| 大きさ | ComfyUI（2 回目以降、保存込み） | ネイティブ（保存込み） |
|---|---:|---:|
| 1024×1024 | 約 2.3 s | 1.1–1.4 s |
| 2912×2912 | 約 11 s（DiT 6.7 s、VAE 1.6 s、保存 1.1 s） | 9.9–10.2 s |
