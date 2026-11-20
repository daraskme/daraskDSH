# Anima LoRA ネイティブ高速学習 全体計画 v2（Astra レビュー反映済み）

方針: **速度 ≥ 品質**。最小の変更で最大の時間短縮を取る。新しい抽象・汎用化・設定項目は、測定で必要と示されるまで追加しない。

関連: [v1 計画](anima-native-plan.md)・[Opus 5.5 レビュー](anima-native-opus-review.md)・[Astra 6.2 敵対的レビュー](anima-native-astra-review.md)

## 置き場所

DSH プラグイン `plugins/lora-training/` に集約する（C++/CUDA・Rust・Python 補助・`anima_lora_train` ツール）。`dsh anima build|prepare|train` はこのフォルダーを呼ぶ。

## 現状（2026-11-19 時点、実測）

- 実装済み: C++/LibTorch の DiT 学習ループ、Optimi 互換の融合 AdamW + クリップ CUDA カーネル、Rust の画像・トークナイザー・safetensors 前処理、`dsh anima build|prepare|train`、eager/compile 参照バックエンド、パリティテスト。
- 実測（RTX PRO 6000 Blackwell, 512px, batch 1, rank 8–16）: **約 78 ms/step**、ピーク VRAM 約 9 GB、前処理 3.6 s（キャッシュ時 6 ms）。
- 未計測: step の内訳（GPU 空白時間）、diffusion-pipe 現行構成の同条件 step 時間。
- 理論下限の概算（約 45 ms）は目安にとどめ、合否には使わない。

## 言語の結論

**C++/CUDA（+ 既存の Rust 前処理）を継続する。** 学習時間の大半は GPU カーネル（GEMM・attention）で決まり、言語の差はホスト側の起動コストだけ。他言語に移しても同じ cuBLAS/cuDNN を呼ぶので速くならない。Triton は融合カーネルが必要になった場合だけ AOT で cubin 化して使い、ランタイム依存にはしない。

## 速度の打ち手（順番どおりに、1 段ずつ測って採否）

1. **内訳の計測** — ウォームアップ後の連続 step で GPU 空白時間と壁時計を同じ区間で取る（`nsys`）。統計回収・ログも区間に含める。diffusion-pipe 現行構成の同条件 step 時間を 1 回だけ測る。
2. **無駄な cast の除去（最安）** — `Projections::linear` が W 用と LoRA A 用に別々に `x.to(dtype)` している。cast を 1 回にして共有し、self Q/K/V は分岐前に 1 回だけ cast する。既存の勾配パリティで確認する。
3. **batch>1（実施済み: ≤約 572px で B=2・lr×2 を自動適用）** — eager で B=1/2/4 を測る。B=1 の ms/step と B>1 の ms/枚を別に記録する。既定値は「同じ処理枚数で品質ゲートを通った構成の完走時間」で決める（画像/秒だけでは決めない）。
4. **凍結 QKV / cross KV の連結（試して不採用: 速度差なし、BF16 の勾配ずれが増加）** — 投影が実測で重い場合だけ、ロード時に凍結 self QKV と cross KV を連結して GEMM 数を減らす（28 block で 84 呼び出し減）。LoRA A/B は独立のまま（A を W に連結すると不要な W 勾配が出るのでやらない）。
5. **CUDA Graphs（手順 1 で空白時間が大きい場合のみ）** — capture は forward + backward + 固定バッファへの grad 回収・norm まで。AdamW と統計書き込みは graph の外に残し、実際の `step + 1` と `slot` を毎回渡す。graph の単位は完全一致の `(B,H,W,L)` で、頻出形状だけ capture して残りは eager。入力は固定バッファへコピーする。context の padding はしない（cross-attention に mask がないため結果が変わる）。
6. **attention の比較（実施済み: 自動選択の flash が最速のため変更なし）** — self/cross を分け、実形状で forward + backward の合計を自動選択と各バックエンドで比較する。速くて両方完走した箇所だけ固定する。
7. **FP8 凍結 GEMM（検証して不採用: 量子化込みで約 4%、GEMM 1 回の誤差 3.8%）** — 最も重い実形状 1 個で、cuBLASLt (SM 12.0) が受理する scale と配置を forward と dX の両方向で確認する。per-tensor scale、量子化コスト込みで BF16 と比較し、勝たなければ打ち切る。勝った場合も FP8 は凍結 GEMM の内部に閉じ、LoRA 分岐と出力は BF16 のまま（現行の `dtype = weight.scalar_type()` を FP8 化すると LoRA まで FP8 になるので不可）。

**追加: 要素ごとの演算の融合（実施済み）** — 融合 RMSNorm、`LoraLinear`（LoRA 項を addmm に統合）、融合 RoPE（`rope.cu`）、LayerNorm の affine への変調統合。1024px は 248.5 → 209.0 ms/step。精度は `tests/bf16_accuracy.py`（FP32 に対して参照 BF16 以上）で判定する。

やらないこと: 汎用モデル対応、マルチ GPU、手書き attention カーネル、backend 抽象、NVFP4、層単位の FP8 フォールバック機構、LoRA A の W 連結、torch.export/AOTInductor。

## 正しさと品質のゲート（最小限）

- **1-step パリティ（既存テスト拡張）**: 参照が書き出した latent/条件/noise/timestep と、A・B とも非ゼロの LoRA。予測と loss の相対誤差、全層 LoRA 勾配のコサイン ≥ 0.999 **と勾配ノルム比**（参照比 ±5% 目安）。
- **学習曲線**: 固定シードで 200 step、参照との loss 移動平均の差が許容内。
- **ComfyUI 適用**: 「lora key not loaded」が 0 件で、448 層すべてに適用されること。最終採否は、学習した LoRA を配布時の BF16 base に載せ、固定シードで参照 LoRA と出力を比べて決める（FP8 経路も同じ）。
- 出力は一時ファイル → fsync → rename（既存の `_publish_new` を維持）。

## 計測ルール

- ウォームアップ後 100 step 以上、CUDA イベントで中央値と p95、ms/枚、ピーク VRAM。512 と 1024 の 2 条件で、各 3 回。
- ユーザー指標として「フォルダー指定から LoRA 完成までの実時間」（初回・キャッシュあり）を併記。
- 結果は `docs/anima-native-benchmarks.md` に 1 表で記録する。

## 完了条件

- 512px・同一処理枚数で、完走時間が現状（78 ms/step 相当）から **35% 以上短縮**、1024px でも短縮していること。
- 品質ゲートを通過し、`dsh anima train` の既定がその最速構成になっていること。
