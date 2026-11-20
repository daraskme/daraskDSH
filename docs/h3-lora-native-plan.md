# MiniMax H3 / P-H3 LoRA 学習のネイティブ化計画

方針: 速度 ≥ 品質。既存の学習方法（diffusion-pipe の `models/minimax_h3.py`。ComfyUI のコードを import し、DeepSpeed のパイプライン、block swap、`unsloth` の CPU 退避チェックポイントを使う）と同じ数式で学習し、遅い仕組みだけを外す。新しい抽象や設定項目は増やさない。Anima の学習と共通にできる部分は共通にする。

## 現状

- H3 の LoRA は diffusion-pipe で学習していた（FolderLoRA ノード経由。2026-11-17 の検証では rank 2・256px・2 step で 107 秒）。ノードのソースはもうない。diffusion-pipe 自体は `.runtime/diffusion-pipe` にある（今の lora-venv では DeepSpeed が `CUDA_HOME` を要求して起動しない）。
- P-H3 の LoRA 学習はない。diffusion-pipe の minimax_h3 は int8 convrot の DiT を読めるので、P-H3 の重みを指せば動く可能性がある（基準の計測で確かめる）。
- 生成側: H3 は Eros beta7（BF16、40 GB）に VBVR Pro V2 を `merge_loras` で合成。P-H3 は int8 convrot（21 GB）をそのまま W8A8 で使う。どちらもユーザーの LoRA は載せられない。

## 学習の数式（diffusion-pipe と同じ）

- flow matching: x_t = (1 − t)·x0 + t·noise、目標は noise − x0、出力は DiT 出力の符号反転。t は uniform、shift は動画 8・画像 1。音声の t は `time_shift_sigma(t, 10, 3)`。
- loss: 映像と音声の MSE を、トークン数の比で重み付けして足す（音声なしは映像のみ）。
- CFG 拡張（cfg = 4）: 同じ層で空プロンプトの uncond を勾配なしで通し、出力を (cond + 3·uncond) / 4 にしてから loss を取る。蒸留済み（8 step、CFG なし）の H3 と P-H3 の両方に使う。
- LoRA: 50 ブロック + token_refiner 2 ブロックの qkv_proj / out_proj / mlp.fc1 / mlp.fc2。既定は rank 32、AdamW（lr 1e-4、betas 0.9/0.99、wd 0.01）、勾配クリップ 1、warmup 10。保存形式は diffusion-pipe と同じ `diffusion_model.<module>.lora_A/B.weight`（ComfyUI でもそのまま読める）。
- データ: 画像は 768px の面積で、アスペクト比のバケット（Anima と同じ 9 段）。動画は 512px で先頭から切り出し（single_beginning）、フレーム数は ComfyUI の VAE エンコーダーが受け付ける長さのバケット。音声は 32 kHz ステレオ。

## 学習の対象モデル

- H3: 生成と同じ Eros beta7 BF16 に VBVR Pro V2 を合成した重み。学習した LoRA は、生成時にその上に載せる。
- P-H3: int8 convrot の重みのまま。順伝播は推論と同じ `int8_linear`（W8A8）。逆伝播の dX は、層ごとにその場で kitchen の `dequantize_int8_convrot` で BF16 に戻した W で計算し、すぐ捨てる（活性の量子化は straight-through）。BF16 の常駐コピーは持たない。

## 速さの打ち手（既存との差）

1. 重みをすべて GPU に常駐させる（H3 40 GB、P-H3 21 GB、96 GB 中）。block swap も DeepSpeed のパイプラインも使わない。
2. 活性チェックポイントは必要なブロック数だけ、GPU 上で取る。CPU への退避はしない。画像（P-H3 1024px で約 1.3k トークン）はチェックポイントなしで収まる見込み。
3. attention は mask なしの flash（batch 1、パディングなし）。diffusion-pipe は mask を渡すため、flash 以外のカーネルに落ちている。
4. 凍結した重みの dW は計算しない（autograd が自動で省く）。LoRA の分岐は `addmm` で本体の出力に足す。
5. latent と文の埋め込みは最初に 1 回だけ計算してディスクに置く（テキストエンコーダーと VAE は学習中は GPU に置かない）。
6. 実装は Python + PyTorch の autograd。H3 の step は数秒の GEMM 律速なので、ホスト側を C++ にしても速くならない。融合 RMSNorm + RoPE（kitchen の順伝播カーネル）などは、計測で効くと分かったものだけ autograd 関数にする。`torch.compile` はブロック単位で 1 回試し、起動時間込みで速ければ使う。
7. やらないこと: SLA のスパース attention での学習、FP8、マルチ GPU、batch > 1（画像の batch は計測後に検討）、汎用モデル対応。

## 新しく書くもの

- `plugins/media-shared/native/train/model.py`: 学習用の DiT（ブロック、token_refiner、final layer、packed layout と位置表は `sampler.py` / `dit.cpp` と同じ）。LoRA と int8 の autograd 関数。
- `train/encode.py`: 動画 VAE のエンコーダー（3D、ComfyUI の `EncoderFCN3D` と同じ）と音声 VAE のエンコーダー（`audio_vae.py` の `encode`）。今の `vae.py` のエンコーダーは 1 フレームのみ。
- `train/train.py`: キャッシュ作成 → 学習ループ → 保存。進捗は `status.json`（Anima と同じ書式）。
- 生成側: `merge_loras` が diffusion-pipe 形式のキーも読む（H3）。P-H3 は int8 に合成できないので、`dit.cpp` の `proj` で LoRA の分岐を足す。ジョブの request に `loras` を通す。

## Anima と共通にするもの

- キャプション: `pipeline.py` の Gemma キャプションをそのまま使う。動画は数フレームを渡してキャプションを付ける。
- ジョブ・パネル: 学習パネルにモデル選択（Anima / H3 / P-H3）を足し、同じキューで実行する。プリセット（`presets.mjs`）は model ごとに既定値を持つ。
- Python の小物（`_publish_new`、LoRA の保存、サンプル順、勾配クリップ、アスペクト比のバケット、flow の入力）は `plugins/lora-training/native/common.py` に移し、両方から使う。
- 共通にしないもの: Anima の C++ エンジン（モデルが違う）、Rust の前処理（動画・音声を扱わない）。

## 確認

- 順伝播: 学習用 DiT（LoRA ゼロ）と `dit_forward`（dense）の出力が BF16 の誤差内（相対 L2 ≤ 1e-2）。P-H3 は int8 経路で同じ。
- 勾配: 短い系列で、FP32 の参照（同じ Python モデルを FP32 で）に対する誤差が BF16 参照と同程度。int8 の逆伝播は dequant した BF16 の参照と比べる。
- エンコーダー: ComfyUI のエンコードとビット一致、または BF16/FP16 の誤差内。
- 速さ: 同じデータ・同じ設定で diffusion-pipe（以前の構成）を 1 回動かし、s/step と完走時間を比べる。H3 は画像 768px と動画 512px × 約 5 秒、P-H3 は 1024px。
- 品質: 同じ step 数で学習した LoRA を生成に載せ、固定シードで diffusion-pipe の LoRA と見比べる。loss 曲線の移動平均も比べる。

## レビューの反映と結果（2026-11-19）

Astra のレビュー（`docs/h3-lora-astra-review.md`）を反映した点:

- H3 の音声の時刻は、モデル側の既定 `time_shift_sigma(t, 12, 3)`（diffusion-pipe と同じ。10/3 は生成時の設定）。P-H3 は生成と同じ 1/1。
- LoRA は refiner 2 ブロックを含む 208 投影。テキストのキャッシュは refiner に入る前の 5120 次元にして、refiner は学習中に毎 step 通す。
- uncond（空プロンプト）は diffusion-pipe と同じく、1 トークンのテキストをキーから外した形（位置は 1 つずれる）。cond はパディングがないので mask なしの flash。
- 新しいファイルは `native/` に置いた（`train/` は作らない）: `lora_model.py`（学習用 DiT）、`train_lora.py`（キャッシュ・学習・保存）、`encode.py`（動画 3D・音声エンコーダー）、`loras.py`（生成で LoRA を載せる）。
- P-H3 の `fc2` は推論と同じ融合 SwiGLU の int8 カーネルで順伝播し、逆伝播は BF16 に戻した重みと PyTorch の SwiGLU 逆伝播。

実装:

- 生成: `merge_loras` が diffusion-pipe / PEFT 形式（`diffusion_model.*.lora_A/B`）も読む。P-H3 の int8 ブロックは合成できないので、`loras.py` が `<層>.lora_a/b` を付け、`dit.cpp` の `proj` が実行時に足す（refiner は BF16 なので合成）。H3 は VBVR → ユーザーの LoRA の順に合成。
- パネル: 「LoRA 学習」にモデル選択（Anima / MiniMax H3 / P-H3）。H3 は動画（先頭の最大 136 フレームと音声）も素材にでき、Gemma は動画の真ん中のフレームでキャプションを付ける。H3 動画・P-H3 画像・Anima 生成の各パネルは共通の LoRA 選択欄を使い、一覧はキーと `dsh_model` メタデータでモデルごとに分ける。チャットの `image_generate` / `video_generate` も LoRA を受け付ける。
- Anima と共通: `plugins/lora-training/native/common.py`（サンプル探索、アスペクト比バケット、画像の読み込み、サンプル順、勾配クリップ、原子的な保存）。ジョブのキュー・`pipeline.py`（キャプション → 学習）・進捗の書式も共通。

確認（`test_lora_model.py h3|ph3`、`test_encode_av_parity.py`）:

| 項目 | H3 | P-H3 |
|---|---|---|
| refiner と `refine_text` | ビット一致 | ビット一致 |
| 1 ブロックと `dit_block`（同じ入力） | 0.01〜0.07% | 0.04〜0.5%（W8A8 の量子化が 1 ulp の差を広げる） |
| 50 ブロック通し（ランダム入力） | 3〜5% | 3〜13% |
| LoRA（FP32 で合成と比較） | 3e-7 | — |
| LoRA 込みのブロックと実行時 LoRA の `dit_block` | — | 0.08〜0.6%（LoRA の効果は 3〜26%） |
| LoRA 勾配（H3: FP32 比、P-H3: BF16 に戻した重み比） | コサイン 0.99998、ノルム比 0.9% | コサイン 0.99996、ノルム比 1.2% |

- 50 ブロック通しの差は、推論版自身が入力の 0.1% の揺らぎで約 10% 動く（ランダム入力での感度）のと同じ大きさで、実装の差ではない。
- 動画エンコーダーは ComfyUI に対して相対誤差 8.5e-4（FP16 の丸め）、音声エンコーダーはビット一致。

速度（RTX PRO 6000、同じ 20 件 = 動画 9 本 + 画像 11 枚、rank 32、CFG 4、2 epoch = 40 step。ユーザーの ComfyUI が 16 GB 使用中）:

| | diffusion-pipe（以前の構成: swap 8・unsloth） | diffusion-pipe（swap 0・GPU チェックポイント） | ネイティブ |
|---|---:|---:|---:|
| 40 step の学習 | 781 s | 155 s（初回 compile 26 s を含む） | **88 s** |
| 768px 画像 1 step（約 600 トークン） | 13〜14 s | 0.55 s | 0.30〜0.40 s |
| 136 フレーム動画 1 step（約 9,800 トークン） | 最大約 27 s | 10.1〜10.4 s | 9.1〜9.9 s（45 ブロックをチェックポイント） |
| フォルダから LoRA まで（キャッシュなし） | 約 23 分 | — | 約 2.5 分（前処理 45 s、読み込み 23 s） |
| ピーク VRAM | — | — | 67 GB |

- P-H3（1024px 画像 11 枚、22 step）: 1 step 約 0.9 s、ピーク 37 GB。BF16 に戻した重みを常駐させても 5% しか速くならず、メモリは 56 GB に増えるので、int8 のまま学習する。
- 学習した LoRA を生成に載せると、H3 も P-H3 も効果が出て、崩れはない（同じシードで比較）。

## 次の段階（別作業）

画像生成（Anima・P-H3）、動画生成（H3）、画像編集（P-H3）の実行スクリプトの共通化（status の書き込み、重みの読み込み、LoRA の適用、出力の保存）。学習側で `jobs.mjs` と生成への LoRA 対応が入ったあとに行う。
