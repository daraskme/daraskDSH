# dsh-anima-lora-trainer

Anima Base v1.0 の LoRA を C++/LibTorch + CUDA（前処理は Rust）でネイティブ学習し、画像もネイティブで生成する DSH プラグイン。

- `index.mjs` / `anima-tool.mjs`: チャット用ツール `anima_lora_train`
- `media-tool.mjs`: チャット用ツール `image_generate`（Anima・P-H3 の生成・編集・参照生成）と `video_generate`（H3 の T2V・I2V・参照）。パネルと同じキューに積み、結果をワークスペースの `generated/` にコピーする
- `native/`: 学習エンジン（`model.cpp`, `engine.cpp`, `optimizer.cu`）、Rust 前処理（`standalone/`）、参照バックエンド（`train.py`）、パリティテスト（`tests/`）
- サイドバー: 「Anima LoRA」（Gemma でキャプション → 学習）と「Anima 生成」（テキストから画像、LoRA 対応）。キューは共通で、GPU は順番に使う（`jobs.mjs`, `client.js`）
- `native/generate/` の編集（img2img・範囲の描き直し）: リクエストに `image`（と `mask`、白が描き直す所）・`denoise` を付けると、画像を VAE でエンコード（`vae.cpp` の `vae_encode`、ComfyUI の VAEEncode とビット一致、`tests/generate_encode_parity.py`）し、KSampler の denoise と同じ途中の sigma から描く（`sampler.sample_image`）。マスクがあれば ComfyUI の SetLatentNoiseMask と同じく毎ステップ外側を元の latent に戻し、最後にぼかしたマスクの外へ元の画素を貼り戻す。幅・高さ 0 で元の縦横比のまま約 1.2 MP。マンガタブの「Anima で直す」（画風をそろえる・範囲を描き直す）が使う
- `native/generate/`: 生成（ComfyUI とビット一致）。`text.cpp`（Qwen3 + LLM adapter）、`dit.cpp`、`vae.cpp`、`sampler.py`（er_sde・CFG）、`lora.py`、`tokens.py`、ジョブ実行の `generate.py`（ComfyUI の venv の torch で動かす）。画像は `.runtime/anima-native/generated/<ジョブ>/`
- CLI: `dsh anima build|prepare|train`（ルートの `dsh` がこのフォルダーを呼ぶ）
- ビルド成果物・キャッシュは `.runtime/anima-native/` に置く

計画: [docs/anima-native-plan-v2.md](../../docs/anima-native-plan-v2.md)、生成は [docs/anima-generate-native-plan.md](../../docs/anima-generate-native-plan.md)

## LoRA 学習パネル

左サイドバーの「LoRA 学習」で、Anima・MiniMax H3・P-H3 の LoRA を作る（Gemma でキャプション → ネイティブ学習）。H3 / P-H3 は `plugins/dsh-minimax-h3/native/train_lora.py`（diffusion-pipe と同じ学習式、[docs/h3-lora-native-plan.md](../../docs/h3-lora-native-plan.md)）。H3 は動画（音声つき）も素材にできる。学習した LoRA は各生成パネルとチャットのツールで選べる。

## H3 動画パネル

左サイドバーの「H3 動画」から、MiniMax H3（Eros beta7 + VBVR Pro V2）の音声つき動画を作る。実行系は `plugins/dsh-minimax-h3/native/generate.py`（ComfyUI のコードを使わないネイティブ実装、[docs/h3-native-plan.md](../../docs/h3-native-plan.md)）。ジョブは学習・Anima 生成と同じキューに入る。長さは最大 60 秒（1440 フレーム）で、362 フレームを超えると約 15 秒ずつの区間を繋いで作る（[docs/h3-long-video-native-plan.md](../../docs/h3-long-video-native-plan.md)）。

## P-H3 画像パネル

左サイドバーの「P-H3 画像」から、p-h3-image（int8）で画像を作る（テキストから画像のみ。プロンプト拡張は使わない）。実行系は `plugins/dsh-minimax-h3/native/ph3.py`（[docs/ph3-image-native-plan.md](../../docs/ph3-image-native-plan.md)）。ジョブは同じキューに入る。
