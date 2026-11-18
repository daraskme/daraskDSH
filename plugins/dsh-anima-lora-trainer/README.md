# dsh-anima-lora-trainer

Anima Base v1.0 の LoRA を C++/LibTorch + CUDA（前処理は Rust）でネイティブ学習し、画像もネイティブで生成する DSH プラグイン。

- `index.mjs` / `anima-tool.mjs`: チャット用ツール `anima_lora_train`
- `native/`: 学習エンジン（`model.cpp`, `engine.cpp`, `optimizer.cu`）、Rust 前処理（`standalone/`）、参照バックエンド（`train.py`）、パリティテスト（`tests/`）
- サイドバー: 「Anima LoRA」（Gemma でキャプション → 学習）と「Anima 生成」（テキストから画像、LoRA 対応）。キューは共通で、GPU は順番に使う（`jobs.mjs`, `client.js`）
- `native/generate/`: 生成（ComfyUI とビット一致）。`text.cpp`（Qwen3 + LLM adapter）、`dit.cpp`、`vae.cpp`、`sampler.py`（er_sde・CFG）、`lora.py`、`tokens.py`、ジョブ実行の `generate.py`（ComfyUI の venv の torch で動かす）。画像は `.runtime/anima-native/generated/<ジョブ>/`
- CLI: `dsh anima build|prepare|train`（ルートの `dsh` がこのフォルダーを呼ぶ）
- ビルド成果物・キャッシュは `.runtime/anima-native/` に置く

計画: [docs/anima-native-plan-v2.md](../../docs/anima-native-plan-v2.md)、生成は [docs/anima-generate-native-plan.md](../../docs/anima-generate-native-plan.md)
