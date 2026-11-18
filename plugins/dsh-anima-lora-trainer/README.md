# dsh-anima-lora-trainer

Anima Base v1.0 の LoRA を C++/LibTorch + CUDA（前処理は Rust）でネイティブ学習する DSH プラグイン。

- `index.mjs` / `anima-tool.mjs`: チャット用ツール `anima_lora_train`
- `native/`: 学習エンジン（`model.cpp`, `engine.cpp`, `optimizer.cu`）、Rust 前処理（`standalone/`）、参照バックエンド（`train.py`）、パリティテスト（`tests/`）
- CLI: `dsh anima build|prepare|train`（ルートの `dsh` がこのフォルダーを呼ぶ）
- ビルド成果物・キャッシュは `.runtime/anima-native/` に置く

計画: [docs/anima-native-plan-v2.md](../../docs/anima-native-plan-v2.md)
