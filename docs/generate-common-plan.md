# 画像生成・動画生成・画像編集の共通化計画

方針: 速度 ≥ 品質、過剰設計を避ける。モデルごとの計算（Anima の `generate/`、H3 の `generate.py`、P-H3 の `ph3.py`。P-H3 は T2I・編集・参照を 1 本で扱う）は変えず、3 本に重複している「ジョブの外枠」だけを 1 つのモジュールにする。出力はビット一致のまま。

## 現状（重複）

- `update()` と `status`（`status.json` の原子的な書き込み）、`request.json` の読み込み、例外を status に書いて終了する `__main__`: 3 本とも同じコード。
- safetensors の読み込み: `generate.py` の `load(path, prefixes, cast)`、`ph3.py` の `tensors(path, keep)`、Anima の `safe_open` 直書き。
- PNG の保存とメタデータ: Anima（別スレッドで保存して次の画像と重ねる）と P-H3（同期保存、LoRA が記録されない）。
- すでに共通: キュー・ジョブ API（`jobs.mjs`）、LoRA の一覧と選択欄、チャットのツール。

## 変更

1. `plugins/dsh-anima-lora-trainer/native/job.py`（学習用の `common.py` の隣）:
   - `run(main)`: `sys.argv[1]` の JOB_DIR と request を読み、`main(job)` を呼び、例外は traceback を stderr に出して status を error にして exit 1。
   - `Job.update(**changes)`: 今の `update` と同じ書式・同じ原子的な置き換え。`job.status` / `job.request` / `job.dir`。
   - `load_tensors(path, keep=None, cast=None)`: CUDA に読む（`comfy_quant` キーは除く）。
   - `PngWriter`: 別スレッドで 1 枚ずつ保存し、保存が終わったら `status.images` に追記して `done` を更新（Anima の今の挙動）。`close()` で待つ。パラメーターは `parameters` テキストチャンク（今と同じキー）。
2. 3 本の入口をこれに置き換える。各モデル固有の読み込み・条件付け・サンプリング・デコードはそのまま。P-H3 も PNG を別スレッドで保存する（数百 ms の短縮）、メタデータに LoRA を足す。
3. やらないこと: 3 本を 1 つの入口に統合、モデルの抽象クラス、設定の共通化。

## 確認

- 既存のパリティテスト（`test_dit_parity`、`test_ph3_parity`、Anima の生成テスト）が変わらないこと。
- 実ジョブ: Anima・P-H3（T2I と編集）・H3 を 1 本ずつ走らせ、status の書式と出力が前と同じこと（同じシードで画素一致）。

## レビューの反映と結果（2026-11-19）

Astra のレビュー（`docs/generate-common-astra-review.md`）で範囲を絞った:

- 共通化はジョブの外枠だけ: `plugins/dsh-anima-lora-trainer/native/job.py` の `run(main, 初期 status)` と `Job.update` / `Job.add_image`（RLock で、保存スレッドと進捗の書き込みを直列化）。
- `load_tensors` はやめた。`generate.load` と `ph3.tensors` は学習のキャッシュ作成やテストからも呼ばれ、`comfy_quant` の扱いや dtype も違う。
- P-H3 は同期保存のまま。PNG の `parameters` に LoRA を足しただけ（名前がない request ではファイル名）。
- `python -I` で起動されるので、各入口は自分のフォルダを先頭に置いたまま、`job.py` のフォルダを後ろに足す（H3 と Anima はどちらも `build.py` を持つ）。
- 3 本を 1 つの入口にまとめることや、モデルの抽象化はしない。キュー・LoRA の選択・チャットのツールはすでに共通。

確認:

- 壊れた `request.json` で 3 本とも `python -I` から stage=error・終了コード 1。
- 以前と同じ request・シードで作り直し: Anima（FP8、2 枚）と P-H3（編集、LoRA ありの T2I）は PNG の画素が完全一致、H3（LoRA あり、73 フレーム）は映像の全フレームと音声が完全一致。
