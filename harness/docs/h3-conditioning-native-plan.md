# H3 の I2V・ref2video と P-H3 の編集・ref 生成のネイティブ化、チャットからの生成

方針: **速度 ≥ 品質**。既存のネイティブ実行系（`plugins/media-shared/native/`、T2V と T2I はビット一致済み）に、画像を条件に入れる部品だけを足す。新しい抽象化は作らない。仕様は ComfyUI v0.45.0 の `comfy_extras/nodes_minimax_h3.py`、`comfy/model_base.py`（MiniMaxH3）、`comfy/ldm/minimax/{model,vae}.py`、`comfy/text_encoders/{minimax,qwen3vl,qwen35}.py`。

## 対象（2026-11-19 時点）

| 機能 | ComfyUI のノード | 入力 |
|---|---|---|
| H3 I2V | `MiniMaxH3ImageToVideo`（first_frame、任意で last_frame） | 先頭（と末尾）フレームの画像 + プロンプト |
| H3 ref2video | `MiniMaxH3ReferenceToVideo`（ref_images のみ、ref_image_size=match） | 参照画像 1〜9 枚 + プロンプト（`<Picture i>` で参照） |
| P-H3 編集 | `PH3ImageConditioning`（image_1） | 編集する画像 1 枚 + 指示（先頭に `Edit <Picture 1>. ` を付ける） |
| P-H3 ref つき編集 | 同（image_1 + image_2..） | 編集する画像 + 参照画像 |
| P-H3 ref 生成 | 同（prompt が `subject_definitions:` で始まる＝verbatim。大きさは width/height） | 参照画像 + プロンプト |
| チャット | 新規（DSH ツール） | 登録済みの LLM が上の全部と T2I・T2V・Anima を呼べる |
| リモート | studio.darask.me（Cloudflare Access → `deploy/gateway.mjs` → 同じ Web UI） | 上の全部と Anima の LoRA つき生成・LoRA 学習をスマホのブラウザから使える |

移植しないもの: ref の動画・音声（`ref_videos`、`ref_audios`）、`MiniMaxH3AddGuide`、プロンプト拡張、ref_image_size=max（必要になったら足す）。

## 必要な部品（チェックポイントで確認済み）

- **Qwen3-VL の vision 塔**: H3 の 32B（hidden 1152、27 層、deepstack 8/16/24）と P-H3 の 4B（1024、24 層、deepstack 5/11/17）。どちらも同じファイルに `visual.*` がある（int8 convrot）。Qwen3.5 vision（patch 16、temporal 2、merge 2、学習済み位置埋め込み 2304 の補間、2D RoPE）+ merger + deepstack merger。前処理は `process_qwen2vl_images`（smart_resize、mean/std 0.5）。
- **LLM 側**: 画像トークンの埋め込みを vision 出力で置き換え、mRoPE（3 軸）の position_ids、deepstack を先頭 3 層の出力の画像位置に加算。`te.cpp` の `encode_text_layers` に「入力埋め込み・position_ids [3, T]・deepstack（位置マスクと 3 本の特徴）」を任意引数で足す。token_tags は vision ブロック（前後の vision_start/end を含む）を 0、それ以外 1。
- **トークン列**: H3 I2V は `<Picture 1>: ` + vision ブロック（末尾フレームがあれば `<Picture 2>: ` + ブロック）+ プロンプト。ref2video も同じ形（`<Picture i>: `）。P-H3 は `qwen3vl` のチャットテンプレート側（`PH3ImageTokenizer`）なので、基準の token ID で確認する。
- **VAE エンコーダー**: H3 と P-H3 で同じ構造（`EncoderFCN3D` 3D CNN、`quant_conv`）。1 フレームは `_adaptive_encode`（タイルあり）→ 最終フレーム。H3 I2V/ref は mean のみ、P-H3（`H3ImageVAE.encode`）は固定 seed の noise で mean + std × noise。どちらも latent を正規化して返す。
- **DiT**: `PackedLayout` の一般化。並びは text | keyframe（cond）| ref_img … | audio | video。cond・ref の行は timestep `max(t_v, 0.999)`、tag 0、位置は keyframe が target の時間原点 + 5/3 × frame_index、ref は text の後ろから 1 枚ごとに時間 +1（target の時間原点も ref の分だけ後ろへ）。cond 行は `video_patch_proj` に通す前に `0.999 × r + 0.001 × noise`（CPU、`manual_seed(seed)`、条件ごとに同じ seed で再開）。出力ヘッドは target の audio・video 行だけ。SLA の sink は「video より前の全ブロック」なので式はそのまま。
- **前処理のリサイズ**: ComfyUI の `common_upscale(lanczos)` は 8 bit を往復する PIL の lanczos。同じ関数をそのまま使う（ComfyUI のコードは実行時に使わないので PIL で同じ手順を書く）。

## リモート対応（ユーザーの要望）

- 入力画像はサーバー上のパスではなく**ブラウザからアップロード**する（POST の JSON に base64、1 枚ずつ `upload` 操作で受け取り、`.runtime` 下のアップロード置き場に保存してトークンを返す）。ローカルでもリモートでも同じ経路。チャットツールはワークスペースのパスを使う。
- LoRA 学習の画像フォルダーも、サーバーのフォルダー選択（今の `list`）に加えて、アップロード（画像と同名の .txt、複数ファイル）で新しいフォルダーを作れるようにする。デスクトップの `pickWorkspace` はリモートでは出さない。
- パネルはスマホ幅（375 px）で崩れないことを確認する。結果の画像・動画は既存の GET（`?image=`）で返す（gateway を通る）。
- gateway は JSON の POST をそのまま通すので、アップロードのサイズ上限だけ決める（1 枚 20 MB 程度）。

## 進め方（1 段ずつ ComfyUI と照合）

1. **基準を採る**（`capture_reference.py`）: H3 I2V（先頭のみ、640×384×39）、H3 ref2video（参照 2 枚、非正方形）、P-H3 編集（1 枚）、ref つき編集（2 枚）、ref 生成（2 枚）。各リサイズ後の画像、token ID・token_tags、vision 出力と deepstack、Qwen の出力、VAE エンコードの latent、DiT の payload（layout の位置・segments）、step 0 の DiT 入出力、最終 latent を保存する。
2. **VAE エンコーダー**（`vae.py` に `encode_image`）: 同じ画素から latent をビット一致（または 1e-6 程度）で確認。
3. **vision 塔**（Python、`vision.py`）: 1 回きりなので PyTorch で書く。同じ画素から merged と deepstack を照合。
4. **LLM の画像入力**（`te.cpp`）: 置換・mRoPE・deepstack。captured の vision 出力から Qwen の出力（H3 は第 50 層、P-H3 は 5 層）を照合、続けて DiT 直前の context と token_tags。
5. **DiT**（`dit.cpp`）: 位置と segments を Python で作って渡す（`PackedLayout` を `sampler.py` に移す）。cond 行の FP32 rows を受け取って前に詰める。captured の入力から step 0 の出力をビット一致で確認し、通しで最終 latent。
6. **ジョブとパネル**: `generate.py`・`ph3.py` に画像の入力（request の `images`: パスとモード）を足す。パネルに画像の選択（ドラッグ＆ドロップ）とモード（H3: T2V / I2V / ref、P-H3: 生成 / 編集 / ref つき編集 / ref 生成）。
7. **チャットツール**: `dsh-anima-lora-trainer` のキュー（`jobs.mjs`）にそのまま積むツールを登録する（`image_generate`（Anima・P-H3）と `video_generate`（H3））。画像はワークスペースのパス。完了まで待ち、出力ファイルのパスを返す。
8. **確認**: DSH から各モードを 1 回ずつ（ローカルとスマホ幅の両方）。速度を ComfyUI と比べる。旧ワークフローの削除はユーザーの確認後。

## レビューの反映（[h3-conditioning-astra-review.md](h3-conditioning-astra-review.md)）

- H3 は **Eros beta7 を先に試す**。I2V・ref が効かないモードだけ専用の DiT（I2V は `fl2va` + fl2v turbo LoRA）に切り替える。
- P-H3 の ref 生成は、専用の文法を作らない。編集と同じ経路で、プロンプトに `Edit <Picture 1>. ` を付けない（verbatim）プリセットとして出し、参照への追従を見てから公開する。
- vision 塔は BF16（H3 は `visual.*`、P-H3 は `model.visual.*`）。両モデルとも `<Picture n>: ` + vision ブロック + 本文。基準では各画像の開始位置・長さ・grid と mRoPE の position_ids も採る（異なる寸法の 2 枚）。
- 基準に、縦横の違う先頭・末尾フレームの I2V（末尾は中央 crop、frame index は実フレーム数 − 1）を 1 件足す。
- `dit.cpp` は segments から audio・video の開始位置を取る（SLA の KV sink は video 開始まで、dense query は audio 開始から video 開始まで）。step 0・1 で照合する。
- 合格基準: 同じ入力を入れた DiT は厳密一致、通しは固定 seed で参照への追従と、画像なしの経路が変わらないこと。
- チャットは既存キューに積み、自分のジョブ ID だけを期限付きで待つ。
- アップロードは 1 ファイルずつ、サイズと画素数に上限を付ける。LoRA 学習データのアップロードは同名の .txt と組にし、上書きしない。動画の GET は Range（206）に対応する。

## 未確定（レビューで決めたい）

- Eros beta7（今の T2V の DiT）で I2V・ref が効くか。効かなければ `minimax_h3_fl2va_pruned_int8_convrot` + fl2v turbo LoRA（blueprint の構成）に切り替える。
- P-H3 の ref 生成のプロンプト形式（`subject_definitions:` の書き方は拡張器の出力でしか分からない）。

## 進捗（2026-11-19）

手順 1〜7 まで完了。H3 の I2V（最初・最後のフレーム）と ref2video（参照画像）、P-H3 の編集・ref つき編集・ref 生成が、サイドバーのパネル（スマホ幅可）とチャットツールから使える。旧ワークフローの削除はまだ。

- **基準**: `.runtime/h3-native/cond-h3-i2v`（640×384×38→39、先頭は縦長の絵の引き伸ばし、末尾は別の縦長の絵の中央切り抜き）、`cond-h3-ref`（参照 2 枚、match）、`cond-ph3-edit`、`cond-ph3-editref`。`capture_reference.py` に vision 塔・VAE エンコード・mRoPE 位置・packed layout のフックを足した。Eros beta7 のままで I2V も ref も効いた（専用 DiT への切り替えは不要）。
- **読み込みとリサイズ**（`native/pictures.py`）: ComfyUI の LoadImage は JPEG を PIL ではなく PyAV で読む（PIL とは最大 0.1 違う）。同じく PyAV で読み、lanczos は PIL の 8 bit 往復で書いた。読み込み・引き伸ばし・中央切り抜き・match・P-H3 の vision 用と VAE 用の大きさが全部ビット一致。
- **VAE エンコーダー**（`vae.py` の `encode_image`）: 1 フレームなので因果 3D 畳み込みは 2D 畳み込み（最後の時間タップ）。ComfyUI は comfy-kitchen の融合 GroupNorm＋パッドと channels_last の cuDNN を使い、こちらは素の PyTorch。latent は相対 6e-4〜1.7e-3（kitchen の融合カーネルを使っても 4e-4〜1.1e-3 で、cuDNN の FP16 演算順の差が残るため、依存の少ない素の版にした）。P-H3 は固定 seed 42 の事後サンプル。
- **vision 塔**（`native/vision.py`、Python、FP32）: 位置埋め込みの補間は埋め込みの dtype（BF16）で計算する（FP32 にすると 1〜2% ずれる）。merged・deepstack とも 1e-6。
- **LLM**（`te.cpp`）: `embed_tokens` と `encode_embeds`（画像行を差し込んだ埋め込み、interleaved mRoPE 24/20/20、最初の 3 層の後に deepstack を加算）。トークン列・token_tags は完全一致、H3 の第 50 層は 4e-6〜8e-6（T2V と同じ水準）。P-H3 の connector 出力は 5e-7、DiT の context は T2I と同じ 0.6%（condition_proj の BF16 丸め）。
- **DiT**（`dit.cpp`・`sampler.conditioning`）: 条件行（keyframe・ref）は FP32 のパッチ行に 0.999 / 0.001 の seed ノイズを混ぜて渡し、target 行と 1 回の GEMM で埋め込む。位置表は Python で作る（`PackedLayout` の移植）。条件行は時刻 `max(t_v, 0.999)`・タグ 0 の 1 区間。SLA の sink は video の開始位置から自動で決まる。4 基準すべてで位置表と step 0・1 の映像・音声出力がビット一致（H3 は SLA と VBVR LoRA あり）。
- **通し**: P-H3 の ref つき編集・H3 の I2V とも ComfyUI とほぼ同じ絵（条件の 0.5% 差による細部の違いだけ）。P-H3 の ref 生成（Edit を付けない verbatim）は、人物・服装を引き継いで新しい構図を作った。
- **速度**（2 回目以降、GPU が空いた状態）: P-H3 の編集 1 枚 3.7 秒（864×1216、読み込み約 15 秒）。H3 の I2V・ref2video は 640×384×39 でサンプリング 4〜5 秒、ジョブ全体 40 秒前後（読み込み込み）。

### DSH への組み込み

- パネル（`dsh-anima-lora-trainer/client.js`）: H3 動画は「テキストから / 画像から（最初・最後のフレーム）/ 参照画像から」、P-H3 画像は「テキストから / 画像を編集 / 参照画像から生成」。画像はファイル選択かドラッグ＆ドロップで 1 枚ずつサーバーへ送る（`upload`、20 MB まで、`.runtime/anima-native/uploads`、14 日で削除）。ジョブのカードに入力画像を表示。
- リモート（studio.darask.me）: ローカルと同じ経路で動く。動画は Range（206）で返すので、スマホでも再生・シークできる。LoRA 学習は、端末の画像と同じ名前の .txt を送って新しいフォルダーを作れる（`dataset`、上書きしない。キャプションを送ったときは作り直しを既定で切る）。「Ubuntu で選択」はローカルでだけ表示。
- チャット（`media-tool.mjs`）: `image_generate`（Anima / P-H3 の生成・編集・参照生成）と `video_generate`（H3 の T2V / I2V / ref）。ワークスペース内の画像だけを受け付け、パネルと同じキューに積んで自分のジョブだけを待ち、結果をワークスペースの `generated/<ジョブ>/` にコピーして返す。止められたらジョブも取り消す。登録済みのどの LLM からも使える。
