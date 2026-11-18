# Anima Base v1.0 のネイティブ画像生成計画

方針: **速度 ≥ 品質**。ComfyUI のコードを実行時に使わない。LoRA 学習のネイティブ実装（`plugins/dsh-anima-lora-trainer`）の部品をできるだけ使い回し、新しい抽象化や設定項目は、測定で必要と分かるまで追加しない。DSH の左サイドバーに「Anima 生成」を置く。

## 現状

- 参照（ComfyUI v0.45.0）: UNETLoader（anima-base-v1.0）、CLIPLoader（qwen_3_06b_base）、VAELoader（qwen_image_vae）、EmptySD3LatentImage（832×1216）、KSampler（er_sde、simple、34 step、CFG 4.5）、VAEDecode。1 枚あたり約 7 s（2 回目以降）。
- モデル: FLOW、shift 3.0、latent_format は Wan21。テキスト条件は、Qwen3-0.6B の hidden states と T5 の token ID を、DiT の中の LLM adapter で変換したもの。
- 使い回せる部品:
  - `native/model.cpp`: Cosmos Predict2 DiT の forward（diffusion-pipe とビット一致を確認済み。LoRA に対応、BF16）
  - `export_frontend.py` が書き出す TorchScript: VAE エンコーダー、caption conditioner（Qwen3 + LLM adapter）
  - Rust の前処理: 画像の読み込み、トークナイザー、safetensors
  - DSH プラグイン: キュー、GPU の排他、サイドバーパネルの実装

## 進め方（Astra のレビューを反映）

レビュー全文: [anima-generate-astra-review.md](anima-generate-astra-review.md)

1. **基準の採取と速度の計測を分ける** — 採取は、Anima 用のフックを入れた ComfyUI で 1 件だけ行う。速度は、フックなし・モデル常駐・事前に決めた異なる 3 つのシードで、「開始 → 画像の保存」の中央値を測る。7 s が基準なら、DiT に使えるのは 1 step 約 206 ms 未満（実際は、条件の生成・VAE・保存の時間を引いた残り）。
2. **テキスト条件** — 正・負それぞれで、Qwen の ID → T5 の ID と重み → DiT が実際に受け取る `c_crossattn` の順に照合する。ComfyUI は `(red:1.5)` のような強調構文を解釈し、T5 側の重みを adapter の出力行に掛ける。512 は上限ではなく最小長（短い出力はゼロで埋める）。強調付きの短い正条件と、空の負条件で確かめる。初版で 512 を超える入力を扱わないなら、黙って切らずにエラーにする。adapter は条件ごとに生成前に 1 回だけ動かす。
3. **DiT** — 現在の `predict()` は融合後なので、diffusion-pipe ともビット一致しない。同じ latent・実際の timestep・条件を入れて、「先頭ブロックの回転後の Q/K と残差 → 1 forward の正・負の速度予測」の順に比べる。差が問題になる場合だけ、split-half の融合（ComfyUI は `rms_rope_split_half` と `addcmul`）に合わせる。
4. **サンプラー** — 仕様は `samplers.py:645–652` と `k_diffusion/sampling.py:1593–1656`。
   - `simple` は 1000 個の表から切り捨ての添字で 34 個を選び、末尾に 0 を足して 35 個の σ にする。
   - ER-SDE は FLOW の先頭 σ = 1 を補正してから λ = σ/(1−σ) を使う。
   - timestep の倍率は 1。solver が受け取るのは、`x − σv` から作った CFG 後の denoised。
   - 初期 noise は CPU の FP32、途中の noise は latent と同じデバイスの別の generator。

   参照値を注入して、第 1・2・3 の更新（履歴と 200 点の積分）と、終端の `x = denoised` を照合してから、シードからの通しで確かめる。
5. **CFG** — 実寸・同じ条件で、batch 2 と batch 1×2 の出力と合計時間を比べ、速いほうを使う。条件の長さが違うときは別の forward にする（cross-attention に埋め草を除くマスクがないため）。CFG = 1 なら負条件の forward を省く。
6. **VAE** — KSampler の出力は `process_latent_out` 済み（逆正規化済み）。solver の終端と KSampler の出口を区別して保存する。前者から FP32 で一度だけ逆正規化して後者と照合し、そのあと `qwen_image_vae` の `conv2 → decoder` と、[0,1] に変換した保存前の RGB を比べる。生成には VAE エンコーダーは使わない。
7. **LoRA** — 選んだ Anima LoRA のすべての A/B の組が対応することを確かめ、未対応のキーがあれば失敗にする。alpha のテンソルがない（学習で作った LoRA はない）場合に、余計に rank で割らない。未変更の base から、参照経路と同じ BF16 演算で一度だけマージし、常駐中の重みに次の LoRA を足し重ねない。「なし → あり → なし」で予測を比べる。
8. **速度は UI より先に測る** — BF16 で 1 枚保存できた時点で、実寸の内訳を測る。step ごとに作り直している画像の RoPE テーブルと、各ブロックで毎回投影している不変の text K/V は、生成の中で一度だけにする。CUDA Graphs は GPU に空き時間がある場合だけ、FP8 は変換込みの総時間で勝つ投影だけに使う。目標を超えたらそこでやめる。
9. **UI** — 「Anima 生成」パネル（キューと GPU の排他は LoRA 学習と共通）。チャット用ツールは初版では付けない。

## 品質の確認

- 同じシード・プロンプトで、ComfyUI の画像と並べて目視する。DiT の各 step はテンソルで比べる。
- LoRA を付けた生成でも、ComfyUI と同じように効いていること。

## 完了条件

- サイドバーから生成でき、ComfyUI と同等の画像が出る。
- 832×1216・34 step で、ComfyUI より速い（基準はフックなし・3 シードの中央値で測り直す）。

## 進捗（2026-11-19）

### 手順 1: 基準の採取（完了）

- ツール: `plugins/dsh-anima-lora-trainer/tools/capture_generate_reference.py`（H3 と同じ方式。ComfyUI を 2 つ目のインスタンスとして起動し、`execution` を読み込んだ直後にフックを入れる）。
- 基準（`.runtime/anima-native/generate-reference`）: 832×1216、er_sde、simple、34 step、CFG 4.5、seed 42。正条件は `@darask, (red scarf:1.3), ...`、負条件は空。
  - 保存したもの: token ID と重み、Qwen の hidden、`c_crossattn`（正・負）、初期 noise（CPU）、35 個の σ、er_sde の step 0・1・2・33 の x / σ / denoised、途中の noise 3 つ、DiT の呼び出し（CFG は batch 2 で `[2, 16, 1, 152, 104]`）、KSampler の出力 latent、デコード後の画像。
- 速度の基準: 計測中にユーザーの学習と生成スクリプトが同じ GPU を使っていたため、まだ確定していない。採取時の内訳（参考値）は、DiT 1 step（CFG batch 2）が 0.11–0.12 s、VAE デコードが 0.2 s。フックなしの 2 回目以降は約 10 s（GPU が混雑していた）。

### 手順 2: テキスト条件（完了・ビット一致）

- 実装: `native/generate/text.cpp`（Qwen3-0.6B と LLM adapter）。トークナイズの規則（強調構文の解析、区切りごとのトークン化、T5 の EOS の扱い、最小長 1 でのパディング）は、`tests/generate_conditioning_parity.py` に Python で移植して確かめた。Rust 版は実行系を作るときに移す。

| 照合点 | 空の負条件 | `(red scarf:1.3)` 付きの正条件 |
|---|---|---|
| Qwen の ID / T5 の ID と重み | 一致 | 一致 |
| Qwen の hidden | ビット一致 | ビット一致 |
| adapter（ComfyUI の hidden から） | ビット一致 | ビット一致 |
| `c_crossattn`（通し） | ビット一致 | ビット一致 |

分かったこと:
- **ComfyUI の Qwen3-0.6B は FP32 で動いている**（BF16 で計算すると、1 トークンの入力で 17% ずれる）。重みを FP32 に戻して計算し、adapter に渡す直前に BF16 にキャストする。
- FP32 では comfy-kitchen の融合 RoPE が使えず、eager の式（対角成分の積 → `addcmul_`）になる。2 回の乗算で書くと 5e-7 ずれる。同じ順序で計算するとビット一致した。
- adapter（BF16）は素直な実装でビット一致した。T5 の重みは adapter の出力行に掛け、512 未満ならゼロで埋める。
- 学習用の `conditioner.pt`（diffusion-pipe 由来の TorchScript）は使わない。生成では上記の C++ 実装を使う。

### 手順 3: DiT（完了・ビット一致）

- 実装: `native/generate/dit.cpp`（`dsh_anima_gen::dit(weights, x, timesteps, context)`）。ComfyUI の MiniTrainDIT と同じ演算順・dtype で書いた。
  - rope3d（時間 22 対 → 高さ 21 対 → 幅 21 対、高さと幅の外挿 4.0、fps 変調なし）の表は FP32 で作る。
  - self-attention の q/k は、vendored の comfy-kitchen `rms_rope_split_half_`（H3 と共通の `libdsh_kitchen.so`）で正規化と回転をまとめて行う。
  - timestep 埋め込み（cos → sin）→ AdaLN-LoRA（256）、`t_embedding_norm`、パディングマスクのチャネル（ゼロ）、`addcmul` の残差、最終層、unpatchify。
- 照合: `tests/generate_dit_parity.py`。基準の DiT 呼び出し（CFG batch 2、`[2, 16, 1, 152, 104]`）の x / timestep / 条件を入れて、速度予測を比べた。

| 呼び出し | σ | 正条件 | 負条件 |
|---|---|---|---|
| call0 | 0.99997 | ビット一致 | ビット一致 |
| call1 | 0.99014 | ビット一致 | ビット一致 |
| call2 | 0.97989 | ビット一致 | ビット一致 |
| 最後（step 33） | 0.08491 | ビット一致 | ビット一致 |

- DiT が受け取る timestep は σ そのもの（FLOW、倍率 1）。
- 速度の参考値: 1 forward（batch 2、832×1216）が 137 ms（ComfyUI の採取時は 110–120 ms）。この時点では最適化しておらず、GPU が完全に空いた状態でもない。text K/V と RoPE 表の使い回し、batch 2 と batch 1×2 の比較は手順 8 で行う。

### 手順 4・5: サンプラーと CFG（完了・ビット一致）

- 実装: `native/generate/sampler.py`（simple スケジューラー、FLOW shift 3、ER-SDE、CFG）。数行のスカラー演算なので Python と torch で書き、ComfyUI と同じ演算順にした。DiT は `dsh_anima_gen::dit`。
  - 空の latent は正規化しない（ComfyUI は全部ゼロなら `process_latent_in` を飛ばす）。初期 x は `σ0·noise + (1−σ0)·0`。
  - 初期 noise は CPU の FP32（`torch.manual_seed(seed)`）、途中の noise は CUDA の別の generator（同じ seed）。
  - ER-SDE は先頭 σ = 1 を `percent_to_sigma(1e-4)` に置き換え、λ = exp(logit σ)。
  - CFG は batch 2（負条件が 0 行目、正条件が 1 行目）。denoised = x − v·σ（FP32）、`u + (c − u)·4.5`。
  - 最後に Wan21 の逆正規化（`latent·std + mean`）をして、KSampler の出力と同じ latent にする。
- 照合: `tests/generate_sampler_parity.py`。17 項目すべてビット一致。
  - 35 個の σ、初期 noise、途中の noise 3 つ
  - 採取した x / σ を入れた CFG denoised（step 0・1・2・33）、採取した denoised から計算した更新 0→1・1→2（履歴と 200 点の積分）
  - seed からの通し: step 0・1・2・33 の x、solver の出力、KSampler の出力 latent
- 参考時間: 34 step の通しで 4.0 s（最適化なし）。


### 手順 6: VAE デコード（完了・ビット一致）

- 実装: `native/generate/vae.cpp`（`dsh_anima_gen::vae_decode(weights, z)`、Wan 2.1 / qwen_image_vae のデコーダー）。
  - 画像 1 枚（T = 1）では ComfyUI は特徴キャッシュを使わない。因果 3D 畳み込みは重みの最後の時間スライスだけを使い（`causal_zero`）、upsample3d の time_conv は通らない。つまり実質 2D。
  - ComfyUI はこの VAE を **BF16** で動かす（working_dtypes の先頭）。FP16 や FP32 で計算すると最大 0.21 ずれる。
  - attention（中間層、384 チャネル 1 ヘッド）は ComfyUI と同じ SDPA の優先順（flash → cuDNN → efficient → math）にした。
  - 入力は KSampler の出力（逆正規化済み）。逆正規化を二重にかけない。出力は `(x + 1) / 2` を [0, 1] に切った FP32 の [H, W, 3]。
- 照合: `tests/generate_vae_parity.py`。KSampler の latent から、VAEDecode の画像とビット一致（832×1216、42 ms）。

### 通しの確認

プロンプト → テキスト条件 → 34 step → VAE を、すべてネイティブ実装でつないだ。保存前の画像は ComfyUI の VAEDecode とビット一致。モデルを常駐させた状態で、プロンプトから画像まで 4.5 s（最適化前、GPU はほぼ空き）。


### 手順 7: LoRA（完了・ビット一致）

- 実装: `native/generate/lora.py`。キーは `diffusion_model.<path>.lora_A/B.weight`（学習で作る形式。alpha なし → 倍率 1、rank で割らない）。
- ComfyUI と同じく **BF16** で `w + ((強さ·alpha) · (B @ A)).to(BF16)` を計算する（FP32 で計算すると 3.7e-3 ずれる）。常駐中の base は変えず、変更したテンソルだけ新しい dict に入れる。未対応のキー・対になっていないキー・DiT にない重みはエラーにする。
- 照合: ComfyUI に `LoraLoaderModelOnly`（@darask、強さ 0.8）を入れて採取（`.runtime/anima-native/generate-reference-lora`）。`tests/generate_lora_parity.py` で「なし → あり → なし」がすべてビット一致。

### 手順 9: サイドバーの「Anima 生成」（完了）

- `native/generate/generate.py`: 1 ジョブ = モデルの読み込み → 指定枚数の生成（seed, seed+1, …）→ PNG（設定を `parameters` テキストチャンクに保存）。ComfyUI の venv の torch で動かし、C++ の演算は初回にビルドして `.runtime/anima-native/generate-build` にキャッシュする。
- `jobs.mjs`: 学習と同じキューに `kind: generate` のジョブとして入る（GPU は順番に使う）。LoRA 一覧はキーで Anima 用だけに絞る（H3・Qwen-Image の LoRA は出さない）。画像は `GET /api/darask.animaLora?image=<ジョブ>/<ファイル>` で返す（キューにあるジョブのファイル名だけ）。
- `client.js`: 「Anima 生成」パネル（プロンプト、ネガティブ、LoRA 最大 4 個と強さ、サイズ、枚数、シード、step、CFG）。キューの一覧は両方のパネルで共通。
- 確認: テスト用の DSH からパネルで生成（@darask 0.8、seed 42・43）。seed 42 の PNG は ComfyUI の LoRA 付きの画像と全画素一致。読み込み 3.5 s、1 枚 4.3 s（832×1216・34 step、最適化前）。

残り: 速度の内訳と最適化（手順 8）。ComfyUI の基準（フックなし・3 シードの中央値）も GPU が空いているときに測り直す。
