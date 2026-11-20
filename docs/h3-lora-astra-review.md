dsh: reasoning:
**Investigating historical logs**

I'm looking into the historical logs, checking out the checkpoint header shapes and DiT configs dropout settings. There's a lot to consider here! I want to make sure I'm thorough, so I'll need to look deeper into each of these elements and their implications. It’s interesting to think about how they all connect and affect performance. Let’s see what insights I can gather from this investigation!

dsh: reasoning:
**Investigating quantized training**

I'm looking into the actual quantized Bwd upstream training and how I can implement custom measurements for memory and compute in a quantitative way. I wonder if there's a way to handle unconditioned attention masks — like, maybe they could be entirely masked out as a reference? I really need to determine how to approach this effectively. It feels like I'm diving into technical details, but it's crucial to understand.

dsh: reasoning:
**Updating the worktree**

I'm updating the code worktree actively. I think I might be able to execute the requested review alongside other implementations, so I need to be precise about the current state. I want to ensure I'm checking the latest changes in the native train and any relevant updates that have happened. It's all about staying on top of everything that's going on!

dsh: reasoning:
**Consolidating and checking**

I realize I have plenty of time to work with, so I need to consolidate my thoughts succinctly. I should listen for input from the agent and verify the actual `preprocess` for fps/crop, while addressing any oddities with the H3 encoding. Gathering empirical data is key, focusing on the documentation’s 8 steps. My goal is to obtain a robust uncond cache embedded artifact, and maybe I can simplify things by reading binary metadata without using the shell. It might be helpful to obtain a global view.

dsh: reasoning:
**Revisiting training metrics**

I think measuring training times sorted by batch size is a good idea, but I am not sure if running requests is necessary. It might help to sort the data without going through unnecessary steps. I'll need to pin the first four exact data points. Also, I need to check the latest documentation to see if anything has changed. I should specify both initial and subsequent steps in my report. The source data seems verified.

dsh: reasoning:
**Reviewing code modifications**

I need to look at the recently edited `dit.cpp` files, focusing on changes related to proj and media. I wonder if I'll find some specific modifications that could help with the loRA alpha settings and any optimization issues. It seems essential to inspect these details carefully to get a clearer understanding of the adjustments made. I’m curious about how the changes impact the overall functionality. Let’s dive in!

dsh: reasoning:
**Verifying optimizer findings**

I'm looking into the 4-bit optimizer and its findings, considering the learning rate warmup schedule and linear starts for learning rate or steps. I need to verify the code about batch size being zero and mixing audio. I'm also noticing a potential speed issue with the long P-H3 configuration, which seems to be costly in terms of performance. Coverage has been completed, but these details will need further inspection.

dsh: reasoning:
**Clarifying encoding conditions**

I need to separate the idea about fixed acceptance conditions and not duplicating the video encoder. It might be better to focus on parity in points 1, 4, and 5, while considering item 7 for existing encoders. Also, I want to take a look at item 8, which is Anima. I wonder if I need to read more about the potential packed layout, including whether hidden alignment padding is necessary. Could DP PackedLayout involve text, audio, video, and alignment tokens?

**H3 は diffusion-pipe と同じ入力で勾配・更新を照合してから速度を測る。P-H3 は「同じ学習の移植」ではなく、新しい学習条件として評価する。** この二つを分けるのが、最も手戻りの少ない修正です。以下、優先度順です。

1. **H3 の学習式は、音声時刻と loss の重みが計画と違います。** `uniform`・動画 shift 8・画像 shift 1 は今回のベンチ設定で、そこから得た映像時刻 `σv` を Eros の学習では **`σa=time_shift_sigma(σv,12,3)`** に変換します。`10→3` は H3 の生成設定です。両モダリティとも `x=(1−σ)x0+σε`、目標 `ε−x0` は正しい。ネイティブの `dit_forward` は `1000*σv` を受けて**すでに負号を付けた出力**を返すため、学習照合では `audio_scale=1`・dense とし、再反転しません。また loss は packed token 数ではなく **FP32 の二乗誤差和÷スカラー要素数**です。映像は1行96要素、音声は32要素なので、行数で重み付けすると音声を相対的に3倍重くします。音声なしは長さゼロで、無音トラックとは別です。[DP の時刻・loss](.runtime/diffusion-pipe/models/minimax_h3.py#L376)／[native の出力符号・音声補正](plugins/media-shared/native/dit.cpp#L386)

2. **CFG=4 の式は正しいですが、「batch 1 なので mask 不要」は誤りです。** 符号を揃えた予測に対する `(cond + 3*stopgrad(uncond))/4` で、両分岐とも同じ noisy latent・**現在の LoRA と refiner**を使います。固定したベースモデルの uncond 出力は使い回せません。実際の DP の空プロンプトキャッシュは埋め込み `[1,5120]`、mask は `[0]`。一方、native の空文字・画像なしは行がなく `torch.cat(rows)` で失敗します。まず **cond は全キー有効なときだけ mask を外し、uncond は pad の位置を残してキーをマスク**すれば十分です。行ごと消すと後続の映像・音声の位置もずれます。DP が実際にどの attention kernel を使ったかは profiler で判断してください。[DP の mask と分岐](.runtime/diffusion-pipe/models/minimax_h3.py#L599)／[native の空入力処理](plugins/media-shared/native/cond.py#L53)

3. **P-H3 の基準は、DP の重みパスを差し替えるだけでは作れません。** 同梱 ComfyUI には P-H3 専用の Qwen3-VL-4B の5層出力→connector→5120次元の経路と、専用画像 VAE の選択がありません。さらに P-H3 生成は shift `1/1` で、静止画でもノイズ入り音声 `[1,32,2,2]` の4行を attention に入れます。DP の音声なし `[B,32,2,0]` とは画像側から見ても別入力です。**H3 の合否**は Eros＋VBVR、latent、ノイズ、時刻、mask、LoRA 初期値を揃え、実 DP と条件別出力・AV loss・A/B 勾配・更新を比較すること。**P-H3 の合否**は専用 encoder と W8A8 順伝播を生成側に照合し、音声行の扱いを明記することです。P-H3 の cfg=4 は実績のある互換条件ではないので、速度優先なら uncond forward が不要な cfg=1 と比べる価値があります。[同梱ローダー](.runtime/diffusion-pipe/submodules/ComfyUI/comfy/sd.py#L2007)／[専用ローダー](.runtime/ComfyUI/comfy/sd.py#L2096)／[P-H3 の条件と音声形状](plugins/media-shared/native/ph3.py#L73)

4. **W8A8 順伝播＋BF16 逆伝播は妥当な近似ですが、`fc2` が落とし穴です。** 凍結ベースの `dX=dY @ W_eff` には、int8 重みに出力行ごとの scale を掛け、**入力軸を256要素ずつ逆 Hadamard 回転**した `W_eff` を使います。これは活性量子化の真の微分ではなく STE です。LoRA は回転・量子化前の特徴に掛けます。特に fused `fc2` の LoRA 入力は生の gate/up ではなく `z=silu(gate)*up` で、`dZ` を両方に戻す必要があります。順伝播は既存 W8A8、逆伝播は**同じ上流 `dY`**を与えた独立な BF16 計算と照合してください。B=0 初期化なら初回の `dA=0` は正常なので、非ゼロ B または2 step目も確認します。dequant 演算自体はすでに登録済みです。200投影を毎 step 展開すると BF16 重みの書き出しだけで**約35.9 GiB/回**になるため、「常駐コピー禁止」は決め打ちせず、96 GB 機では一度だけ BF16 展開する案と実測比較すべきです。[`proj` の fused 経路](plugins/media-shared/native/dit.cpp#L54)／[既存 dequant 演算](plugins/media-shared/native/kitchen/kitchen_ops.cpp#L163)

5. **「速くなる」はまだ仮説ですが、「DP が起動しない」はすでに古いです。** 既存ログには rank 32、学習パラメータ **155,058,176** で、最初の4 step が **54.032 / 14.354 / 13.897 / 14.277 秒**と記録されています。これは Eros 単体・swap 8・unsloth・compile 有効で、形状も混在しており、bucket 別の定常値ではありません。凍結重みの `dW` 省略と TE/VAE キャッシュは DP にもあり、ネイティブ固有の利得に数えられません。比較は両者を **Eros＋VBVR、同じ optimizer・CFG・入力**に揃え、DP 側も swap 0・収まる最小限の checkpoint を試すべきです。1024² 画像は映像1024行ですが、512²×136フレームは映像だけで9472行です。「画像で収まる」から動画を外挿せず、bucket 別に同期後の定常 s/step 中央値とピーク VRAM、別にキャッシュ・compile 込みの完走時間を出せば判断できます。[実設定](.runtime/h3-lora-bench/dp/train.toml#L13)／[実測ログ](.runtime/h3-lora-bench/dp/run.log#L523)

6. **LoRA の208投影という対象は正しい一方、保存後の生成で refiner 8投影が抜けます。** 実重みは両モデルとも `(DiT 50 + refiner 2) × 4 = 208`。P-H3 は本体200投影が int8、refiner 8投影は BF16 で、後者は `dit.cpp::proj` を通らず `condition.cpp` の `at::linear` を直接呼びます。したがって学習キャッシュは **refiner 前の5120次元**まで（P-H3 の `condition()` の第3戻り値）。生成では conditioning 前に BF16 refiner にも適用してください。H3 は複数 LoRA を順に処理するループが既にあり、不足は主に request とキー形式です。現行 merge は Kohya up/down＋alpha 専用、DP の A/B は `alpha=rank` で実効倍率1。**H3 は VBVR→ユーザー LoRA の順、両モデルで refiner だけ非ゼロの LoRA を保存・再読込し、strength 0/1 で出力差を確認**すれば、ゼロ LoRA テストが見逃す適用漏れを検出できます。[merge と refiner](plugins/media-shared/native/condition.cpp#L17)／[H3 の既存適用ループ](plugins/media-shared/native/generate.py#L149)／[DP の alpha 規約](.runtime/diffusion-pipe/train.py#L115)

7. **`train/encode.py` の新設は不要です。足りないのは前処理を含む合否判定です。** 現在の [`native/encode.py`](plugins/media-shared/native/encode.py#L88) には動画・音声 encoder があり、[既存比較](plugins/media-shared/native/test_encode_av_parity.py#L26)も34フレーム・静止画・ステレオを通しています。ただし誤差は表示だけです。新設する代わりに許容誤差を assert し、24 fps 化、32 kHz リサンプル、同じ先頭区間への AV 切り出し、音声トラックなしを含む入力を1組追加する方が有効です。学習 bucket は **1または17の倍数**で、生成の `17k+5` ではありません。静止画も H3 の posterior mean と P-H3 の固定 seed による posterior sample を共通化しないでください。

8. **Anima と今すぐ共有するのは、GPU キュー・進捗・成果物の公開までで十分です。** 現在の [`pipeline.py`](plugins/lora-training/pipeline.py#L88) は静止画1枚を Gemma に渡し、Anima 専用 CLI を起動します。「動画の数フレームをそのまま渡す」のは再利用ではなく新機能です。空間 bucket も Anima の16刻みと H3 の32刻みでは一致しません。初版は同名 `.txt` を使う CLI とし、`common.py`、動画 captioner、パネル統合は後回しでよいです。**H3 の実 DP 更新一致 → H3 の同条件速度比較 → P-H3 の STE 検証・速度比較 → UI** の順なら、速度差が出なかった場合にも不要な実装を残しません。
