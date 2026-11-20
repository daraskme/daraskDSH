dsh: reasoning:
**Investigating sources**

I want to review the Comfy operation itself. It seems important to inspect both native and upstream likely web sources. There might be a lot of information to sift through, so I'll make sure to look at all possible angles to gather insights. It's about getting a comprehensive understanding, I guess. I should focus on ensuring all relevant details are covered as I go along. let's keep everything organized!

dsh: reasoning:
**Checking audio output**

I need to make sure that the actual working tree is adjusting while it's being used, especially regarding the parent implementation. I should verify the upstream audio negative offset rounding to see how that’s affecting everything. There’s also the need to look at the Comfy scale and its latent processing, checking the current references against my earlier assumptions. It's crucial to get all these details right!

dsh: reasoning:
**Considering sample code updates**

I need to think about whether the sample code has changed while I'm reviewing it. It's important to mention the existing features, but only if I'm sure about them. Also, I’m considering whether there could be another bug related to the Comfy model, particularly with how it converts cond_audio and rescale. I really want to ensure that I’m inspecting everything thoroughly for any potential issues!

dsh: reasoning:
**Considering cublas performance**

I'm thinking about the specific issue with the cublas status not being supported. It could be related to the SLA limits concerning the kernel and how dynamic routing plays a role here. I need to examine the kernel's T limits and assess any additional performance costs that may come into play. There's a lot to unpack in this situation, and I'll need to gather more information to clarify what's happening.

**区間を順に生成する方針でよいです。** Context Windows、汎用ガイド API、クロスフェードは初版には不要です。優先して直すのは次の6点です。

1. **参照画像ありの2区間目は、行の順序が違います。** `dit.cpp` と `sampler.py` にはすでに `cond_audio_rows`、音声の一括投影、無ノイズ処理があります。作り直す必要はありません。ただし現状は `映像keyframe → ref画像 → 条件音声`、ComfyUI はこのケースで `映像keyframe → 条件音声 → ref画像` です。**位置表・埋め込み・変調区間を同じ順に**直してください。SLA は `v0=target映像開始、a0=target音声開始` として、KV は `[0, ceil(v0/64))`、dense query は `[floor(a0/64), ceil(v0/64))` のままで正解です。統計は区間ごとにリセットします。

2. **「362上限」と末尾の `+24` は両立しません。** 30秒＝720枚では、第2区間は `720−362＋22＝380 → 396枚`、出力は `362＋396−22＝736枚` です。[上流の区間割り](https://raw.githubusercontent.com/ooizj/ComfyUI-H3-Long-Video/main/long_video.py)も396枚を許します。速度優先なら396枚を実機で1回測り、通れば「通常362・末尾最大396」と明記。362が厳守なら分割してください。API は `seconds` を増やさず既存の `length` を使い、**60秒が納品上限なら生成後に1440枚で切る**のが簡単です。

3. **音声の丸めと照合方法を修正してください。** 丸めるのはフレームではなく**40 Hz の整数座標**です。直前が362枚なら音声603 step、overhang＝−1/3。末端は `round(22×5/3−1/3)=36`、開始は `36−40=−4` tick（**−2.4フレーム**）です。[上流の配置式](https://raw.githubusercontent.com/ooizj/ComfyUI-H3-Long-Video/main/nodes.py)どおり、0に丸めません。公式 `AddGuide` の `latent` はガイドではなく生成対象で、負の整数も末尾基準です。照合はノードを増やさず、採取側で **1-step映像×7＋別位置の音声40 step＋ref画像1枚**を直接条件に入れるのが確実です。映像7枚を1ブロックにまとめると、条件ごとに再開するノイズ列も変わります。

4. **MP4 の逐次書き込みだけでは VRAM ピークは下がりません。** `vae.decode_video` は区間の FP32 RGB をリストに保持してから `cat` します。1280×704×362なら RGB 一組で約3.65 GiB、連結時は約7.3 GiBです。一方、同じ区間の映像 latent は約35 MiB。**60秒までなら各区間の latent を CPU に置き、最後に DiT を解放して1区間ずつデコード・書き込み**で十分です。専用ストリーミング VAE を先に作る必要はありません。

5. **先頭・末尾画像をどの区間に効かせるか決めてください。** 先頭画像は第1区間だけ、末尾画像は**最後に納品するフレーム**だけ、ref画像は全区間です。末尾を切るならアンカーも切り詰め後の位置に置きます。Qwen は1回ロードでよいですが、I2V の「先頭画像あり」の埋め込みを後続にも流すのは別の挙動です。必要な条件だけ先にエンコードし、Qwen は解放してください。

6. **通し試験は30秒・60秒の目視二本より、接合と最大負荷を分ける方が有効です。** 短い2区間で step 0/1 の映像・音声出力を同一入力で照合し、接合音は**耳でも**確認。32 kHz では22枚の頭切りは29,333サンプル、362枚区間は末尾に約267サンプルのゼロ詰めが生じます。AAC は区間ごとに再開せず、累積サンプル数を PTS にして最後だけ flush。長い実測は最大対応条件で1本でよく、**区間別秒数・ピーク VRAM・最終 A/V 尺**を記録してください。1280×704では継続条件だけで SLA の厳密 KV が **6,240行**増えるので、第1区間の速度だけでは見積もれません。
