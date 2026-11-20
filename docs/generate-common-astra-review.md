dsh: reasoning:
**Evaluating evidence and tools**

I see we have strong points to consider. I want to dig deeper into the hidden issues related to the evidence on the speed of P-H3 count in the tool. We could look at the jobs.mjs and the tool usage default. However, it seems like we don't need to quantify the default unless we can verify it. There’s a lot to think about here, and I want to ensure we have everything lined up properly!

dsh: reasoning:
**Evaluating parity tests and outputs**

Okay, I'm looking at the parity tests, and it seems they don't execute the main or assert functions. The P-H3 meta additions change the output blob, so I need to compare the pixels or chunks separately. I should stick to the original edited prompt path and incorporate error saving inline. Also, H3 results, including schema or filename, might fit better in tests 2 or 5. I want to ensure the user prefers speed over quality in the design without overburdening. Each bullet needs a clear focus without excessive details.

初版は **`run(main)` と status の書き込みだけを共通化**するのが妥当です。テンソル読み込みと PNG 保存は、見た目ほど同じ処理ではありません。優先度順に5点です。

1. **本番の `python -I` で `job.py` を import できるようにする。** [jobs.mjs:376–380](plugins/media-shared/jobs.mjs#L376) は3本とも隔離モードで直接起動します。各入口が `sys.path` に足すのは自分の `HERE` だけで、新しい `job.py` はその外です。単純な `from job import run` は失敗します。[既存の H3 学習入口](plugins/media-shared/native/train_lora.py#L26)と同じく、モデル側の `HERE` を先頭に保ち、共有ディレクトリは後ろに追加してください。逆順だと H3 が Anima 側の同名 `build.py` を拾います。3入口を `-I` と不正な `request.json` で起動し、GPU を使わず `stage=error`・終了コード1まで確認できます。

2. **原子的な `os.replace` だけでは、進捗と保存完了の競合は防げません。** [Anima](plugins/image-generation/native/anima/generate.py#L101) では保存スレッドが `images`・`done` を、生成スレッドが `step` を更新します。個々の JSON は完全でも、遅れて届いた古い書き込みで進捗が巻き戻り得ます。非同期保存を残すなら、`images` への追記から JSON の置換までを `step` 更新と同じロックで直列化し、最後の `Future.result()` が成功してから `stage=done` にしてください。[jobs.mjs:387–391](plugins/media-shared/jobs.mjs#L387) はこの stage と終了コードで成否を決めます。また [H3](plugins/media-shared/native/generate.py#L204) は `done/total` を持たず、MP4 も `images` に載せます。書き込み手段は共有しても、画像用の初期値まで統一しない方が安全です。

3. **`load_tensors` は今回の変更から外す。** `ph3.tensors()` は生成内部だけでなく、[P-H3 学習のテキストキャッシュ作成](plugins/media-shared/native/train_lora.py#L132)からキー選別付きで呼ばれ、`generate.load()` も[条件付き生成のテスト](plugins/media-shared/native/test_cond_dit_parity.py#L59)から呼ばれています。削除すると生成ジョブが通っても別経路が壊れます。さらに [H3](plugins/media-shared/native/generate.py#L143) は `comfy_quant` をテキストエンコーダーでだけ除外し、[P-H3](plugins/media-shared/native/ph3.py#L47) は全ロードで除外したうえで scale・buffer の dtype を保ちます。一律除外は現状と同じ契約ではなく、ジョブ外枠の共通化にも不要です。

4. **P-H3 の非同期保存に「数百 ms 短縮」を見込まない。** [チャットの既定は `count=1`](plugins/lora-training/media-tool.mjs#L110) なので、最後に `close()` で待てば重ねる次の生成がありません。また [P-H3 の `images[].seconds`](plugins/media-shared/native/ph3.py#L158) は保存後、[Anima](plugins/image-generation/native/anima/generate.py#L126) は保存投入前の計測です。共通化で数字だけ速く見えるのを避け、まず P-H3 は同期のまま LoRA メタデータを追加するだけで十分です。非同期化は複数枚のジョブ完了時間で差が出た場合に限り、Anima 同様に未完了の保存を1枚に制限してください。

5. **「ビット一致」の対象を決め、変更箇所を直接テストする。** P-H3 は `parameters` に LoRA を足すので、画素が同じでも PNG ファイル全体は一致しません。[`test_dit_parity`](plugins/media-shared/native/test_dit_parity.py#L28) と [`test_ph3_parity`](plugins/media-shared/native/test_ph3_parity.py#L27) は主に中間値の差を表示し、status や保存失敗を検証しません。旧版・新版で **PNG は復号画素を完全一致、`parameters` は意図した LoRA 差分だけ許容、H3 は映像に加えて音声も比較**と明記してください。非同期保存の合否には、高価な生成を増やすより、小さな固定画像2枚と意図的な保存失敗で、ファイル順・`done`・例外伝播を確認する方が直接的です。
