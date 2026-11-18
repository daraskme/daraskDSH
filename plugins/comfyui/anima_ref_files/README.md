# Animaのキャラクター参照を書き出す

参照画像1枚をAnima用VAEでエンコードし、通常解像度と半解像度の参照を1つの `.safetensors` に保存します。キャラごとの学習は不要です。

## セットアップ

1. `anima_ref_files` フォルダーを `ComfyUI/custom_nodes/` に置きます。
2. [Anima Ref Adapter](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter) をインストールし、ComfyUIを再起動します。
3. 書き出しには `models/vae/qwen_image_vae.safetensors` を使用します。画像生成にはAnima base v1.0、Qwen3 0.6Bテキストエンコーダー、Anima Ref Adapter v2も必要です。モデルの配置は[公式のセットアップ手順](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter#quick-start)に従ってください。

## 書き出し

`example_workflows/Anima Ref - Export Safetensors.json` をComfyUIにドラッグし、参照画像とキャラ名を指定して実行します。保存先は `ComfyUI/output/anima_refs/<キャラ名>.safetensors` です。出力ディレクトリを変更している場合は、その中の `anima_refs/` に保存します。同名のファイルがある場合は `_002` 以降の連番が付きます。実行後の「保存先」ノードに実際のパスが表示されます。

## 画像生成

`example_workflows/Anima Ref - Generate from Safetensors.json` を開き、「保存したキャラを選択」でファイルを選び、シーンのプロンプトを入力して実行します。一覧に新しいファイルが出ない場合は、ノード定義を更新するかブラウザーを再読み込みしてください。別の端末に渡すときは、その端末の `output/anima_refs/` にファイルを置きます。

自分のグラフでは `Load Anima Ref (.safetensors)` の出力を `Anima Ref Guider.ref_latent` に接続します。共通のアダプター重みは `Anima Ref Adapter Loader` で読み込みます。書き出したファイルはAnima専用です。MiniMax H3のRefModローダーとの互換性はありません。

## Gemma 4 の部位候補

「Gemma 4 - 部位候補だけを出す」では、最初のノードにComfyUIの `input/` を起点とする画像名を1行ずつ入力します。ローカルのComfyUI-Folder-LoRAに設定されたGemma本体とmmprojを使います。実行すると入力画像ごとの候補を `output/gemma_candidates/` に保存します。各PNGに対応する `candidates.json` に、部位、出典、元画像のピクセル座標、可視性、Gemmaがその部位を選んだ理由が入ります。候補は画像ごとに変わります。出力された説明と座標は元画像で確認してください。

## Qwen Image 2.1 の四方向図と衣装アップ

この端末ではComfyUIのワークフロー一覧の「氷渡香名江」に衣装ごとの作成グラフがあります。元資料の候補と座標補正から、Qwen Image 2.1が四方向図、装飾アップ、髪／上半身／下半身／衣装固有のアップを作ります。Qwen用のVAEで画像を生成し、最後にAnima用VAEで衣装別の参照ファイルを書き出します。4コマの順番は正面、画面左向き、画面右向き、背面です。

読み込みノードの `reference` で使用する参照を選びます。`front` は一人の正面全身、`front_details` は正面と装飾、`overview` は四方向と元資料を含む一覧です。`upper_body` は元画像、`qwen_upper_body` は描き起こした上半身です。同様に `hair_fastening`、`lower_body`、`clothing_detail` が使えます。古い単一参照ファイルは `overview` で開けます。

別キャラに使う場合は基準画像とGemmaの入力画像を差し替え、元キャラ固有の座標補正を消してから候補を確認してください。補正は `{"部位名":{"file":"inputからの相対パス","box":[左,上,右,下],"description":"特徴"}}` の形式で、座標は元画像のピクセル単位です。不要な候補は `null` で除外できます。参照シートに使う部位名とQwenの衣装指示も新しい衣装に合わせます。
