# ComfyUI for DeepSeek Harness

## この端末の保存場所

| 開きたいもの | このフォルダーからの場所 |
|---|---|
| 拡散モデル | [models/diffusion_models](models/diffusion_models/) |
| テキストエンコーダー・VAE | [models/text_encoders](models/text_encoders/) · [models/vae](models/vae/) |
| Gemma 4 の GGUF・画像用 mmproj | [models/gguf](models/gguf/) |
| 通常のLoRA・学習済みLoRA | [loras](loras/) · [loras/trained](loras/trained/) |
| Animaのキャラ別・衣装別参照 | [refmods](refmods/) |
| ComfyUIの保存済みワークフロー | [workflows](workflows/) |
| 衣装資料・四方向図・装飾アップ | [characters/氷渡香名江](characters/氷渡香名江/) |
| Gemmaの自動切り抜き候補 | [gemma_candidates](gemma_candidates/) |
| フォルダーキャプション・LoRA学習ノード | [custom_nodes/ComfyUI-Folder-LoRA](custom_nodes/ComfyUI-Folder-LoRA/) |

DeepSeek HarnessのチャットからローカルのComfyUIを操作するプラグインです。Web版ではサイドバーの **ComfyUI** から同じサーバーのノードエディターを開けます。

## 導入

Node.js 24.9以上、pnpm、DeepSeek Harness `0.2.2-rc.1` と、起動済みのComfyUIが必要です。ComfyUI `0.45.0` で検証しています。リポジトリを取得し、そのディレクトリで実行します。

```sh
pnpm pack
dsh plugin --profile web add "file:$PWD/dsh-comfyui-0.1.0.tgz"
dsh web
```

ターミナルでも使う場合は同じパッケージを `--profile headless` に追加します。GitHub公開後は `dsh plugin --profile web add github:OWNER/dsh-comfyui` でも導入できます。OWNER は公開先アカウント名に置き換えてください。すでに同名ツールを読み込む構成では、登録元を一つにしてください。

接続先の既定値は `http://127.0.0.1:8188` です。別ポートはHarnessの起動環境で `DSH_COMFYUI_URL=http://127.0.0.1:8189` のように指定します。WebプロファイルではHTTPとWebSocketの両方をHarnessのオーナーセッションで認証します。ComfyUI本体とモデルは別途用意してください。[ComfyUIセットアップ](https://docs.comfy.org/installation/manual_install)

## チャットで使う

- 「チェックポイントの一覧を見せて。SDXLのモデルで青い陶器のカップを生成して、画像リンクを返して」
- 「保存済みの cup.json を読み、背景を夕景にしたワークフローを別名で保存して」
- 「このAPI形式のワークフローを実行して、完了したら結果を見せて」

| ツール | 操作 |
| --- | --- |
| comfyui_catalog | モデル、LoRA、ノードの検索と入力仕様の取得 |
| comfyui_generate | SD 1.x / SDXLチェックポイントでの文章からの画像生成 |
| comfyui_run | エディターからエクスポートしたAPI形式のグラフの投入 |
| comfyui_job | 実行状態、失敗理由、生成物、実行したグラフの取得 |
| comfyui_workflow | エディター形式のワークフローの一覧、読み込み、保存 |

Animaなどの分割モデルや学習したLoRAを使うときは、対応モデルのAPI形式グラフを `comfyui_run` に渡します。生成物にはWeb版の `url` と、ComfyUIが動作する端末で開ける `localUrl` が付きます。[ComfyUIのAPI形式とルート](https://docs.comfy.org/development/comfyui-server/comms_routes)

開発時のテストは `pnpm install && pnpm test` で実行できます。削除は `dsh plugin --profile web remove dsh-comfyui` です。
