# DeepSeek Harness の復元

GitHub の `anima-native-generate` ブランチからコードを取得し、Hugging Face の非公開バックアップからモデルと制作データを復元します。Hugging Face では `darask0` など、両方の保存先に読み取り権限があるアカウントでログインしてください。

- [モデルと LoRA](https://huggingface.co/darask0/daraskDSH-models) にモデルライブラリの90ファイル（約215 GiB）があります。
- [制作データと設定](https://huggingface.co/datasets/darask0/daraskDSH-data) に漫画プロジェクト、参照素材、生成結果、ComfyUI の入力とワークフロー、DSH の設定と履歴、復元用 CUDA ツールチェーンがあります。
- [復元マニフェスト](../scripts/restore/snapshot.json) に Hugging Face のコミット ID、ファイルサイズ、SHA-256 を固定しています。ComfyUI、diffusion-pipe、CUTLASS、Anima Ref Adapter の取得コミットは [復元スクリプト](../scripts/restore.py) に固定しています。

## 新しい環境に復元する

Linux x86-64、Git、Node.js 24.9 以上、npm または Corepack、uv、空き容量260 GiB以上が必要です。GPU 機能のビルドには CUDA 13.2 に対応する NVIDIA ドライバー、SM 12.0 の GPU、C++ コンパイラーと Rust/Cargo が必要です。

復元先ではハーネス本体を `harness/`、拡張機能を `plugins/`、モデルと LoRA を `models/`、生成結果を `outputs/` に配置します。

既存の履歴とサービス設定に絶対パスが含まれるため、元のパス `/mnt/6tb/daraskDSH` への復元を推奨します。既存の作業フォルダーがある場合は、別の場所にクローンして復元を確認できます。

```bash
git clone --branch anima-native-generate https://github.com/daraskme/daraskDSH.git /mnt/6tb/daraskDSH
cd /mnt/6tb/daraskDSH
uvx --python 3.12 --from huggingface-hub==0.36.2 hf auth login
uv run --no-project --python 3.12 --with huggingface-hub==0.36.2 \
  python scripts/restore.py --with-user-state
./scripts/setup-restored.sh
./dsh --version
./dsh web
```

ダウンロードが中断した場合は、同じ復元コマンドをもう一度実行します。保存済みファイルを照合して再利用します。内容が異なる既存の設定や制作ファイルがある場合は、対象パスを表示して停止します。別フォルダーでデータだけ確認するには `--skip-models` を使い、ホームに設定・履歴を入れない場合は `--with-user-state` を省略します。

別のパスへ復元すると学習設定のプロジェクトルートは復元先に合わせて書き換わります。履歴に記録済みのパスや、外部プロジェクト・ワークスペースのパスは必要に応じて設定画面で更新してください。利用するエージェント CLI や Cloudflare Tunnel の認証は復元先で設定し直してください。systemd で起動する場合は [harness/deploy](../harness/deploy/) のユニットにある作業フォルダーと実行ファイルのパスを合わせてください。

Python の仮想環境、npm の依存パッケージ、ネイティブビルド、学習用キャッシュは再作成します。ベンチマーク用の一時出力、認証トークン、作業フォルダー外のプロジェクトそのものはバックアップの対象外です。

## 共有データフォルダー `/mnt/6tb/dataset`

[非公開データセット](https://huggingface.co/datasets/darask0/darask-dataset) に、有効なバックアップを示す `manifest.json` がある場合は次の手順で復元できます。コミット ID を指定する場合は `--revision <コミットID>` を加えます。

```bash
uv run --no-project --python 3.12 --with huggingface-hub==0.36.2 \
  python scripts/mirror-dataset.py restore /mnt/6tb/dataset \
  --cache /mnt/6tb/daraskDSH/.runtime/dataset-downloads
```

元データ、注釈、画像、動画、LoRA とモデルを含みます。仮想環境、再生成できるキャッシュ、認証ファイルを除外します。一部のシンボリックリンクは元の環境でもリンク先が存在せず、フォルダー外へのリンクは元のリンク先を記録しています。ハーネスのモデルと同一のファイルは SHA-256 が一致する場合にモデル保存先から復元します。

更新後のファイルを保存するときは、新規の作業ディレクトリを指定します。別のファイルシステムを使う場合は、大きなファイルの一時コピー分の容量が必要です。アップロードが中断した場合は、同じ作業ディレクトリに対して `push` を再実行できます。全ファイルの照合後にマニフェストを更新します。

```bash
uv run --no-project --python 3.12 --with huggingface-hub==0.36.2 \
  python scripts/mirror-dataset.py pack /mnt/6tb/dataset .runtime/dataset-stage
uv run --no-project --python 3.12 --with huggingface-hub==0.36.2 \
  python scripts/mirror-dataset.py push .runtime/dataset-stage
```
