# DeepSeek Harness の使い方

ターミナルで作業したいフォルダーに移動して起動します。このインストールの本体は DeepSeek 公式の `@deepseek-ai/dsh@0.2.2-rc.1` です。

```bash
cd /mnt/6tb/daraskDSH
dsh web
```

ブラウザーが自動で開かない場合は、ターミナルに表示されたログイン用 URL を開いてください。標準プリセットで、入力欄のモデル選択で **使用するLLMサービス → モデルの種類 → エフォート・オプション** の順に指定できます。Google AI Ultra、Muse、Grok、Cursor、Codex、Claude Code、Devin に対応します。初期選択は Codex です。各サービスの CLI のログインを使います。Local CLI を使うための DeepSeek API キーは不要です。

**Claude Code の再ログイン**は `dsh login claude` で開始できます。

## Google AI Ultra

Google AI Ultra の契約枠には公式 **Antigravity CLI** (`agy`) を使います。[Google公式ダウンロード](https://antigravity.google/download)からCLIを導入して、`dsh login google` でUltra契約のGoogleアカウントにログインします。ブラウザーに表示されたコードを端末に入力し、追加のアカウント確認が求められた場合はGoogleの画面で完了してください。

この端末には `/home/darask/.local/bin/agy` を導入済みです。DSHの「使用するLLMサービス」で **Google AI Ultra** を選びます。モデル一覧はログインしたアカウントから取得し、同じモデルのエフォート違いは「エフォート」で選択します。契約や提供モデルが変わった場合は「一覧を更新」を押してください。APIキーやCloudのADC認証への切り替えは行いません。

```bash
dsh login google
dsh ask google --model gemini-4-carbon-medium --effort high '差分をレビューして'
```

[GoogleのCLI移行案内](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/) · [Antigravity CLIの仕様](https://antigravity.google/docs/cli/headless/)

## 親と子を指定する

Web UI では「Muse に設計レビューを頼んで。Muse からは Grok に境界条件を確認してもらって」と依頼できます。親が `subagent` ツールで指定された子を起動し、子の完了通知と結果が親に戻ります。7サービスすべてを親にも子にも選べます。この構成の上限は親の下に4段です。

ターミナルからも親を指定できます。サービス ID は `google`、`muse`、`grok`、`cursor`、`codex`、`claude`、`devin`、`devin-free` です。

```bash
dsh ask cursor 'Codex にテスト計画をレビューしてもらって'
```

CLIが説明文と実行用JSONを混ぜて返した場合は、そのステップで一度だけ形式の修正を求めます。修正できなければ未実行のエラーとして表示します。説明文に含まれるJSONを直接実行することはありません。

## TWAのタスク仕分け・並列実行・自動引き継ぎ

TWAで作業の継続を依頼すると、[専用の作業指示](plugins/task-router/TWA_INSTRUCTIONS.md)に従い、現行の正本を読み、全フェーズ・全タスク・依存関係・受入条件を `docs/dsh/plan.json` に仕分けてから実行します。統括は**サブスク枠のCodex Astra 6.2（high）に固定**し、通常の担当は **Cursor・Grok**。Devinはこの担当プールから除外しています。Cursor/Grokが解決できないタスクを **Astra 6.2 → Claude Opus 5.5**（両方high）へ引き継ぎます。Astraへ通常タスクを新規割当しません。親のAstraは計画・独立レビュー・統合を担当します。

並列は**最大モード**です。Cursor/Grokは各4枠から開始し、成功に応じて増やし、制限エラーで減らします。総枠は端末のCPU・空きメモリから算出し、この端末では調査時48枠。これは提供元が保証する最大数ではありません。引き継ぎ担当のAstra/Opusは合計1件とし、統括のAstraとは別枠で動きます。同じ範囲を変更・参照する仕事や依存先が未受入の仕事は待機します。[上限調査・実測・設定方法](plugins/task-router/CONCURRENCY.md)

TWAの親チャットはAstra 6.2を統括にし、難所をOpusへ委任しても全体の統括はAstraが続けます。担当変更でも旧タスクと試行・証拠・受入記録を保持します。旧通常Astra担当の未着手・中断作業はCursor/Grokへ再割当し、確認待ち・受入済みの成果は保持します。他のフォルダーでは一般チャット用の「自動引き継ぎ（Cursor → Astra → Opus）」も選べます。通常担当のCursor/Grokは「設定 → LLMサービス」の既定モデルを使います。Astraは統括・引き継ぎ担当とも `gpt-6.2-astra`、effort `high` に固定し、Codex CLIのサブスク認証を使います。

エージェント用 `task_queue` は `import / run / status / review / pause / retry / escalate` を提供します。結果はまず確認待ちになり、親が証拠を確認して受け入れるまで依存タスクは開始しません。未解決事項・試行・証拠・変更ファイル・担当モデルは `$DSH_HOME/task-runs/<workspace-hash>/state.json` と `UNRESOLVED.md` に保存し、実際のパスは `status` で返します。再起動後は状態確認後に `run` で再開します。既存TWAの作業・正本の書き換えは、継続依頼を受けてから行います。

`node tests/task-router.mjs`、`node tests/auto-routing.mjs` と `python3 tests/task-router-integration.py` で、割当・増減・競合・依存・引き継ぎ・停止再開を確認できます。

## コンテキストの自動圧縮

長い会話は、Local CLIの推定コンテキスト量が **49,152トークン** に達すると自動圧縮します。以前の32,768より間隔を延ばし、長い解析作業で圧縮が頻発しない設定です。直近 **8,192トークン程度** を原文で保持し、それ以前は目的・決定事項・未解決事項・次の作業を含む引き継ぎ要約にします。ツール呼び出しと結果の組を分断しないため、保持量は前後します。元の会話ログとタスクキューの状態は残ります。

設定は `config/agents.yml` の `compaction-basic` です。`thresholdRatio` を小さくすると早めに圧縮し、大きくすると間隔が延びます。Local CLIでは `0.25` が32,768、現在の `0.375` が49,152トークンに相当します。実際の間隔は会話・ツール出力の量によります。変更後は `node scripts/patch-ui.mjs` を実行し、作業の区切りでDSHを再起動します。Web版の4つの組み込みプリセットにも同じ設定が適用され、保存済み会話は再開時に新しい設定を使います。手動で圧縮する場合は、応答が止まっているときに `/compact` を送信します。

## 別のセッションで続きを扱う

同じ作業フォルダーで新しい Harness セッションを始めると、直近に作成された会話のユーザー発言と回答が、最大 8 KiB の抜粋として最初の入力に付きます。空の会話は飛ばします。古い決定も「前のセッションで決めたテスト方針を探して」のように頼めます。エージェントは同じフォルダーのセッション履歴を検索できます。Web UI の `@` で過去の会話を明示的に指定することもできます。

Harness の保存先は標準で `~/.local/state/darask-dsh` です。設定済みの `DSH_HOME` または `XDG_STATE_HOME` があれば、その場所を使います。共有の対象はこの Harness で作成したセッションです。選んだ CLI には現在の会話と引き継ぎ内容がモデル入力として渡ります。

## 動作確認

`python3 tests/agents.py` はテスト用の CLI を使って、実際の Harness 上で親子49通りの結果返却、プロセス再起動後の引き継ぎ、作業フォルダーの分離を検証します。各アカウントの疎通は `dsh ask grok 'OK とだけ答えて'` のように確認できます。

[DeepSeek 公式リポジトリ](https://github.com/deepseek-ai/deepseek-harness)

## Studio

[Studio](https://studio.darask.me/) は既存の Cloudflare Access のメール認証を使います。新規セッションでは入力欄の右下にあるモデル名を押して、最初のメッセージに使うモデルを選べます。「ワークスペースを選択」を押すと Ubuntu デスクトップにフォルダー選択画面が開きます。スマートフォンなどでは Studio 内の一覧から Ubuntu のフォルダーを選べます。クラウド側のトンネルと Access ポリシーは既存のものを使用しています。

1つのチャットで複数のフォルダを扱うには、最初に通常どおりワークスペースを選び、新規チャットではワークスペース選択行、会話開始後はチャット上部の **作業フォルダ** を開きます。**フォルダを選択…** で追加先を選び、**保存** を押します。登録済みワークスペースの一覧や、絶対パスの入力からも追加できます。相対パスはメインフォルダを基準にします。追加先の絶対パスはAIにも伝わり、ワークスペース書き込みモードではそのフォルダも編集できます。**外す → 保存** でそのチャットの対象から外せます。追加先はチャットを開き直しても保持されます。新しい環境では `./dsh workspace setup` で機能を有効にします。

「設定 → スキル・指示ファイル」で、DSH 全体または登録済みのワークスペースを選べます。全体の指示は `$DSH_HOME/AGENTS.md`、プロジェクトの指示は選択したフォルダーの `AGENTS.md`、`CLAUDE.md` と各 `.local.md` に保存され、新しい会話で読み込まれます。スキル一覧では名前と説明を確認でき、選択した範囲のスキルを編集・作成できます。プロジェクト用スキルはGitルート（Git管理外なら選択したフォルダー）の `.dsh/skills/` に保存されます。

**Devin の無料モデル**は「使用するLLMサービス → Devin（無料のみ）」から使えます。設定の「LLMサービス」で Devin がアカウントに対して無料と表示したモデルを選べます。無料の判定は実行時にも行います。通常の「Devin」では有料モデルも選べます。エフォートと速度のバリエーションはモデル名に含まれます。

```bash
dsh ask devin-free 'この関数をレビューして'
dsh ask devin-free --model swe-2-flash-medium 'テストの観点を3つ挙げて'
dsh ask codex --model gpt-6.1-sol --effort high --fast '差分をレビューして'
dsh ask claude --model opus --ultracode '設計をレビューして'
```

「設定 → LLMサービス」でサービス別の既定値を保存できます。入力欄で「適用」した選択はチャットごとに保存され、再読み込み後も維持されます。後から既定値を変更しても、そのチャットの明示指定は変わりません。既存のチャットやサービス ID のみを指定する子エージェントは既定値を使います。コマンドの引数はその実行に適用されます。Grok の速度、Cursor のエフォート・速度は対応バリエーションのモデル名で選びます。Claude の高速モードはアカウントの利用条件に従います。

**OpenAI API** は「設定 → モデル → OpenAI API → 編集」で API キーを登録するとモデル一覧に表示されます。ターミナルでは登録済みのキーまたは環境変数 `OPENAI_API_KEY` を使えます。API の利用料は OpenAI プロジェクトに計上されます。

```bash
dsh ask openai --model gpt-5-mini '短く要約して'
dsh ask openai --model gpt-6.2-astra --effort high '設計をレビューして'
```

`--effort` はモデルが対応する段階（例: `low`〜`max`）を指定でき、未対応の値は送信前にエラーになります。pi-ai の同梱カタログにまだない `gpt-6.2-astra` と `gpt-6.2-sol` は、`config/agents.yml` の `llm-pi-ai.providers.openai.models` に追加しています。この一覧は同梱カタログを置き換えるため、pi-ai の更新で増えたモデルを使うときは、この一覧にも追記してください。

サイドバーには残量が最も少ない枠と次のリセット日時（日本時間）が表示されます。各行にポインターを合わせるか、キーボードでフォーカスすると、取得できた他の枠も確認できます。「約」は CLI が残り時間を丸めて返す場合に表示されます。検索には OSS の DDGS と lxml を使い、X は Yahoo! リアルタイム検索の公開投稿と検索エンジンのインデックスを参照します。

本番のユーザーサービスは `darask-dsh.service` と `darask-dsh-tunnel.service` です。Web の入口はループバックの 8789 番で、HTTP と WebSocket の双方で Cloudflare の署名と対象アプリ、期限を検証します。ローカル開発では `npm ci` と `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt` で依存関係を復元できます。

## Anima LoRA 学習

### Anima Base v1.0 のネイティブ LoRA 学習

画像と同名の UTF-8 キャプション（例として `01.png` と `01.txt`）をフォルダーに揃えます。学習に使う Anima Base、VAE、Qwen3 と、キャプション用 Gemma のパスは `~/.config/comfy-folder-lora/config.json` から読みます。

初回セットアップは次の 2 コマンドです。学習環境の CUDA 対応 PyTorch・NVRTC、C++ コンパイラ、Ninja、Rust/Cargo を使います。`prepare` は GPU 上で凍結エンコーダーの変換結果を検証し、約 1.5 GiB のファイルを `.runtime/anima-native/frontend` に作成します。

```bash
./dsh anima build
./dsh anima prepare
```

学習は CLI からも、DSH チャットの `anima_lora_train` ツールからも起動できます。

左サイドバーの **Anima LoRA** を押すと、フォルダを選んで「キャラ・スタイル・コンセプト」とトリガーワードを指定するだけで学習できます。フォルダ内の画像はすべてローカルの Gemma 4 Ortenzya 31B（GGUF、`~/.config/comfy-folder-lora/config.json` の `vision`）でキャプションを付け、各キャプションの先頭にトリガーワードを入れます。既存のキャプションを作り直す場合は、元のファイルをフォルダ内の `.caption-backup-日時/` に退避します。学習設定は目的ごとの推奨値（1024px、スタイルは rank 8・cross-attn 1e-5・リスタート cosine、キャラ・コンセプトは rank 16）を使い、出力は ComfyUI の `trained/` に保存されます。実行中は進行状況と残り時間が表示され、中止もできます。

```bash
./dsh anima train --folder /path/to/images \
  --output ./my-anima.safetensors --resolution 512 --rank 16 --steps 200 \
  --cache-dir ./.anima-cache
```

通常の学習は C++/LibTorch と CUDA で行い、Rust が画像・トークナイザー・safetensors を扱います。出力には新しいファイル名を指定します。ComfyUI からすぐ選びたい場合は `--output` に `plugins/dsh-comfyui/loras/trained/my-anima.safetensors` を指定してください。

`--epochs` では素材一巡を単位に期間を指定できます。約 572px 以下の同じサイズの画像は 2 枚ずつまとめて学習し、その step は学習率を 2 倍にします（`--learning-rate` は 1 枚あたりの値）。512px では 1 枚あたり約 10% 速くなります。モジュールごとの学習率は `--self-attn-lr`・`--cross-attn-lr`・`--mlp-lr`・`--adaln-lr` で指定し、0 にしたモジュールは学習しません（例: MLP のみの画風 LoRA は `--self-attn-lr 0 --cross-attn-lr 0 --adaln-lr 0`）。`--warmup 0.06 --lr-cycles 4` でウォームアップ付きのリスタート cosine、`--timestep-scale 1.2` で timestep 分布、`--weight-decay 0` で減衰を変えられます。`--backend compile` で PyTorch Inductor による学習も選べます。解像度と学習回数ごとの選び方は [実測結果](docs/anima-native-benchmarks.md) を参照してください。[実装計画 v2](docs/anima-native-plan-v2.md)・[Astra 6.2 レビュー](docs/anima-native-astra-review.md)・[v1 計画](docs/anima-native-plan.md)・[Opus 5.5 レビュー](docs/anima-native-opus-review.md)

学習エンジンとチャット用ツールは DSH プラグイン [`plugins/dsh-anima-lora-trainer`](plugins/dsh-anima-lora-trainer/README.md) にまとめています。

DSH の ComfyUI 連携（サイドバーの ComfyUI、`comfyui_*` ツール、`/comfy/` の中継）と ComfyUI-Folder-LoRA ノードは削除しました。学習済み LoRA とモデルは `plugins/dsh-comfyui/` に残り、ローカルの ComfyUI（`.runtime/ComfyUI`、`darask-comfyui.service`）からは引き続き読み込めます。MiniMax H3 の動画生成と P-H3 の画像生成・編集はネイティブ実装に移行する予定です。
