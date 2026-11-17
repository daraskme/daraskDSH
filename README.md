# DeepSeek Harness の使い方

ターミナルで作業したいフォルダーに移動して起動します。このインストールの本体は DeepSeek 公式の `@deepseek-ai/dsh@0.2.2-rc.1` です。

```bash
cd /mnt/6tb/daraskDSH
dsh web
```

ブラウザーが自動で開かない場合は、ターミナルに表示されたログイン用 URL を開いてください。標準プリセットで、モデル選択の **Local CLI** から Muse、Grok、Cursor、Codex、Claude Code、Devin を選べます。初期選択は Codex です。各サービスの CLI のログインを使います。Local CLI を使うための DeepSeek API キーは不要です。

**Claude Code の再ログイン**は `dsh login claude` で開始できます。

## 親と子を指定する

Web UI では「Muse に設計レビューを頼んで。Muse からは Grok に境界条件を確認してもらって」と依頼できます。親が `subagent` ツールで指定された子を起動し、子の完了通知と結果が親に戻ります。6つすべてを親にも子にも選べます。この構成の上限は親の下に4段です。

ターミナルからも親を指定できます。モデル ID は `muse`、`grok`、`cursor`、`codex`、`claude`、`devin`、`devin-free` です。

```bash
dsh ask cursor 'Codex にテスト計画をレビューしてもらって'
```

## 別のセッションで続きを扱う

同じ作業フォルダーで新しい Harness セッションを始めると、直近に作成された会話のユーザー発言と回答が、最大 8 KiB の抜粋として最初の入力に付きます。空の会話は飛ばします。古い決定も「前のセッションで決めたテスト方針を探して」のように頼めます。エージェントは同じフォルダーのセッション履歴を検索できます。Web UI の `@` で過去の会話を明示的に指定することもできます。

Harness の保存先は標準で `~/.local/state/darask-dsh` です。設定済みの `DSH_HOME` または `XDG_STATE_HOME` があれば、その場所を使います。共有の対象はこの Harness で作成したセッションです。選んだ CLI には現在の会話と引き継ぎ内容がモデル入力として渡ります。

## 動作確認

`python3 tests/agents.py` はテスト用の CLI を使って、実際の Harness 上で親子36通りの結果返却、プロセス再起動後の引き継ぎ、作業フォルダーの分離を検証します。各アカウントの疎通は `dsh ask grok 'OK とだけ答えて'` のように確認できます。

[DeepSeek 公式リポジトリ](https://github.com/deepseek-ai/deepseek-harness)

## Studio

[Studio](https://studio.darask.me/) は既存の Cloudflare Access のメール認証を使います。新規セッションでは入力欄の右下にあるモデル名を押して、最初のメッセージに使うモデルを選べます。「ワークスペースを選択」を押すと Ubuntu デスクトップにフォルダー選択画面が開きます。スマートフォンなどでは Studio 内の一覧から Ubuntu のフォルダーを選べます。クラウド側のトンネルと Access ポリシーは既存のものを使用しています。

**Devin の無料モデル**は「Local CLI → Devin（無料のみ）」から使えます。設定の「ローカル CLI」で Devin がアカウントに対して無料と表示したモデルを選べます。無料の判定は実行時にも行います。通常の「Devin」では有料モデルも選べます。エフォートと速度のバリエーションはモデル名に含まれます。

```bash
dsh ask devin-free 'この関数をレビューして'
dsh ask devin-free --model swe-2-flash-medium 'テストの観点を3つ挙げて'
dsh ask codex --model gpt-6.1-sol --effort high --fast '差分をレビューして'
dsh ask claude --model opus --ultracode '設計をレビューして'
```

「設定 → ローカル CLI」の選択は保存後の親・子エージェントに反映されます。コマンドの引数はその実行に適用されます。Grok の速度、Cursor のエフォート・速度は対応バリエーションのモデル名で選びます。Claude の高速モードはアカウントの利用条件に従います。

**OpenAI API** は「設定 → モデル → OpenAI API → 編集」で API キーを登録するとモデル一覧に表示されます。ターミナルでは登録済みのキーまたは環境変数 `OPENAI_API_KEY` を使えます。API の利用料は OpenAI プロジェクトに計上されます。

```bash
dsh ask openai --model gpt-5-mini '短く要約して'
```

サイドバーには残量が最も少ない枠と次のリセット日時（日本時間）が表示されます。各行にポインターを合わせるか、キーボードでフォーカスすると、取得できた他の枠も確認できます。「約」は CLI が残り時間を丸めて返す場合に表示されます。検索には OSS の DDGS と lxml を使い、X は Yahoo! リアルタイム検索の公開投稿と検索エンジンのインデックスを参照します。

本番のユーザーサービスは `darask-dsh.service` と `darask-dsh-tunnel.service` です。Web の入口はループバックの 8789 番で、HTTP と WebSocket の双方で Cloudflare の署名と対象アプリ、期限を検証します。ローカル開発では `npm ci` と `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt` で依存関係を復元できます。
