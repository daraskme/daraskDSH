TWAの作業方針（ユーザー指定、2026-11-18 JST）

この方針は作業の継続・解析・移植を依頼されたときに適用する。状態確認や質問だけでは作業を開始しない。ユーザーの最新の停止・変更指示を優先する。

全体の統括はサブスク枠の Codex Astra 6.2（local-cli、gpt-6.2-astra、effort high）に固定し、親チャットが計画、独立レビュー、統合を担当する。独立した実作業は task_queue で管理する。通常の並列プールは Cursor・Grok。Astraへ通常タスクを新規割当しない。Cursor/Grokで未解決のタスクは Codex Astra 6.2 → Claude Opus 5.5（effort high）の順に引き継ぐ。結果は統括のAstraが確認する。AstraはCodex CLIのサブスク認証を使う。Devinは親・子・相談先から除外する。過去のCursor統括、Devin起点、Luna→Sol、Opus→Astraの担当順は現在のユーザー指定で置き換える。

TWAの親チャットはAstra 6.2を統括にする。計画・統合の難所は独立した子タスクとして依頼し、親自身をOpusへ切り替えずAstraが統括を続ける。担当変更でも旧タスク・試行・証拠・受入記録は保持し、中断した作業を確認してから引き継ぐ。旧通常Astra担当の未着手・中断作業は履歴を保持してCursor/Grokへ再割当し、確認待ち・受入済み成果は保持する。外部承認や実行gate不足・安全上の拒否はモデルを替えて回避しない。子タスクは通常プールから開始する。

1. 最初に現在の正本を確認する。開始時の参照候補は ../work-management/20261115/goal01/PLAN.json・TASK_INDEX.jsonl・番号が最新のSTATE_REV*.json、docs/CHECKPOINT46_20261113.md、docs/checkpoints/20261113/STATE.json、docs/STATUS.md、docs/NEXT_SESSION.md、GOAL/PROGRESS文書。更新されていれば新しい正本を実測して使う。過去のlive_workersを現在も稼働中と決めつけず、他の作業との重複や競合を確認する。
2. 実装に先立ち、完全解析のフェーズ・領域・全タスクを仕分け、完全移植・統合・検証の依存関係も別途明示する。既知の未分類、未帰属、未解析、入力不足も省略しない。既存台帳の全IDとの対応・重複・抜けを照合し、根拠の文書と件数を残す。タスク数だけで完全性を宣言しない。
3. docs/dsh/plan.json と人間向け計画を作る。既存の正本・凍結revisionを上書きしない。各タスクは一つの受入条件群に収まる大きさにし、依存タスク、読み取り範囲、書き込み範囲、検証手順を明示する。分類の網羅確認が済むまで inventoryComplete を true にしない。
4. task_queue(action="import", planPath="docs/dsh/plan.json") → task_queue(action="run")。並列は最大モード。Cursor/Grokの枠は各4枠から開始し、成功に応じて拡大、制限エラーで縮小する。端末のCPU・空きメモリから総実行枠を計算し、実際の枠数はstatusのmaxParallel/basePoolを参照する。これはDSHの運用枠であり、サービスが保証した契約上限ではない。依存先が未受入、変更範囲同士または変更と参照が重なるものは同時実行せず、確認待ちの範囲も保持する。書き込みなしの静的解析は writes:[]。範囲を限定できなければ writes:["*"] として直列化する。親も稼働中の子が読む正本・共有ファイルを変更しない。
5. 子は範囲内で作業し、具体的に異なる二つの方法でも閉じない技術問題を unresolved で報告する。CLI障害や未解決報告はキューがAstra、Opusの順で引き継ぐ。引き継ぎ先として動くAstra/Opusの子は合計同時1件とする。統括のAstraとは別枠で、通常プールにAstraの子を追加しない。経緯・証拠・子セッションを次の担当へ渡す。実行権限、外部入力、プロジェクトのnative gate不足は external_blocked とし、別モデルで権限を回避しない。
6. 子の完了通知が来たら task_queue(action="status") を読む。途中の子の通知だけでキュー全体の完了を宣言しない。resolved は review（確認待ち）になる。親または別の独立担当が変更と実証を確認し、task_queue(action="review", taskId=..., accepted=true, evidence=[...]) で受け入れる。失敗なら accepted=false と根拠を渡し、次のモデルへ回す。既存方針が別作者・別モデルを要求する箇所ではその独立性を維持する。
7. 最後のOpus 5.5でも解けないものは unresolved のまま保存し、他のLLMへ渡せるようにまとめる。状態ファイルと UNRESOLVED.md にはタスクID、試したこと、証拠、変更箇所、停止理由、担当モデルを残す。手動で原簿を編集せず、task_queueを使う。再試行には新しい情報と理由が必要。同じ試行を無制限に回さない。
8. 待つだけならターンを終える。子の完了で自動的に次のターンが起きる。実行可能な仕事がある間は継続する。全体完成は全スコープの受入と統合検証を満たすまで宣言しない。Source静的確認、テスト未実行、局所一致、hash一致をゲーム全体の完成へ算入しない。
   通常担当の実行中が0件でも、現キューの受入完了を全体の作業終了と扱わない。未完了の正本・受入済みの次作業一覧から、依存を満たし書込範囲が競合しない次の仕事を新しいタスクIDで計画へ追加し、既存タスクと履歴を保持して import → run する。独立した仕事が複数あればCursor・Grokの両方を動かし、その後に統括の深い検証を進める。Rootの実行条件待ちの1件で他の静的解析まで止めない。投入できる仕事がない場合は不足する入力・依存・権限を具体的に記録する。
9. 一時停止は task_queue(action="pause")。再開は statusで状態と実ファイルを確認してから run。DSH再起動による中断は記録され、勝手に成功や上位モデルへの失敗に置き換えない。中断作業を同じ担当段階から再開し、二重適用を避ける。

statusの応答が切り詰められたらlimitを小さくし、offsetでページを進める。完全な履歴はstatePathとunresolvedPathから必要な範囲を読む。

計画JSONの形式（sources・acceptanceには具体的な文書位置/判定条件を入れる）:

```json
{
  "version": 1,
  "objective": "TWAの完全解析と完全移植",
  "inventoryComplete": true,
  "sources": ["現行の正本と全件対応表のパス"],
  "phases": [{"id": "analysis", "title": "完全解析"}],
  "tasks": [{
    "id": "analysis-001", "phase": "analysis", "title": "具体的な解析対象",
    "prompt": "作業内容・入力・検証方法を具体的に記載する",
    "acceptance": ["実証できる受入条件"], "dependsOn": [],
    "reads": ["analysis/source"], "writes": ["docs/dsh/tasks/analysis-001"],
    "baselineEvidence": []
  }]
}
```

baselineEvidence は既存正本で既に受入済みと実証されているタスクだけに使う。未解析のものをbaselineにして飛ばさない。baselineと今回のacceptedを区別して報告する。計画更新では旧タスクと試行記録を削除しない。既存タスク定義を変える場合は新revisionのIDを作る。
