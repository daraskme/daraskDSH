# dsh-manga-studio

左サイドバーの「マンガ」タブ。プロジェクト → コマ割りされたページ（見開き）→ 各コマ、の順にたどって漫画を作る DSH プラグイン。

- **ネーム**: [ja-manga-writing](https://github.com/daraskme/ja-manga-writing)（`skill/` に写し。版は `skill/VERSION`）の作法で、エージェント CLI（既定は Claude Code）がネームを JSON で切る。プロンプトには SKILL.md と参照（書式・企画・コマ割り・演出・台詞・人物・絵づくり・ジャンル）、漫符の語彙、コマ割りの型の要約（Manga109-s 全 87 冊の集計）を入れる。返ったネームは `manga.py`（`skill/scripts` の `manga_name` / `name_lint` / `name_export`）で割り付け・検査し、コマごとの英語プロンプトを作る
- **コマ割り**: ページごとに型（`skill/scripts/data/layouts.json`、図と解説は `skill/assets/layouts/`）を選ぶと、そのコマ枠になる。AI も `pages[].layout` で型を指定できる
- **絵**: コマごとにネイティブ Anima（LoRA 対応。全コマ共通の画風 LoRA と、人物ごとのキャラ LoRA）か、P-H3 の参照生成（人物の設定画から）。生成した絵は P-H3 の編集で直す（設定画も参照に渡せる）。GPU は Anima LoRA / 生成 / H3 と共通のキュー（`plugins/dsh-anima-lora-trainer/jobs.mjs`）
- **写植**: 台詞は吹き出し（縦書き）として絵の上に載せる。擬音は既定で写植、設定で絵に描かせることもできる。ページは PNG で保存できる
- 保存先: `.runtime/manga/projects/<id>/`（`project.json`、`images/`、`refs/`）。削除したプロジェクトは `.runtime/manga/trash/`
- API: `/api/darask.manga`（`api.mjs`）。ネームの解析は `python3 manga.py analyze name.json`

スキルを更新するときは、ja-manga-writing の `SKILL.md`・`README.md`・`CREDITS.md`・`references/`・`scripts/`（tests を除く）・`assets/templates/`・`assets/layouts/{svg,text,README.md}` を `skill/` に写し、`skill/VERSION` を書き換える。
