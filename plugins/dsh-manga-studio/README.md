# dsh-manga-studio

左サイドバーの「マンガ」タブ。プロジェクト → コマ割りされたページ（見開き）→ 各コマ、の順にたどって漫画を作る DSH プラグイン。仕組みの全体は [skill/docs/integrations/darask-studio-manga-tab.md](skill/docs/integrations/darask-studio-manga-tab.md)（ja-manga-writing に入れた正本の写し）。写植・層・検品・CBZ の道具もスキルの scripts を使う。

- **ネーム**: [ja-manga-writing](https://github.com/daraskme/ja-manga-writing)（`skill/` に写し。版は `skill/VERSION`）の作法で、エージェント CLI（既定は Claude Code）がネームを JSON で切る。プロンプトには SKILL.md と参照（書式・企画・コマ割り・演出・台詞・人物・絵づくり・ジャンル）、漫符の語彙、コマ割りの型の要約（Manga109-s 全 87 冊の集計）を入れる。返ったネームは `manga.py`（`skill/scripts` の `manga_name` / `name_lint` / `name_export`）で割り付け・検査し、コマごとの英語プロンプトを作る
- **コマ割り**: ページごとに型（`skill/scripts/data/layouts.json`、図と解説は `skill/assets/layouts/`）を選ぶと、そのコマ枠になる。AI も `pages[].layout` で型を指定できる
- **絵**: コマごとにネイティブ Anima（LoRA 対応。全コマ共通の画風 LoRA と、人物ごとのキャラ LoRA）か、P-H3 の参照生成（人物の設定画から）。生成した絵は P-H3 の編集で直す（設定画も参照に渡せる）。GPU は Anima LoRA / 生成 / H3 と共通のキュー（`plugins/dsh-anima-lora-trainer/jobs.mjs`）
- **写植・演出は AI が層で載せる**: Anima（と P-H3）には人物と背景だけを描かせ、吹き出し・台詞・書き文字（擬音）・作中の文字・効果線（集中線・スピード線・フラッシュ・縦線など）・頭上の漫符は描かせない（プロンプトから外し、ネガティブにも入れる）。顔・体そのものの表現（赤面・涙・震えなど）は絵に残す。
  - 「AI に置かせる」（ページ・コマ単位、指示も書ける）で、Claude Code がコマの絵（コマの縦横比に切り抜いたもの）を Read で見て、ネームの台詞・擬音・効果・漫符を層として置く（`api.mjs` の `letterTask`、写植の作法は `skill/references/dialogue-and-lettering.md`）。絵の無いコマはネームの配置から推定。置く前はネームからの自動配置（`skill/scripts/name_layers.py` の `auto_layers`）
  - 層はコマ画面でドラッグして動かし（吹き出しのしっぽの先も）、形・文字・スタイル・大きさ・回転を直せる。自動配置に戻せる
  - `skill/scripts/typeset.py` が層をコマごとの SVG にする。文字はフォントの輪郭をパスにする（HarfBuzz で縦書きの字形）ので、どの端末でも、PNG に保存しても同じに見える。DSH が常駐させる
- **書き文字のスタイル（19 種）**: CLIP STUDIO ASSETS の無料の描き文字ブラシの傾向（マーカー・べちょ筆・SE 文字ペン・横太・ふち / 二重フチ・ぽてぽて・手書きセリフ・丸筆）を参考に、フリーフォント＋加工（フチ・斜体・字ごとの揺れ・だんだん大きく・震えの残像・立体の影・残響・速度の尾）で作った。迫力 / 威圧 / 鋭い SE / 横太・二重フチ / ポップ・ふち / ぽてぽて / 描き文字マーカー / べちょ筆・墨 / 草書 / 手書きペン / 丸文字 / やわらか手書き / 電子音 / 立体 / 袋文字の立体 / 震え・不気味 / 静寂 / 鼓動・残響 / 抜き文字。用途と作り方は `skill/references/sfx-styles.md`、見本は [skill/assets/sfx/sfx-styles.png](skill/assets/sfx/sfx-styles.png)
- **フォント**: すべて Google Fonts の SIL OFL 1.1（`skill/scripts/data/fonts.json`。台詞はしっぽりアンチック B1、ナレーションは Zen Old Mincho、書き文字は Dela Gothic One・Reggae One・RocknRoll One・Mochiy Pop One・Darumadrop One・Yusei Magic・Zen Kurenaido・Hachi Maru Pop・DotGothic16・Rampart One・Yuji Boku・Yuji Syuku・Potta One・Yomogi・Train One）。本体（約 74 MB）は git に入れず `.runtime/manga/fonts` に置く
- **AI 検品と自動の直し**（ComicStudio の Vision QA → targeted retry を参考）: 「検品する」で、Claude Code が各コマの絵（と人物の設定画）を Read で見て、ネームの人物・人数・動作・小道具・ショットと、`skill/references/visual-check.md` に照らす。問題のあるコマには直し方（描き直しの追加プロンプト、または P-H3 の修正指示）を出し、自動で直す設定なら直しをキューに入れ、仕上がった絵を採用して検品し直す（既定は最大 2 回）。結果はページのコマに「検品 OK / 要確認」として出る
- **人物の設定画を AI で描く**（AI Comic Generator・StoryDiffusion を参考）: 人物の外見の英文から Anima で全身の立ち絵（solo）を描き、設定画がまだなければ 1 枚目を使う。設定画は P-H3 の参照生成・修正と検品に渡る。コマの P-H3 修正には「人物を設定画にそろえる」指示を一押しで入れられる
- **ページ**: PNG で保存できる（絵・枠・層）。作品まるごと **CBZ**（ページの PNG と ComicInfo.xml、右綴じ `Manga=YesAndRightToLeft`）でも書き出せる（comic-maker を参考）
- 保存先: `.runtime/manga/projects/<id>/`（`project.json`、`images/`、`refs/`）。削除したプロジェクトは `.runtime/manga/trash/`
- API: `/api/darask.manga`（`api.mjs`）。ネームの解析は `python3 manga.py analyze name.json`

準備（初回）:

```bash
uv venv .runtime/manga-venv --python 3.12 && uv pip install --python .runtime/manga-venv/bin/python fonttools uharfbuzz pillow
.runtime/manga-venv/bin/python plugins/dsh-manga-studio/skill/scripts/fetch_fonts.py --dest .runtime/manga/fonts
```

スキルを更新するときは、ja-manga-writing の `SKILL.md`・`README.md`・`CREDITS.md`・`references/`・`scripts/`（tests を除く）・`assets/templates/`・`assets/layouts/{svg,text,README.md}` を `skill/` に写し、`skill/VERSION` を書き換える。
