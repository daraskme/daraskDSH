# ja-manga-writing

日本語の漫画を企画し、ネーム（コマ割り・構図・台詞・吹き出し・漫符・効果・擬音）を切り、講評し、画像生成モデルへ受け渡すための Claude Code スキル。4コマ、読切、連載、縦読みを扱う。[ja-novel-writing](https://github.com/daraskme/ja-novel-writing) の設計（書く前に読むもの・書いた後に読むものの分離、機械検査、別コンテキストのレビュー、proposed / confirmed の台帳）を漫画に移した。

## 何が違うか

- **ネームを構造化データで持つ。** 1 コマ = 1 レコードの TOML。段（tier）と幅で割ると、右綴じの読み順（段は上から下、段の中は右から左）で配置が決まる。自然文のネームのように、コマの境目・話者・カメラが曖昧にならない。
- **ネームを機械で検査する。** `scripts/name_lint.py` が、書いた順と配置の読み順の食い違い、めくりの位置、ノド側へのタチキリ、同じショットの連続、顔の寄りと台詞だけの連続、場面の頭で場所を見せていないコマ、吹き出しの字数・行の長さ・数、右の人物から話しているか、話者がコマにいるか、漫符の重なりと連続、成人向けの記号の混入、同じ場面での服の変化、台詞の中の字数の数え違いを拾う。点数は出さない。
- **ネームが見える。** `scripts/name_render.py` が、コマ割りと吹き出し（縦書き）・人物・漫符の位置を描いたネーム画像を、右綴じの見開きで並べた HTML にする。読者役に渡す台本は、ページの狙いやめくりの設計を落とした版を別に作る。
- **Darask AI Studio に組み込み済み。** Studio のプラグイン「ネームを作る」「ネームを検査・書き出す」が、このスキルの作法と scripts でネームを切り、検査し、書き出す（Studio の `skills/ja-manga-writing` に写しを置く）。
- **AI っぽくない絵づくり。** 決めコマ（`detail = "key"`）だけ人物も背景もきちんと描き、ほかは線を減らす。白黒は線・ベタ・トーン・白の四つで、灰色の厚塗りをさせない。`references/art-direction.md`。生成した絵の灰の塗りは `scripts/tone_check.py`（Pillow が要る）で測る。
- **コマ割りの型の保存庫。** よく使う型 30、珍しいが狙いのはっきりした型 12、作者の型を、目的・使う場面・コマごとの中身（各コマの shot の目安）・根拠の強さつきで保存。Manga109-s 全 87 冊のページを型に当てはめて集計した（`scripts/layout_library.py`、`references/layout-library.md`、一覧の絵は `assets/layouts/layouts.png`、型ごとの図（SVG・PNG）と解説は [`assets/layouts/`](assets/layouts/README.md)）。
- **実際の漫画の数字。** Manga109-s を注釈した 6,741 ページから、コマ数・大ゴマ・人物の大きさ・吹き出し・漫符を場面の種類ごとに集計した（`references/corpus-norms.md`）。
- **画像生成へそのまま渡せる。** `scripts/name_export.py` が、コマ別のプロンプト（画風ブロックと人物ブロックを全コマ同じ文面に固定）、ページ一括の設計図プロンプト、Krea 2 の `krea_layout_v1`（JSON / BBOX、人物ごとの `lora` 別名と `pose`、台詞の `speakers`）、写植用の文字一覧を書き出す。
- **漫符・演出・コマ割りの型を厚く持つ。** 漫符は 5 系統（感情の記号／体から出る効果／動きと画面の効果／擬音／文字の描き方）で、形・置き場所・意味・転用・使いすぎを一覧にした。場面の種類から引く「この場面にはこのコマ割り」のパターン集（36 型、ASCII 図と TOML の骨格つき）。
- **自分で描いたものを自分で採点しない。** 意図を知らされない読者役・編集役・校閲役を別コンテキストで走らせる。別系統のモデル（Codex CLI など）に見せる手順も同梱。
- **生成していない絵を「できた」と言わない。** 成果物はネームとプロンプト。生成した絵は検品の表で、読み順 → 人物 → 台詞 → 文字 → 演出 → 体 → 背景の順に見る。

## 導入

```bash
git clone <このリポジトリ> ~/.claude/skills/ja-manga-writing
```

必要なのは Python 3.11 以上（標準ライブラリのみ）。Python が無くてもネームは書けるが、機械検査と書き出しは飛ばされる。

## 使い方

Claude Code で普通に頼めばよい。

- 「猫と暮らす一人暮らしの 4 コマを 3 本」
- 「32 ページの読切のネームを。ジャンルはラブコメ、新人賞に出したい」
- 「連載を始めたい。プロジェクトを作って企画から」
- 「第 3 話のネームを、この筋で」
- 「このネームを Krea の layout JSON にして」「Nano Banana 用にページごとのプロンプトを」
- 「このネーム、コマ割りと演出を見て直して」
- 「この参考ページの漫符と演出を分析して、うちの演出シートに」

明示的に呼ぶなら `/ja-manga-writing`。

## 構成

```
SKILL.md            本体（不変の約束、モード判定、ネームを切るときの操作、読む条件表）
references/         必要なときだけ読む詳細
  name-format.md        ネーム TOML の書式
  story.md              企画・ログライン・ページ配分
  paneling.md           コマ割りの原則（読み順、コマ数、めくり、見開き、タチキリ、4コマ、縦読み）
  paneling-patterns.md  場面別のコマ割りパターン集
  staging.md            演出・構図・カメラ・想定線・コマ間の移り方
  manpu.md              漫符・効果・擬音・描き文字（5 系統）と分析テンプレート
  dialogue-and-lettering.md  台詞・吹き出し・写植
  characters.md         人物設計と見た目の一貫性
  genres.md             ジャンル契約
  image-generation.md   画像生成への受け渡し（コマ別・ページ一括・krea_layout_v1）
  visual-check.md       生成した絵の検品
  ai-tells.md           AI のネームに出やすい癖（書いた後に読む）
  formats.md            判型・ページ数・投稿規定・AI の申告と権利
  review-protocol.md    レビューと外部モデル
  longform.md           連載の運用
  roles/                読者役・編集役・校閲役・台帳係（サブエージェント用）
scripts/            検査・描画・書き出し・プロジェクト管理（Python 標準ライブラリのみ。tone_check.py だけ Pillow）
assets/templates/   プロジェクトの雛形とネームの見本（横開き 2 ページ、4コマ 1 本）
docs/dev/           調査の記録（出典つき）
evals/              スキルの評価用プロンプト
```

## スクリプトを単体で使う

```bash
python scripts/name_lint.py assets/templates/name.toml --stats
python scripts/name_render.py assets/templates/name.toml --html renders/sample.html
python scripts/name_render.py assets/templates/name.toml --md - --reader
python scripts/name_export.py assets/templates/name.toml --to krea --out renders/krea
python scripts/name_export.py assets/templates/name.toml --to krea --layout-prefix ph3 --out renders/ph3
python scripts/name_export.py assets/templates/name.toml --to page-prompt --out -
python scripts/name_export.py assets/templates/name.toml --to sections --out -
python scripts/name_convert.py name.json --out name.toml
python scripts/init_project.py my-manga --title 題名
```

`name_lint.py` の終了コードは FAIL なし `0`、FAIL あり `1`、入力エラー `2`。`name_export.py --to krea` は、書き出した layout に検査エラーがあれば `1`。`--krea-repo` に `krea2_layout` パッケージのあるディレクトリを渡すと、その `validate_layout` でも検査する。

テスト:

```bash
python -m unittest discover -s scripts/tests
```

## 出典

`CREDITS.md` を参照。調査の記録は `docs/dev/research-*.md`。投稿サイト・新人賞の規定、生成 AI の申告ルールは変わるので、投稿・応募の前に必ず最新の公式情報を確かめること。
