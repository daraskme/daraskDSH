# Darask AI Studio の「マンガ」タブ（仕組みの全体）

Studio（DeepSeek Harness）のプラグイン `dsh-manga-studio` が、このスキルを使って漫画を作る。左のサイドバーの「マンガ」から、**プロジェクト → コマ割りされたページ（見開き）→ 各コマ**の順にたどる。プラグインはこのリポジトリを `skill/` に写して使う（版は `skill/VERSION`）。この文書は、そのタブが何を、どの順に、どの道具でしているかをまとめたもの。

## 1. 流れ

```
企画（題名・アイデア・ページ数・ジャンル・画風）
  │  ① ネームを切る（エージェント CLI。既定は Claude Code）
  ▼
ネーム（name JSON）── name_lint で検査、コマ割りの型を当てる
  │  ② コマごとに絵を描く（ネイティブ Anima、または P-H3 の参照生成）
  ▼
コマの絵（候補から採用）── ③ AI 検品 → 直し（描き直し / P-H3 編集）→ 再検品
  │  ④ 写植・演出の層（AI が絵を見て置く → 手で動かして直す）
  ▼
ページ（絵＋枠＋層）── ⑤ PNG / CBZ で書き出す
```

人物の三面図と場所の背景（⑥）は、②の「参照で描く」・③の検品・P-H3 の編集に使う。

## 2. 各段階

### ① ネームを切る

- エージェント CLI（Claude Code・Codex・Google・Grok・Cursor・Muse から選ぶ）に、`SKILL.md` と参照（`name-format.md`・`story.md`・`paneling.md`・`staging.md`・`dialogue-and-lettering.md`・`characters.md`・`art-direction.md`・`genres.md`）、漫符の語彙（`manpu.json`）、コマ割りの型の要約（`layouts.json`）を渡し、まず `<plan>` に作品の約束・ログライン・ページ配分・めくり・決めコマ、次にネームを JSON で書かせる。
- ページに `layout`（型の ID）を書かせると、そのページのコマ枠は型どおりになる（bbox に直し、`order = "written"`）。
- 返ったネームは `manga_name`・`name_lint` で割り付けと検査をし、FAIL・WARN をタブに出す。ページ単位で「このページを直す」と指示して書き直させられる。
- 実際に Claude Code に 3 ページの読切を切らせると約 3 分、検査は FAIL 0・WARN 1 だった。

### コマ割り

ページごとに、コマ数の合う型を図で選んで当てる（`layout-library.md`、図と解説は `assets/layouts/`）。コマの追加・削除もできる（型は外れて段組みに戻る）。

### ② 絵

- **参照で描く（既定）**: 画風は Anima、指示どおりのポーズ・表情・構図は P-H3。コマの人物の三面図と場所の背景を `<Picture k>` として渡し、`name_layers.reference_prompt` と同じ文（カメラ・人数・位置・ポーズ・強い表情・場面・描き込み）で P-H3 に描かせ、白黒にする（`image-generation.md` §7）。大きさはコマの縦横比で面積およそ 1344²。
- **Anima だけで描く**（ネイティブ実装、LoRA 対応）: `name_export.py --to anima` と同じ形のプロンプト（文字・層で描く演出を外し、T5 512 トークンに収め、avoid と文字類をネガティブへ）。大きさは面積およそ 1024²。全コマ共通の画風 LoRA と、その人物が出るコマだけのキャラ LoRA をかける。
- **P-H3**（設定画とネームのプロンプト）: 以前のやり方。
- 既定の描き方と白黒化は作品の設定で変えられる。
- 1 枚目は自動で採用。候補から採用し直せる。GPU は学習・生成・動画と共通のキューで順番に使う。

### ③ AI 検品と直し

`visual-check.md` §5 の手順。Claude Code が各コマの絵（コマの縦横比で切り抜き）と人物の設定画を Read で見て、ネームと突き合わせ、コマごとに `ok`・問題・直し方（`regenerate` の追加プロンプト / `edit` の指示）を返す。自動で直す設定なら直しをキューに入れ、描けた絵を採用して検品し直す（既定 2 回）。直しが全部失敗したら同じ絵は見せずに止める。ページのコマには「検品 OK / 要確認」が出る。

### ④ 写植・演出の層

`lettering-layers.md` のとおり。置く前はネームからの自動配置（`name_layers.auto_layers`）。「AI に置かせる」（ページ・コマ単位、指示も書ける）で、Claude Code が絵を見て置く。コマ画面では層をドラッグで動かし（吹き出しのしっぽの先も）、形・文字・スタイル・大きさ・回転を直せる。層は `typeset.py` が SVG（文字はパス）にする。プラグインは `typeset.py serve` を常駐させ、1 ページ（5 コマ）でおよそ 0.3 秒。書き文字のスタイルは `sfx-styles.md` の 19 種。

### ⑤ 書き出し

- ページの PNG（絵・枠・層。ブラウザの canvas で組む）。
- 作品まるごとの **CBZ**（全ページの PNG と ComicInfo.xml。`Manga=YesAndRightToLeft`、`BlackAndWhite=Yes`）。スクリプトだけで作るなら `page_compose.py --png` と `make_cbz.py`。

### ⑥ 三面図と場所の背景（下準備）

「下準備（三面図・背景）」で、まだ無い人物の三面図と場所の背景をまとめて Anima で描き、1 枚目をそれぞれの参照にする（人物は「AI で三面図を描く」、場所は人物タブの「場所（背景）」で個別に描き直し・選び直しもできる。`characters.md` §4）。場所はネームの `place_en`（無ければ `place`）で、その場面の後のコマへ引き継ぐ（`name_layers.places`）。生成したコマの絵を設定画にすることもできる。コマの P-H3 編集には「人物を設定画にそろえる」指示を一押しで入れられる。

見本の 3 ページの読切で、下準備（三面図 3 人・背景 1 か所）は 32 秒、P2 の 5 コマ（各 2 枚）は 137 秒だった。

## 3. 保存するもの

`.runtime/manga/projects/<id>/project.json`:

| 項目 | 中身 |
|---|---|
| `brief` | 企画（アイデア・ページ数・ジャンル・形式・rating・ネームを切る AI） |
| `name` | ネーム（このスキルの name JSON。人物には `lora_name`・`lora_strength`・`ref`（設定画）を足す） |
| `plan` | AI がネームの前に書いた設計 |
| `render` | Anima の先頭の文・ネガティブ・画風 LoRA・step・CFG、既定の描き方（`engine`: ref / anima / ph3）、白黒化（`mono`） |
| `places` | 場所ごとの背景（`key`・`text`・`ref`） |
| `takes` | コマごとの生成の記録（エンジン・ジョブ・状態・絵のファイル・プロンプト・修正の指示）。`@char:<id>` は三面図、`@place:<id>` は場所の背景 |
| `adopted` | コマごとの採用した絵 |
| `layers` | コマごとの層（`source`: ai / edited。無いコマは自動配置） |
| `qa` / `qaLoop` | コマごとの検品の結果、ページの検品の回数と待っている直し |
| `task` | AI の作業（ネーム・写植・検品）の状態 |

絵は `images/`、設定画は `refs/`、AI に見せる切り抜きは `crops/`。

## 4. 準備

```bash
pip install fonttools uharfbuzz pillow      # 写植（typeset.py・page_compose.py）
python scripts/fetch_fonts.py               # フォント（SIL OFL、約 74 MB）を MANGA_FONT_DIR か .fonts/ へ
```

ネームの解析・検査・書き出しは標準ライブラリだけで動く。

## 5. 参考にしたもの

| もの | 取り入れたこと |
|---|---|
| ComicStudio（Vision QA → targeted retry） | 検品して、直しをプロンプトに戻して描き直し、回数に上限を付けて検品し直す |
| AI Comic Generator | 設定画を自動で作る、コマ単位で描き直す |
| StoryDiffusion | 人物をコマ間でそろえる（ここでは三面図・場所の背景からの参照生成と、検品で） |
| ComicCrafter AI | コマ単位で絵や台詞を直し、他のコマは残す |
| comic-maker | CBZ で書き出す |
| CLIP STUDIO ASSETS の無料の描き文字ブラシ | 書き文字のスタイルの傾向（`sfx-styles.md` §2）。素材そのものは使っていない |

## 6. 分かっている限界

- 画像モデルは動作を取り違えやすい。参照で描くと「台に両手をつく」は合ったが、「頭を下げる」が合掌になった。検品で見つかるが、2 回で直りきらないことがある。`pose` を具体的に書く。
- 参照で描くと、表情が控えめになりやすい（指示文で「はっきり強く」と書いている）。
- 写植の層を AI に置かせるには、画像を読めるモデル（Claude Code）が要る。他の AI ではネームと人物の位置から推定する。
- 吹き出しの字の大きさは箱に合わせて自動で決まる。字数に対して箱が小さいと字が小さくなる。
