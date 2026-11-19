# ネームの書式（name TOML）

- いつ読むか: ネームを初めて書くとき、フィールドの意味や配置の指定を確かめたいとき、lint の E 群（構造の誤り）が出たとき。
- いつ読まないか: 企画・プロットの段階（`story.md`）。コマ割りの考え方そのもの（`paneling.md`）。

ネームは 1 話 1 ファイルの TOML で書く。1 コマが 1 つの `[[pages.panels]]`。ファイルに書いた順が、作者が意図した読み順になる。`name_lint.py` は、この順と、配置から計算した読み順が食い違うと P02 を出す。

同じ形の JSON（拡張子 `.json`）も、すべてのスクリプトがそのまま読む。LLM に書かせるときは JSON のほうが崩れにくい。人が直すときは `name_convert.py` で TOML にする。

TOML にしたのは、LLM が書きやすく、`tomllib`（標準ライブラリ）で検査でき、差分が読めるから。自然文のネームは一見読みやすいが、コマの境目・話者・カメラが曖昧になり、画像生成への受け渡しで崩れる（`docs/dev/research-1-ai-manga-pipeline.md` の 1 節、自然文だとコマ同士が混ざるという実践談）。

## 全体の形

```toml
[meta]
title = "放課後の忘れ物"
episode = 1
format = "page"               # page（横開き・右綴じ）/ yonkoma（4コマ）/ webtoon（縦読み）
first_page_side = "left"      # 1 ページ目が左ページ（奇数ページ＝左）。扉を右に置く媒体なら "right"
rating = "general"            # general / adult
cast = "../characters/cast.toml"   # 人物表を別ファイルにするとき（ネームからの相対パス）
style = "..."                 # 画風ブロック（省略すると manga.toml の [render].style）
page_aspect = 1.414           # ページの 縦/横（省略すると manga.toml か 1.414）

[[characters]]                # 人物（cast と両方あれば、ここが勝つ）
id = "misaki"
name = "美咲"
lora = "misaki"
appearance = "teenage girl, short black bob hair, ..."

[[pages]]
number = 1
purpose = "掴み: ..."         # このページの狙い（設計情報。読者用の台本には出ない）
tiers = [0.42, 0.33, 0.25]    # 段の高さの比（上から）
turn = "..."                  # 左ページで、めくりの引きと、めくった先で見せるもの

  [[pages.panels]]
  # コマの中身（下の「コマ」の節）
```

`meta` の `format` `first_page_side` `rating` `title` と、`style` `negative` `page_aspect` は、書かなければ上の階層の `manga.toml`（`[work]` と `[render]`）から入る。単発の 4 コマや一場面なら `manga.toml` 無しでネーム 1 枚で完結してよい。

## ページ（`[[pages]]`）

| キー | 意味 |
|---|---|
| `number` | ページ番号。左右（奇数＝左が既定）とめくりの位置はここから決まる |
| `purpose` | このページの狙い。読者に見せない設計情報 |
| `tiers` | 段の高さの比。省略すると使っている段の数で等分 |
| `turn` | 左ページに書く。最終コマで何を引き、めくった右ページの 1 コマ目で何を見せるか。無いと P07（INFO） |
| `spread` | `true` でこのページから見開き。右ページから始める（違えば P08） |
| `splash` | `true` で大ゴマ 1〜2 枚の扉・見せ場。コマ数の警報（P01）を止める |
| `summary` | 画像生成用のページ全体の英語の説明（省略時は画風ブロック＋コマ数） |
| `note` | 作者へのメモ（読者用の台本には出ない） |

## コマ（`[[pages.panels]]`）

### 配置

段組み（ふつうはこれ）:

| キー | 意味 |
|---|---|
| `tier` | 段。0 が最上段 |
| `width` | 段の中の幅の比（既定 1.0）。段の中のコマは **右から左へ** 並ぶ |
| `column` | 同じ段で同じ値のコマは、縦に積む（右側だけ 2 コマに割る、など）。幅は最初のコマの `width` |
| `height` | `column` で積んだときの高さの比 |
| `bleed` | タチキリ。`true`（四辺）か `"top,left"` のように辺を指定。基本枠に接している辺だけ紙の端まで伸ばす。ノド側（左ページの右辺、右ページの左辺）は P12 |
| `inset` | はめ込みゴマ。重なりの警報（P10）を止める |

段の指定（`tier` `column` `width` `height`）がページに一つも無く、`tiers` も無ければ、叩き台の段組みが入る。1 段に 2 コマずつ上から、コマ数が奇数なら最上段を 1 コマの横長にする。読み順は書いた順のまま。見せ場の大ゴマは、自分で段を書いて作る。

自由配置: `bbox = [x1, y1, x2, y2]`（ページ全体を 0〜1000 とした座標。y も 0〜1000 で、縦横比は `page_aspect` で戻す）。斜めゴマは描けないので、外接する長方形で書き、`note` に斜めであることを残す。

4コマ（`format = "yonkoma"`）は配置を書かなくてよい。4 コマずつ縦一列に等分し、8 コマなら右の列から 2 本並べる。縦読み（`webtoon`）は `height`（幅に対する高さの比、既定 1.0）と `gap`（前の余白、同じく幅に対する比、既定 0.25）で縦に積む。

### 中身

| キー | 値 | 意味 |
|---|---|---|
| `id` | 文字列 | 省略時 `P<ページ>-<順>` |
| `scene` | 文字列 | 場面 ID。値が変わったコマが場面の頭（P03・P11 の判定） |
| `place` / `time` | 日本語 | 場所と時間。場面の頭では必ず書く |
| `place_en` / `time_en` | 英語 | 画像生成に渡す版（無ければ日本語のまま渡る） |
| `shot` | `extreme_long` `long` `full` `medium` `bust` `closeup` `extreme_closeup` `insert` | ショットサイズ。日本語（ロング、バストアップ、アップ…）でも書ける |
| `angle` | `eye` `high` `low` `bird` `worm` `dutch` `pov` `over_shoulder` `back` `profile` | アングル。日本語（俯瞰、煽り、主観…）でも書ける |
| `transition` | `moment` `action` `subject` `scene` `aspect` `non_sequitur` | 前のコマからの移り方（`staging.md`） |
| `size` | 自由 | 「大」「小」などのメモ（配置には効かない。大きさは `width`・`tiers` で決める） |
| `beat` | 日本語 | このコマで起きること。読者に見える中身として書く（「封筒を拾った美咲が固まる」） |
| `visual` | 英語 | 画像生成に渡す画の説明。`beat` の翻訳ではなく、絵に描けるものだけを書く |
| `background` | `detailed` `simple` `white` `tone` `black` `effect` | 背景の描き込み。寄りで `simple` 以下が続くと P05 の判定に入る。`detailed` は場面の頭と決めコマに絞る（P15） |
| `order`（ページ） | `written` | 書いた順を読む順にする。重なり・はめ込みで形から読む順を推定できないページ用。書き出しもこの順になる |
| `detail` | `key` `normal` `light`（日本語: 決め・通常・簡略） | コマの仕上げ（人物と線の密度）の段階。`key` は人物を丁寧に描く決めコマで 1 ページに 1 つまで（P15）。`background` を書いていない決めコマは背景もきちんと描かせる（書いた `background` が優先）。`light` はつなぎで線を減らす。省略は `normal`。画材・画風は変えない（画風ブロックが決める）。ネームの最後の工程で付ける（`art-direction.md` §1） |
| `action` | `true` | 寄りでも動きがあるコマ（P05 の連続から外す） |
| `effects` | 配列 | 集中線などの効果（`scripts/data/manpu.json` の `effects` のキーか日本語の別名） |
| `role` | 自由 | 4コマの起承転結などの役割メモ |
| `note` | 自由 | 作者へのメモ |

### 人物（`[[pages.panels.characters]]`）

| キー | 意味 |
|---|---|
| `id` | 人物表の ID（無いと E04） |
| `pos` | `right` `center` `left`（日本語の 右・中央・左 も可）。省略時は書いた順に右から |
| `pose` | ポーズ（英語。krea の `pose` に入る） |
| `pose_ja` | 読者用の台本に出す日本語のポーズ（任意） |
| `expression` | 表情（英語推奨。画像生成のキャプションに足される） |
| `manpu` | 漫符の配列（`manpu.json` の `marks` のキーか別名）。1 人 2 個まで（M02）。意味は `manpu.md` |
| `outfit` / `outfit_en` | 服装の名前と英語の記述。同じ場面で変わると X01。着替えなら `outfit_change = true` |
| `box` | コマの中の相対座標 `[x1, y1, x2, y2]`（0〜1）。推定の配置を上書きする |

### 文字（`[[pages.panels.text]]`）

| キー | 意味 |
|---|---|
| `type` | `dialogue`（台詞・既定）`shout`（叫び）`whisper`（ささやき）`thought`（心の声）`monologue`（モノローグ）`narration`（ナレーション）`sfx`（擬音・描き文字）`aside`（欄外の小文字・手書きのつぶやき）`sign`（作中の看板・画面・手紙の文字） |
| `speaker` | 人物の ID。人物が 2 人以上いるコマでは必ず書く（B09） |
| `text` | 文字。`／` か改行で、吹き出しの中の行を分ける |
| `lines` | 行の配列（`text` の代わり） |
| `offscreen` | `true` で画面外の声。話者がコマに描かれていなくても B07 を出さない |
| `at` | 擬音・作中文字の置き場所 `left` `right` `top` `bottom` `center` |
| `box` | コマの中の相対座標（0〜1）。推定を上書き |
| `balloon` | 吹き出しの形のメモ（写植の一覧に出る。例: 「電話」「二重線＝回想」） |

吹き出しの中の行は、縦書きで右から左へ読まれる。行の区切りは文節で（`dialogue-and-lettering.md`）。句点は付けないのが写植の普通（N03）。

## 構造の検査（E 群）

`name_lint.py` は配置を計算する前に、型と範囲を確かめる。ここで誤りのあるページは配置を計算せず、`name_render.py` と `name_export.py` は何も書き出さずに終了する（終了コード 2）。

| 番号 | 内容 |
|---|---|
| E08 | `bbox` が 0〜1000 の数 4 つでない、または配置の結果が壊れた矩形になった |
| E10 | ページ番号が 1 以上の整数でない、または重複している（書き出しで上書きされる） |
| E11 | 型の誤り（`tiers` が正の数の配列でない、`tier` が 0 以上の整数でない、`width`・`height` が正の数でない、`bleed` の辺の名前が違う、配列であるべきものが配列でない） |
| E12 | 人物・文字の `box` がコマの中の 0〜1 の数 4 つでない |
| E06 | 文字の `type` が語彙に無い |
| E07 | `tiers` の数が、使われている段より少ない |

`meta` が表でない、`characters` が表の配列でない、人物の `id` が無い・重複している、のような最上位の誤りは、読み込みの時点で入力エラー（終了コード 2）になる。配置を計算した後に幅・高さが小さすぎる（0〜1000 の座標で 8 未満）コマも E08。

## 文字の方針の決まり方

画像に文字を描かせるか（`in-image`）、吹き出しだけ描かせて後で写植するか（`blank`）は、`name_export.py --text` ＞ ネームの `meta.text` ＞ `manga.toml` の `[render].text` ＞ 既定（`--to krea` は `in-image`、ほかは `blank`）の順で決まる。

## 書き方の注意

- **`beat` と `visual` を分ける。** `beat` は読者に見える出来事（日本語）、`visual` は画像モデルに渡す英語の画の説明。`visual` に台詞や心情の説明を書いても絵にはならない。描けるもの（人物の位置、物、光、背景）だけにする。
- **人物の見た目は人物表だけに書く。** コマの `visual` に髪型や服を毎回書き直すと、言い回しが揺れて人物が揺れる。コマごとに変えてよいのは `pose` と `expression` と、着替えの `outfit_en` だけ。
- **配置は叩き台。** 人物・吹き出し・漫符の位置は `manga_name.place_panel` の推定で、ネーム画像（`name_render.py`）と krea の書き出し（`name_export.py`）で同じ計算を使う。気に入らない位置は `box` で固定する。
- **ファイルに書いた順が読み順。** 段を組み替えたら、コマの並びも直す。P02 は「書いた順」と「配置から読む順」の食い違い。
- **数え上げは lint が検算する。** 台詞の中で「『ありがとう』の四文字」のように字数を言ったら C01 が確かめる。

## スクリプトとの対応

```bash
python "SKILL_DIR/scripts/name_lint.py" name/ep001.toml --stats
python "SKILL_DIR/scripts/name_render.py" name/ep001.toml --html renders/ep001.html --md renders/ep001.md
python "SKILL_DIR/scripts/name_render.py" name/ep001.toml --md renders/ep001.reader.md --reader
python "SKILL_DIR/scripts/name_export.py" name/ep001.toml --to krea --out renders/ep001/krea
python "SKILL_DIR/scripts/name_export.py" name/ep001.toml --to panels --out renders/ep001/panels.md
python "SKILL_DIR/scripts/name_export.py" name/ep001.toml --to page-prompt --out renders/ep001/pages.md
python "SKILL_DIR/scripts/name_export.py" name/ep001.toml --to lettering --out renders/ep001/lettering.csv
python "SKILL_DIR/scripts/name_export.py" name/ep001.toml --to sections --out renders/ep001/sections.md
python "SKILL_DIR/scripts/name_export.py" name/ep001.toml --to compose --out renders/ep001/compose.json
python "SKILL_DIR/scripts/name_convert.py" name/ep001.json --out name/ep001.toml
```

例は `assets/templates/name.toml`（横開き 2 ページ）と `assets/templates/yonkoma.toml`（4コマ 1 本）。
