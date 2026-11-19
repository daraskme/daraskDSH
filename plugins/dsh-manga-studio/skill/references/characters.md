# 人物設計と見た目の一貫性（characters）

- いつ読むか: 人物表（`characters/cast.toml`、またはネーム内の `[[characters]]`）を作るとき。画像生成で人物の顔・服・髪が揺れたとき。台詞で誰が話しているか分からないと言われたとき。人物を増やすか迷ったとき。
- いつ読まないか: 話の骨格を決めているとき（`story.md`）。コマ割り・構図を決めているとき（`staging.md` `paneling.md`）。
- 漫画の人物は、絵で一目で見分けられ、声（台詞）で聞き分けられる必要がある。AI で描く場合は、それに加えて、毎コマ同じ人物として描かれる必要がある。この三つは別々の作業で、どれか一つが崩れると読者は人物を取り違える。

## 目次

1. シルエットと記号で見分ける
2. `appearance` の固定文面の書き方
3. 服を変える回（`outfit` / `outfit_en` / `outfit_change`）
4. キャラシートと参照画像
5. 領域別 LoRA（`lora` 別名）
6. 声の書き分け（`voice`）
7. 人物を増やしすぎない
8. 名前の付け方
9. 実在の作家・既存のキャラクターを使わない
10. 成人向けの作品の人物
11. 人物表の確認

## 1. シルエットと記号で見分ける

- **影だけで誰か分かるか。** 髪型、帽子、体格、服の形のどれかで、主要な人物同士が影でも区別できるようにする。`cast.toml` の `silhouette` に一句で書く（「頭頂の跳ねた一房」「肩幅の広い長身」「大きなリュック」）。
- **記号を一つずつ持たせる。** 髪型・小物・色の組み合わせで、その人物にしか無い記号を決める（赤いマフラー、丸眼鏡、片耳のピアス）。記号は、`long` や `extreme_long` で人物が小さく描かれても見える大きさのものにする。
- **似た人物を並べない。** 同じ髪型・同じ制服の人物が二人以上いる場面では、記号の違いを強める（片方は髪を結ぶ、片方はジャケットを脱ぐ）。モノクロでは色の違いが消えるので、色だけで区別しない。ベタ（黒髪）・トーン（茶髪）・白（金髪・銀髪）の配分で区別する。
- **表情の幅を決める。** その人物がめったにしない表情（いつも無表情な人物の笑顔）は決め所のために取っておく。`voice` か `note` に「笑顔は決め所だけ」と書く。

## 2. `appearance` の固定文面の書き方

`appearance` は、画像生成のプロンプトに毎回そのまま貼る、人物の外見の固定文面。`name_export.py` は、krea の人物項目の `caption`、コマ別プロンプトの人物ブロック、ページプロンプトの人物の行に、この文面を使う。空のままだと `name_lint.py` が X02 で知らせる。

- **英語で書く。** 画像モデルの多くは英語のほうが指示に従いやすい。日本語の説明は `silhouette` と `voice` に書く。
- **項目ごとに書く。** 順番を決めて、いつも同じ順に並べる。

  | 順 | 項目 | 例 |
  |---|---|---|
  | 1 | 年齢層と性別 | adult woman in her late twenties / teenage boy |
  | 2 | 体格 | tall and slim / short and stocky |
  | 3 | 髪（長さ・形・色の明暗） | short black bob hair / long wavy light hair in a loose ponytail |
  | 4 | 目と顔の特徴 | sharp narrow eyes / round glasses / a mole under the left eye |
  | 5 | 服（いつもの服） | navy school blazer, red ribbon tie, pleated grey skirt |
  | 6 | 小物・記号 | red scarf / large backpack / silver earring on the right ear |

- **長くしない。** 最大でも 1〜2 行。長い文面は、モデルが後ろの項目を落とす。特徴を項目ごとに分けて書くと人物が変わりにくい、という実践談がある（https://note.com/rich_emu7600/n/n267085726c72 ）。
- **毎回同じ文面にする。** 画風ブロックと人物ブロックを全コマに同じ文面で貼り、変えるのは構図・表情・動作・背景だけにする、という実践談がある（https://aipicks.jp/mag/ai-manga-pro-quality-tips ）。アンカーになる文面を途中で変えると別人になる、という報告もある（https://note.com/note_ai_s/n/nf2a1f1847ad6 ）。直したいときは、`cast.toml` を直し、それ以降の全コマを生成し直す。直したことを `decisions.md` に残す。
- **表情とポーズを入れない。** 表情はコマの人物の `expression`、ポーズは `pose` に書く。`appearance` に「smiling」と書くと、泣く場面でも笑う。
- **強い表情語は弱めに。** 顔がブレたら、強い表情語を弱める（「大笑い」より「口角を上げる」）という実践談がある（https://aipicks.jp/mag/ai-manga-pro-quality-tips ）。`expression` は短く具体的に（surprised, wide eyes / slight smile）。
- **固有名詞を入れない。** 既存の作品名・キャラ名・作家名を `appearance` に書かない（§9）。

## 3. 服を変える回（`outfit` / `outfit_en` / `outfit_change`）

- いつもの服は `appearance` に書く。服を変える回（私服、水着、礼服、変身）は、ネームのコマの人物に `outfit`（日本語の呼び名）と `outfit_en`（英語の文面）を書く。`name_export.py` は `outfit_en` を `appearance` の後ろに足す。
- 同じ場面の中で `outfit` が変わると、`name_lint.py` が X01 で知らせる。着替えの場面なら、着替えたコマに `outfit_change = true` を書く。
- 服を変えた場面では、その場面の全コマに同じ `outfit` と `outfit_en` を書く（書き忘れたコマは、いつもの服で描かれる）。
- 服で人物を見分けている作品（制服の色、隊服）で私服回を描くときは、髪型と小物の記号（§1）が残っているかを確かめる。

## 4. キャラシートと参照画像

- **先にキャラシートを作る。** 三面図（正面・横・後ろ）と表情集（喜・怒・哀・驚・照れ・無表情）を一枚にまとめ、`cast.toml` の `reference` にパスを書く。
- **毎回オリジナルの参照から生成する。** 参照画像を使うモデルでは、人物ごとに同じキャラシートを毎回渡す。前のコマの出力を次のコマの参照にしない。伝言ゲームになって少しずつ別人になる、という実践談がある（https://note.com/rich_emu7600/n/n267085726c72 ）。
- **参照にない角度は、別に参照を作る。** 真後ろ、真上、横顔など、キャラシートに無い角度を何度も使うなら、その角度の参照を足す（同上）。
- **理想の一枚を参照に加える。** 狙いどおりに描けた一コマを参照画像に加えると、服や小物の色が安定した、という報告がある（https://note.com/note_ai_s/n/nf2a1f1847ad6 ）。前のコマの出力を連鎖させるのではなく、「認めた一枚」を固定の参照として足す、という区別を守る。
- **モデルの参照枚数を確かめる。** たとえば Nano Banana Pro は参照画像を最大 14 枚使え、最大 5 人の同一性を保てると公式に発表されている（https://blog.google/technology/ai/nano-banana-pro/ 、2025-11-20）。一コマに出す人物が上限を超えないよう、ネームの段階で人数を絞る。モデルごとの違いは `image-generation.md`。
- **AI で設定画を描くとき。** 外見の固定文面（`appearance`）と画風ブロックに、`solo, character reference sheet of one person, full body, standing straight, front view, arms relaxed, neutral expression, plain white background` を足し、ネガティブに `2boys, 2girls, multiple boys, multiple girls, multiple views, turnaround, extra people` を入れる。`solo` が無いと、Anima では 1 枚に 2 人描かれることがあった（AI の検品で見つかった）。2〜4 枚描いて、外見の文面にいちばん合う 1 枚を設定画にする。設定画は、参照生成・編集（P-H3 など）と検品（`visual-check.md` §5）に渡す。
- **キャラシートに無い特徴を後から足さない。** 本文やネームで「左手に傷がある」と書いたら、キャラシートと `appearance` も直す。直さないまま進めると、描かれた絵と設定が食い違う。

## 5. 領域別 LoRA（`lora` 別名）

- 人物ごとに LoRA を持っている場合、`cast.toml` の `lora` に別名を書く。`name_export.py --to krea` は、人物項目に `character_id`（人物の `id`）と `lora`（別名）を書き、台詞の `speakers` をこの別名で結ぶ。LoRA のファイルの場所は、書き出し先の `character_adapters.json`（雛形が出る）に書く。
- **人物の bbox を重ねない。** bbox ごとに LoRA の効きを空間で区切るノードでは、box を重ねないこと、マスクした LoRA は強度 1.5 以上が要ることが多いこと、キャンバスサイズを一致させることが制約として挙げられている（https://github.com/CliffNodes/Krea2-Multi-Character-Lora-Node-with-bounding-box-Scene-and-Outfit-Edit の README）。ネームで二人を密着させる構図（抱擁、取っ組み合い）は、領域別 LoRA と相性が悪い。そのコマは人物の `box` を手で分けるか、一人を手前・一人を奥に置いて重なりを減らす。
- **`pose` と `appearance` を分ける。** 人物ごとの見た目は `appearance`（caption に入る）、そのコマの姿勢は `pose` に書く。混ぜると、LoRA の効きとポーズの指定が衝突しやすい。
- **LoRA の学習元に注意する。** 特定のクリエイターの少量の作品群で LoRA を作り、作風を超えて創作的表現を出力させる目的の追加学習は、著作権法 30 条の 4 の対象外になり得る、とされている（文化庁「AIと著作権に関する考え方について」の要約。`docs/dev/research-3-legal-platforms.md` §1）。人物 LoRA は、自分で描いた・権利を持つ素材から作る。

## 6. 声の書き分け（`voice`）

`voice` には、その人物の台詞の作り方を書く。名前を伏せても、台詞だけで誰が話しているか分かるのが目標。

- **一人称:** 私・あたし・僕・俺・自分・名前呼び。同じ一人称の人物が二人いるなら、ほかの項目で差をつける。
- **語彙の層:** 日常語・若者言葉・専門用語・古風な言い回し・方言。人物の生まれ育ちと仕事から決める。
- **文の長さ:** 言い切りが短い人物と、一文が長い人物。吹き出しの字数の制約（`dialogue-and-lettering.md`）の中でも、短い・長いの差は出せる。
- **敬語の段階:** 誰にでも敬語、目上にだけ敬語、誰にでもため口。相手によって段階が変わる人物は、その変わり方が関係を見せる。
- **言いよどみ方:** 言いさす、黙る、話を逸らす、冗談でごまかす。隠しごとのある場面では、この癖が台詞の代わりになる。
- **語尾キャラに頼らない。** 「〜ですわ」「〜なのだ」のような語尾だけで書き分けると、語尾を取ると誰か分からなくなる。語尾は様式として使ってよい（`[contract] keep`）が、語彙・文の長さ・敬語の段階でも差をつける。
- **関係が変わったら言葉づかいも変える。** 仲が深まった、決裂した、秘密を知った。その後の台詞の敬語の段階や呼び名が変わらないと、関係の変化が紙面に出ない。
- **確かめ方:** ネームを書いたら、`name_render.py --md` の台本から台詞だけを抜き、話者を伏せて読む。誰の台詞か迷ったら、その人物の `voice` のどの項目で差がついていないかを見る。読者役の「話者当て」（`roles/reader.md`）で確かめてもよい。

## 7. 人物を増やしすぎない

- キャラが多すぎないかは、小学館のストーリーの自己チェックの一項目（https://www.clipstudio.net/oekaki/archives/151965 ）。読切で名前と顔が要るのは主要 2〜3 人が目安（目安。`story.md` §8）。
- 画像生成では、一コマの人物が増えるほど崩れやすい。左右の人物の入れ替わり、指定外の人物の出現、モブの消失が報告されている（https://qiita.com/KLaboratory/items/b4472dd691e1c486c892 ）。一コマに 3 人以上を描くなら、`pos` と `box` で全員の位置を指定し、`name_export.py` が出す「Exactly N characters」で人数を固定する。
- 名前の無いモブ（通行人、生徒たち）は人物表に入れず、コマの `visual` に「background crowd of students」と書く。モブに台詞を言わせるときは、`cast.toml` に `id = "mob"` のような人物を一つ作り、`appearance` を地味に固定する。

## 8. 名前の付け方

- 名前は、時代・地域・家庭から付ける。その人物が生まれた年に多かった名前、その土地らしい名字、親がどんな気持ちで付けたか。最初に浮かんだ名前は、誰でも思いつく名前であることが多い。一度疑う。
- 主要な人物同士で、名前の最初の音と長さを変える（美咲と美優は、吹き出しの中で読み間違える）。
- 読みの難しい名前は、初出でルビを振る。ルビは写植で入れる（`dialogue-and-lettering.md`）。
- 異世界・ファンタジーの名前は、その世界の言語の音の規則を一つ決めて揃える（濁音が多い国、長音の多い国）。

## 9. 実在の作家・既存のキャラクターを使わない

- 既存の作品名・キャラクター名・作家名を、`appearance`、`visual`、`[render] style` に入れない。作品名やキャラ名などの固有名詞をプロンプトに入れることは、既存作品に依拠したことを推認させる事情になる、と文化庁の考え方は整理している（`docs/dev/research-3-legal-platforms.md` §1）。
- 既存作品の画像を i2i の入力にしない。特定の作家の画風を名指しで指定しない。画風は「clean ink line art, solid black fills and sparse screentone」のような技法の言葉で書く（`art-direction.md`）。
- 作風・画風が共通しているだけでは侵害にならないが、出力が既存作品の創作的表現に似ていれば問題になる。公開する前に、画像検索などで既存作品と似ていないか確かめる（文化庁のチェックリスト 3-1-5 の要約。`formats.md`）。
- ユーザーが既存作品の二次創作を頼んだ場合は、私的に楽しむ範囲と公開・頒布する範囲で扱いが変わること、投稿先の規定があることを一行で告げ、判断はユーザーに任せる。二次創作でも、生成画像のプロンプトに作家名を入れない。

## 10. 成人向けの作品の人物

- `[work] rating = "adult"` の作品では、登場人物は全員成人とする。`cast.toml` の各人物の `appearance` を `adult woman` `adult man` のように成人であることが分かる語で始め、年齢を書く項目（`note` など）に成人の年齢を明記する。
- 制服や小柄な体格など、未成年に見える要素を組み合わせない。設定上成人でも、絵が未成年に見えれば同じこと。体格・顔立ち・服装で成人に見えるかを、キャラシートの段階で確かめる。
- 実在の人物を成人向けの作品に描かない。

## 11. 人物表の確認

人物表を作ったら、ネームを切る前に一度だけ確かめる。

```
- [ ] 主要な人物同士が、silhouette と記号で区別できる（モノクロでも）
- [ ] appearance が英語の固定文面で、項目の順が揃い、1〜2 行に収まる
- [ ] appearance に表情・ポーズ・固有名詞（既存作品・作家）が入っていない
- [ ] reference にキャラシート（三面図・表情集）のパスがある（画像生成をするなら）
- [ ] lora を使う人物は別名が一意で、character_adapters.json にファイルの場所がある
- [ ] voice に 一人称 / 語彙の層 / 文の長さ / 敬語の段階 / 言いよどみ方 がある
- [ ] 名前と顔の要る人物の数が、形式に対して多すぎない
- [ ] rating = "adult" なら、全員が成人であることが人物表に明記されている
```

`python "SKILL_DIR/scripts/name_lint.py" name/epNNN.toml` を流すと、人物表に無い ID（E04）、`appearance` が空の人物（X02）、場面の中での服の食い違い（X01）が出る。

## 12. 例: 人物表の二人

```toml
[[characters]]
id = "misaki"
name = "美咲"
lora = "misaki"
appearance = "teenage girl, slim, short black bob hair, large calm eyes, navy school blazer, red ribbon tie, pleated grey skirt, small silver hairpin on the left"
silhouette = "肩で切りそろえたボブと、左の小さな髪留め"
reference = "characters/ref/misaki_sheet.png"
voice = "一人称は私。言い切りが短い。敬語は先生にだけ。照れると語尾が消え、話を物の話に逸らす"
wants = "手紙の差出人を、自分で確かめたい"

[[characters]]
id = "kaito"
name = "海斗"
lora = "kaito"
appearance = "teenage boy, tall and lanky, messy light brown hair, sleepy eyes, navy school blazer worn open, white shirt, loose tie, sneakers with red laces"
silhouette = "前を開けたブレザーと、寝ぐせの跳ねた頭"
reference = "characters/ref/kaito_sheet.png"
voice = "一人称は俺。軽口と一文の長い言い訳が多い。肝心なところで黙る。美咲の名前を呼ばず「お前」で通す"
wants = "手紙を、読まれる前に取り返したい"
```

- 二人とも同じ制服だが、髪の形（ボブ／跳ねた頭）、ブレザーの着方（閉じる／開ける）、小物（髪留め／赤い靴ひも）で区別している。モノクロでは、美咲の黒髪はベタ、海斗の明るい髪は白かトーン。
- `appearance` は、年齢層・体格・髪・目・服・小物の順で、表情とポーズを含まない。
- `voice` は一人称のほかに、文の長さ（短い／長い言い訳）、敬語の段階、言いよどみ方（逸らす／黙る）、呼び方で差をつけている。関係が変わる場面では、海斗が初めて「美咲」と呼ぶ、のように呼び方の変化を決め所に使える。
