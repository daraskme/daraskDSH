# 連載の運用

- いつ読むか: 連載・複数話の読切・同人誌のシリーズを始めるとき、続きの話のネームを切るとき。`manga.toml` のあるプロジェクトで各話を書くとき。
- いつ読まないか: 4コマ 1 本、読切 1 本、一場面だけの依頼。

連載で壊れるのは、絵より先に情報だ。会話の中にしかない決定は、圧縮やセッションの切り替えで消える。決めたことはファイルに書き、次の話はファイルから始める。

## 1. 始めるとき

```bash
python "SKILL_DIR/scripts/init_project.py" my-manga --title "題名" --format page
```

最初に埋めるのは 4 つだけ。ほかは話を進めながら。

1. `manga.toml`: 形式、1 話のページ数、rating、`[contract].promise`（読者への約束）。
2. `plot/logline.md`: 一言で言えるか、主人公の感情がどう変わるか（`story.md`）。
3. `characters/cast.toml`: 主要人物の `appearance`（画像生成の固定文面）と `voice`（`characters.md`）。
4. `style/directing.md`: コマ数・吹き出し・漫符の量の傾向（`paneling.md` `dialogue-and-lettering.md` `manpu.md`）。

質問は、答えで骨格が変わり後から安く直せないもの（形式、ページ数、rating、結末の方向）に限り、最大 3 問。

## 2. 1 話ごとの流れ

```
- [ ] STATUS.md を読む
- [ ] build_context.py --project . --episode N --out renders/ctx-epNNN.md を作り、パックだけを読む
- [ ] plot/episodes/epNNN.md に話の筋（場面の並び、引き、reader_should）を書く
- [ ] name/epNNN.toml を書く（1 回に書くのは数ページ。1 話を一度に出さない）
- [ ] 保存のたびに name_lint.py
- [ ] name_render.py で台本とネーム画像を作り、通して読む
- [ ] レビュー（review-protocol.md）→ 直す
- [ ] canon を更新: 新しく出た事実を facts.md に proposed で追記、canon/summaries/epNNN.md に要約（1 行目は「状態: proposed」）
- [ ] 要約と proposed をユーザーに短く示し、認められたものだけ confirmed に（要約は 1 行目を「状態: confirmed」に書き換える）
- [ ] decisions.md に決定と却下を追記
- [ ] STATUS.md を更新
- [ ] （作画に進むなら）image-generation.md
```

## 3. canon（既出の事実）

- ネームに LLM が書き足した事実（人物の過去、家族、住所、能力の制限、物の在りか）は `proposed`。作者が認めて初めて `confirmed`。`build_context.py` は confirmed だけを次の話に渡す。
- 事実には、出た場所（ページ-コマ）を書く。後で絵と台詞を確かめられるように。
- 見た目の事実（傷、髪を切った、服を替えた）も canon に書き、人物表の `appearance` を更新するか、話ごとの `outfit_en` にする。見た目の変化を人物表に反映し忘れると、画像生成で前の姿に戻る。

## 4. 要約

`canon/summaries/epNNN.md` は 200 字程度。1 行目に「状態: proposed」か「状態: confirmed」を書く。`build_context.py` は confirmed の要約だけを次の話の前提に入れる（要約に書いた事実も、承認前は前提にしない）。何が起きたか、誰が何を知ったか、終わりの状態（誰がどこで何をしているか、引き）。絵の説明は要らない。次の話の冒頭は、この終わりの状態から始める。

## 5. 決定ログ

却下した案は理由とともに `decisions.md` に残す。LLM は同じ案を何度も出す。ログにある案は再提案しない。

## 6. 再開するとき

1. `STATUS.md` → `decisions.md` の直近 → 前話の要約とネームの最終ページ。
2. ネームの声（台詞の癖、コマ数の傾向）は、前話の台本（`name_render.py --md`）を数ページ読んで取り戻す。
3. 未確認の proposed があれば、書く前にユーザーに確かめる。
