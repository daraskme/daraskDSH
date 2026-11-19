# レビューの手順（ネームを別の目で読む）

- いつ読むか: ネームを書き上げ、lint の処置を終えた後。読切・連載の各話。外部のモデル（Codex CLI など）に見せるとき。
- いつ読まないか: ネームを書いている最中。4コマ 1 本や一場面だけの依頼（読者役 1 回で足りる。編集役は省いてよい）。

自分で書いたネームを自分で採点しない。書いた本人は意図を知っているので、紙面に無いものまで読めてしまう。レビューは、作者との会話も設計資料も引き継がない別のコンテキストに任せる。

## 1. 流れ

```
- [ ] name_lint.py の指摘をすべて確かめて処置する（FAIL も WARN も、誤りは fix、誤検出・意図は keep と理由、迷うものは要確認。SKILL.md「lint の読み方」）
- [ ] 読者用の台本を作る: name_render.py --md reviews/epNNN.reader.md --reader
- [ ] （任意）ネーム画像: name_render.py --html renders/epNNN.html（画像を読める読者役に、PNG にして渡す）
- [ ] 読者役（roles/reader.md）: 読者用の台本だけを渡す
- [ ] 編集役（roles/editor.md）: 全情報の台本、話の筋、契約、読者のレポート
- [ ] 校閲役（roles/proofreader.md）: 台本、人物表、canon、lint の結果
- [ ] 指摘を裁く（下の §4）→ ネームを直す → lint → 直す前の版と比べ、負けたら戻す
```

ネーム画像を PNG にするには、Chrome があれば `google-chrome --headless=new --screenshot=out.png --window-size=1100,1600 file:///.../epNNN.html`。無ければ台本だけでよい。

## 2. 委任文の型

委任文はこの型だけにする。作者の意図、苦労した点、「ここを見てほしい」は書かない（意図を知った読者は「伝わった」と錯覚する）。

```
最初に SKILL_DIR/references/roles/reader.md を読み、その指示に従うこと。
対象: <reviews/epNNN.reader.md のパス>（とネーム画像 <PNG のパス>）
（連載なら）前話までのあらすじ: <canon/summaries/ のパス>
上記以外のプロジェクト内ファイルを読まないこと。
出力先: <reviews/epNNN.r1.reader.md>
```

編集役・校閲役も同じ型で、読むファイルだけを変える。引き継ぎの無いことを確かめられない環境では「盲検」と呼ばず、「情報を制限したレビュー」と記録する。

サブエージェントが使えない環境では、台本ファイルを読み直してから、役を一つずつ演じる。効果は落ちる。

## 3. 外部のモデルに見せる（Codex CLI など）

同系のモデルどうしは盲点を共有する。別系統のモデルに見せる価値は高い。

```bash
mkdir -p work/ext/ep003
cp reviews/ep003.reader.md work/ext/ep003/
cp SKILL_DIR/references/roles/reader.md work/ext/ep003/
# request.md に §2 の型で依頼を書く（パスは work/ext/ep003 の中のものに）
codex exec -m <モデル名> -s read-only --skip-git-repo-check -C work/ext/ep003 \
  -o reviews/ep003.r1.ext-codex.md - < work/ext/ep003/request.md
```

- 渡すディレクトリには、渡したいファイルだけを複製する（プロジェクト全体を見せると、設計資料まで読まれる）。
- フラグは CLI の版で変わる。初回は `codex exec --help` で、上の指定が使えるか確かめる。
- 返ってきた指摘も §4 の基準で裁く。外部だからという理由で優先しない。

## 4. 指摘の裁き方

- 読者役の反応（読み順に迷った、誰の台詞か迷った、説明されたと感じた、先が読めた）は、事実として受け取る。理由づけは編集役の仕事。
- 読者役の再話を、話の筋の `reader_should` と突き合わせる。再話に出てこなかった項目は「確認候補」で、すぐ未達とは決めない。重要なものは、意図を明かさない追加の問いで確かめる。
- 編集役の方向は、採る前に設定・既出事実（canon）と矛盾しないか照らす。新しい人物・過去・設定を足す方向は、作者に確かめてから。
- 好みの指摘（S4）は、作者が採らなくてよい。
- 改稿は良くなるとは限らない。直す前に守る長所（読者が「読み返した（良くて）」と書いたコマ）を決め、直した後は前の版の読者用台本と比べ、負けたら戻す。

## 5. 記録

```
reviews/ep003.reader.md          読者用の台本（レビューに渡した版）
reviews/ep003.r1.reader.md       読者役の返答
reviews/ep003.r1.editor.md       編集役の返答
reviews/ep003.r1.ext-codex.md    外部の返答
reviews/ep003.r1.decisions.md    指摘ごとの採否と理由
```
