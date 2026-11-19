# 出典とライセンス上の注意

このスキルは、下記の公開資料を調査し、そこから得た考え方・手順・数値の目安を参考にしたうえで、文面・構成・例のすべてを新たに書き下ろしたものである。元資料の文章は転載していない。数値の目安には出典を付け、確認できなかったものは「未確認」と書いた。調査の記録（URL・日付・一次か二次か）は `docs/dev/research-1-ai-manga-pipeline.md`、`docs/dev/research-2-manga-craft.md`、`docs/dev/research-3-legal-platforms.md`。

## 土台

| 資料 | 学んだこと |
|---|---|
| [ja-novel-writing](https://github.com/daraskme/ja-novel-writing)（同じ作者） | 書く前に読むものと書いた後に読むもの（AI 臭のカタログ）の分離、lint を検知器として扱う読み方、別コンテキストの読者役・編集役・校閲役、外部モデルのレビュー、proposed / confirmed の台帳、コンテキストパック、数え上げの検算。役割ファイルの構成はこのスキルの漫画向けに書き直した |
| krea2-json-bbox-ja-manpu-regional-lora（同じ作者のローカルプロジェクト） | `krea_layout_v1` の形式（絶対ピクセル xyxy、norm_1000 のプロンプト、`character_id`・`lora`・`pose`・`speakers`・`balloon_id`）、漫符の kind の分類、検査の規則。`name_export.py` の検査はその `validate_layout` と同じ規則を独立に実装した |

## 漫画の作法（主なもの）

CLIP STUDIO の講座（コマ割り・間白・めくり・WEBTOON）、小学館新人コミック大賞の講座（フキダシ、タチキリ、見開き、背景）、マンナビ（文字数、縦スクロール）、comico 編集部の note（WEBTOON の読まれ方と分量）、シーモア編集部（ネームの見方、感情の変化）、MANZEMI（構図の繰り返し）、かかし朝浩（イマジナリーライン、posfie のまとめ）、元「デザート」編集長の note（ネームで見ること）、Too の漫画講座（ネームで分かるべきこと）、原作者・猪原賽のブログ（ネームの書式）、Wikipedia「漫符」。数値と主張ごとの URL は `docs/dev/research-2-manga-craft.md`。

## AI 漫画の実践知見

Qiita・note・aipicks などの実践談、Google の Nano Banana Pro の発表、StoryDiffusion・DiffSensei・MangaDiffusion・MangaFlow・TaleDiffusion の論文、Krea 2 向け領域別 LoRA ノードの README、既存のツールとスキル（AI Comic Factory、baoyu-comic、LlamaGen のスキル作成ガイド、agent-comic-kit ほか）。URL は `docs/dev/research-1-ai-manga-pipeline.md` と `docs/dev/research-3-legal-platforms.md`。既存スキルのコード・文面は取り込んでいない。

## データ（集計値とコマ枠の座標の出典）

`references/corpus-norms.md`・`references/layout-library.md`・`references/manpu.md` の実測の数字、`scripts/data/layouts.json` のコマ枠の座標、`scripts/manga_name.py` の配置の目安は、下記のデータを手元で集計したもの（2026-11-15）。このリポジトリに入っているのは集計値と、コマ枠の座標（ページを 0〜1000 にした矩形）だけで、画像・台詞・作品名・注釈ファイルそのものは含めていない。集計のスクリプトはリポジトリの外（所有者の手元）にある。

| データ | 提供元・版 | 使ったもの | 条件 |
|---|---|---|---|
| Manga109-s | 東京大学 相澤・山肩・松井研究室。2026-05-21 公開版（注釈 v2021.12.30 と v2026）。https://huggingface.co/datasets/hal-utokyo/Manga109-s | コマ・台詞の枠と文字・顔・体の注釈の集計。コマ枠の座標（型の形） | Manga109-s の利用規約（再配布の禁止、成果の公開時に Manga109-s の利用を明記）。引用: Aizawa et al., "Building a Manga Dataset Manga109 with Annotations for Multimedia Applications", IEEE MultiMedia 27(2), 2020。Matsui et al., "Sketch-based Manga Retrieval using Manga109 Dataset", Multimedia Tools and Applications 76(20), 2017。Baek et al., "Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding", ICML Workshop, 2026 |
| Manga109Dialog | 同研究室。https://github.com/manga109/public-annotations（revision fd20965） | 台詞と話者の対応の集計（`corpus-norms.md` §7.1） | CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）。集計値に加工して使用。引用: Li, Aizawa, Matsui, "Manga109Dialog: A Large-scale Dialogue Dataset for Comics Speaker Detection", ICME 2024 |
| COO（Comic Onomatopoeia） | 同研究室。Manga109-s に同梱の注釈（public-annotations と同じもの） | 擬音の枠と文字の集計（§7.4）、よく出る語の一覧 | CC BY 4.0。集計値に加工して使用。引用: Baek, Matsui, Aizawa, "COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts", ECCV 2022 |
| MangaUB の読者層の分類（genre.csv） | 同研究室。public-annotations（revision fd20965） | 作品ごとの読者層（少年・少女・青年・女性）での集計 | CC BY 4.0。集計値に加工して使用。引用: Ikuta, Wohler, Aizawa, "MangaUB: A Manga Understanding Benchmark for Large Multimodal Models", IEEE MultiMedia, 2025 |
| manga lora の注釈（JSON_BBOX_Lora） | リポジトリの所有者が Manga109-s のページに付けた注釈（漫符・吹き出し・要約） | 承認済み 6,741 ページの集計、型の形 | 元のページは Manga109-s の規約に従う |

## 権利と規定

文化庁「AIと著作権に関する考え方について」（2024-03-15）とチェックリスト&ガイダンス（2024-07-31）、pixiv・BOOTH・Kindle Direct Publishing の告知とヘルプ、新人賞の要項。`references/formats.md` の記載は 2026-11-03 時点の確認内容で、投稿・応募の前に必ず最新の公式情報を確かめること。

## 漫符の分類について

`references/manpu.md` の 5 系統（感情の記号／体から出る効果／動きと画面の効果／擬音／文字の描き方）と、成人向けの節の分類は、リポジトリの所有者が参考資料を分析した整理を元にしている。

## 再配布について

- 文面・構成・例はすべて書き下ろしで、元資料の文章は含めていない。
- このリポジトリには現時点でライセンスを付けていない。再利用の条件はリポジトリの所有者が決める。
- 生成 AI で作った漫画は、投稿先によっては申告が必要で、新人賞によっては応募対象外になる。利用者自身が各規定に従うこと。
