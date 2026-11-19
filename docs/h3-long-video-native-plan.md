# MiniMax H3 長尺動画のネイティブ化計画

方針: 速度 ≥ 品質。ComfyUI-H3-Long-Video（ooizj、GPL-3.0。コードは写さず、手順だけ合わせる）と同じ「区間を順に生成して latent で繋ぐ」方式を、既存の `plugins/dsh-minimax-h3/native/` に足す。ComfyUI 本体の Context Windows（1 回のサンプリングで窓をずらす）は採らない（窓ごとに全 step を回すので速くならず、実装も大きい）。

## 方式（ooizj と同じ数値）

- 区間の長さ: 既定 362 フレーム（約 15 秒、H3 の学習範囲の上限）。重なり 22 フレーム（latent 7 step）。
- 1 区間目: 今の T2V / I2V / ref と同じ。
- 2 区間目以降: 前の区間のサンプラー出力（正規化済みの latent）の末尾 7 step を、オフセット [0, 1, 5, 9, 13, 17, 18] の 7 個の 1-step keyframe として先頭に固定する（`resolved_frame_index` がそのオフセット）。
- 音声: 前の区間の音声 latent の末尾 40 step（1 秒、40 Hz）を音声 keyframe にする。位置は `end_frame − 24`（`end_frame = 22 + overhang / (5/3)` を 5/3 刻みに丸める。overhang = 音声 step 数 − 5/3 × フレーム数、±0.5 外は 0）。
- seed は区間ごとに seed + i。sigmas・step は同じ。プロンプトは全区間で同じ（区間ごとのプロンプトは今回入れない）。
- 繋ぎ: 2 区間目以降はデコード後の先頭 22 フレームと先頭 round(22/24 × sr) サンプルを捨て、音声は残りフレーム数ちょうどの長さに切る／ゼロ詰め。クロスフェードはしない。
- 区間割り: 残りフレーム + 重なりが「区間長 + 24」以内なら最後の区間をその長さ（17k+5 に切り上げ）にする。合計は要求より最大 16 フレーム長くなりうる。

## 変更

1. `dit.cpp`: `dit_forward` に `cond_audio_rows`（任意）を足す。並びは text | cond（映像）| cond_audio | audio | video。cond_audio の時刻は max(t_a, 1.0)、タグは音声、`audio_patch_proj` は [cond_audio | target audio] の 1 回の GEMM。SLA の範囲は今の式のまま（sink は video の開始まで、dense query は target 音声の開始から）。
2. `sampler.conditioning` / `sample`: 音声 keyframe（フレーム位置、音声 latent）を受け、位置表（`_audio_grid`）と行を作る。音声の条件行にはノイズを混ぜない（AUDIO_COND_TIMESTEP = 1.0）。
3. `generate.py`: `seconds`（合計秒数）が区間長を超えたら区間ループ。Qwen と DiT は 1 回だけ読み込む。区間ごとにデコード → 切り詰め → MP4 へ逐次書き込み（全フレームをメモリに持たない）。
4. `jobs.mjs`・パネル・`video_generate`: 長さの上限を 60 秒に上げる（区間長を超えたら自動で長尺）。

## 確認

- ComfyUI の基準: 公式の `MiniMaxH3AddGuide` で、latent（7 step）と音声 latent を持つ keyframe を入れた 1 区間を capture し、step 0・1 をビット一致で照合する（ooizj のノードは入れない）。
- 通し: 30 秒・60 秒を生成し、繋ぎ目の映像と音声を目で確認、時間を計る。

## レビューの反映と結果（2026-11-19）

- Astra のレビュー（`docs/h3-long-video-astra-review.md`）を反映: ref ありのときの行の順序（映像 keyframe → 条件音声 → ref 画像、`dit_forward` の `cond_audio_at`）、音声位置は 40 Hz の整数座標で丸める、区間は 362 フレーム以内で同じ長さに揃える（末尾 396 は使わない）、latent は CPU に置き DiT を解放してから区間ごとにデコードして MP4 へ逐次書き込み、先頭画像は第 1 区間・末尾画像は最終区間の最後のフレーム、ref 画像は全区間、SLA の統計は区間ごとにリセット。
- 区間割り: 必要な区間数 n = ceil((F − 22) / 340)、各区間を同じ長さに揃える。702 / 1042 / 1382 フレームは 362 × n でちょうど割り切れる。
- 照合（`test_cond_dit_parity.py` の `cond-h3-long`）: ComfyUI に 1-step の映像 keyframe 7 個・音声 keyframe 1 個・ref 画像 2 枚を入れた基準（採取用ノード `DSHCaptureGuide`、`capture_reference.py`）と、位置表と step 0・1 の映像・音声出力がビット一致。既存の T2V・サンプラー・I2V・ref・P-H3 もビット一致のまま。
- 速度（1280×704、2 回目以降、GPU が空いた状態、読み込み込みのジョブ全体）:

| 長さ | 区間 | 生成 | 合計 |
|---|---|---:|---:|
| 5 秒（124） | 1 | 37 s | 85 s |
| 15 秒（362） | 1 | 134 s | 205 s |
| 18 秒（640×384、430） | 2 × 226 | 43 s | 82 s |
| 61 秒（1440 → 1467） | 5 × 311 | 667 s | 837 s（約 14 分） |

- 60 秒のピーク VRAM は 72.4 GB（96 GB 中）。映像 61.125 s・音声 61.117 s。繋ぎ目のフレーム差分は通常のフレーム間と同程度で、人物・服装・場面は最後まで一貫。
