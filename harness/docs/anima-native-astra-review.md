# Anima ネイティブ計画 v2 の敵対的レビュー（gpt-6.2-astra, effort high, OpenAI Responses API）

**C++/CUDA＋既存の Rust は継続でよいです。** ただし、78 ms の内訳が出る前に「Graphs が第1候補、FP8 が主要レバー」と決める根拠はありません。とくに、現行コードのまま optimizer を capture することと、凍結重みの保存 dtype をそのまま FP8 に変えることは、学習結果を壊します。

1. **【高】45 ms は下限として使えず、Graphs の改善余地とも整合していない。**  
   78→50 ms には **28 ms、現状の36%** を削る必要があります。仮にクリティカルパスの95%が GPU カーネル実行なら、ホスト起動の隙間を全部消しても約4 msです。また、現行の step イベントは `ends[slot]` の**後**の統計回収・ログを含みません。  
   **最小修正：** ウォームアップ後の連続 step で GPU の空白時間を測り、同じ区間の壁時計も取る。45 ms と「1024px で概算下限の1.2倍」は合否判定から削り、実測の ms/枚と完走時間で判定する。

2. **【高】batch は Graphs より先に試す。ただし画像/秒だけで既定値を決めない。**  
   例えば B=4 が 160 ms/step なら 40 ms/枚で目標達成に見えますが、同じ200更新なら 15.6秒→32秒です。同じ枚数に揃えれば今度は更新回数が1/4になり、同じ品質に達する保証はありません。  
   **最小修正：** まず eager の B=1/2/4 で候補を絞る。B=1 のレイテンシと B>1 の ms/枚は別記し、既定値は同じ処理枚数で既存の出力品質ゲートを通った構成の完走時間で決める。

3. **【致命】optimizer まで素直に capture すると、AdamW の step が進まない。**  
   `adamw_step(..., step + 1)` のホスト呼び出しは replay されません。そこで step から作った補正係数を `adamw_update` に値渡しすれば、同じ係数で全更新が走ります。`statistics[slot]` の書き先も capture 時のままです。なお、ループ内で同期しているのは `.item()` ではなく、ログ時または最大32 step ごとの `.to(kCPU)` です。  
   **最小修正：** 最初は forward＋backward＋固定バッファへの grad 回収・norm までを capture。既存の AdamW 1 launch と統計書き込みは graph 外に残し、実際の `step + 1` と `slot` を渡す。

4. **【高】graph の単位は解像度ではなく `(B,H,W,L)`。特に context 長 `L` を丸めると結果が変わる。**  
   現行の cross-attention には padding mask がないため、短い条件をゼロ埋めしても余分な位置が softmax の分母に入ります。また、同じ形状でもサンプルのポインタを差し替えるだけでは、replay は capture 時のアドレスを読み続けます。  
   **最小修正：** 実際に頻出する完全一致形状だけ capture し、残りは eager。同じ固定入力バッファに毎回の latent・context・time・noise を入れ、graph 数削減のための padding はしない。

5. **【高】「backward が efficient なら cuDNN を強制」は削る。**  
   efficient は math fallback と同義ではありません。SM 12.0 という情報だけでは、使用中の LibTorch/cuDNN が、その stride・head dim・系列長で学習用の実行プランを持つとは決まりません。強制すると backward で失敗するか、短い cross-attention だけ遅くなります。  
   **最小修正：** 実機の self/cross を分け、実際の形状で **forward＋backward の合計**を比較。自動選択より速く、両方完走した箇所だけ固定する。

6. **【高】融合を書く前に、self Q/K/V の同一入力を6回 BF16 化するのを止める。**  
   `modulated_norm` は FP32 を返し、`linear` は凍結 W 用と LoRA A 用に別々に `x.to(dtype)` を呼びます。self Q/K/V がすべて LoRA 対象なら、同じ `[B,S,2048]` の変換が **6回→1回、28 block で forward だけで140回減**です。Graphs にしてもこの読み書きは消えません。  
   **最小修正：** BF16 経路では self Q/K/V の分岐前に一度だけ cast して共有する。他の `linear` でも W と A に同じ cast 結果を渡し、既存の勾配パリティで確認する。

7. **【高】「LoRA A を W に連結」は削る。束ねるなら凍結 QKV。**  
   通常の autograd で `cat(W,A)` を `linear` に渡すと、A が学習対象なので連結重み全体の勾配が計算されます。2048² の BF16 投影なら、rank 8 の A 勾配のために約8 MiBの不要な W 勾配部分と、その dense GEMM が発生します。  
   **最小修正：** A/B は独立のまま。投影が実測上重ければ、凍結 self QKV と cross KV だけロード時に連結する。これなら forward の凍結 GEMM は28 block 合計で84呼び出し減り、W 勾配も不要。

8. **【高】FP8 は「重みを一度変換すれば forward/dX が両方速い」わけではない。**  
   出力チャネル別に `W[o,i]=s[o]Q[o,i]` とすると、forward の `s` は出力側に掛けられますが、`dX=(dY*s)Q` では縮約軸側です。同じ scale 指定で転置するだけでは同じ演算になりません。さらに X と dY の量子化・レイアウト変換を除いた GEMM 単体の数字は、step 短縮を過大評価します。  
   **最小修正：** 最も時間を使う実形状1個で、使用中の SM 12.0/cuBLASLt が受理する scale・配置を forward/dX **両方向**で確認。まず per-tensor で量子化込みの合計を BF16 と比較し、勝たなければ全層展開も層別フォールバックも削る。

9. **【致命】現行 `linear` は「凍結 W だけ FP8」という契約を守れない。品質ゲートも方向誤差だけでは足りない。**  
   `dtype = weight.scalar_type()` のまま W を FP8 にすると、X だけでなく `a->second.to(dtype)` と `b->second.to(dtype)` まで FP8 になります。これは予定外の精度低下か未対応演算です。また、勾配が参照の10倍でも cosine は1なので、現行ゲートはクリップや更新量の異常を見逃します。  
   **最小修正：** FP8 化は凍結 GEMM の内部だけに閉じ、出力と LoRA 分岐は現行 BF16 のまま。既存1-step 判定に勾配ノルム比を加え、最終採否は **FP8 学習 LoRA を配布時の BF16 base に載せた**固定条件の出力で決める。

**判定：上記修正を条件に着手可。**