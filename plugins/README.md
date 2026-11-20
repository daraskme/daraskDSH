# 拡張機能

| フォルダー | 機能 |
|---|---|
| [manga-generation](manga-generation/) | 漫画のネーム、コマ割り、作画、写植、書き出し |
| [image-generation](image-generation/) | Anima・P-H3 の画像生成と参照画像からの生成 |
| [video-generation](video-generation/) | MiniMax H3 の音声付き動画生成 |
| [image-editing](image-editing/) | P-H3 の画像編集。参照画像と編集の指示を受け取る |
| [lora-training](lora-training/) | キャプション作成と Anima・H3・P-H3 の LoRA 学習 |
| [media-shared](media-shared/) | 各拡張機能が使う GPU ジョブキュー、ネイティブ処理と共通 UI |

有効なプラグインは [harness/config/agents.yml](../harness/config/agents.yml) に登録しています。モデルは [models](../models/)、生成物は [outputs](../outputs/) にあります。
