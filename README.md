# 2D-GRD
## 概要
ソニーのNeural Network Console / Libariesを使って二次元のジェスチャー認識／予測を行うトイ・プログラムです。
本リポジトリは以下のプログラム（Python3スクリプト）を含みます。

- ２次元のジェスチャー（一筆書きの点列データ）を記録するプログラム
	- src/gesture_painter.py
- 記録したオリジナルのジェスチャーを非線形の幾何変換でデータ拡張するプログラム
	- src/make_datafiles.py
- ソニーのNeural Network Librariesを使うシンプルなジェスチャー認識器／予測器の学習・評価用プログラム
	- src/mlp.py                  # 多層パーセプトロン
	- src/lenet.py                # LeNet(BatchNormalizationを利用した改変版)
	- src/lstm.py                 # LSTM
	- src/lstm_with_baseshift.py  # LSTM版ジェスチャー予測器作成・評価プログラム(現時刻の座標を(0,0)にシフト)
	- src/delta2_lstm_trainer.py  # ジェスチャー予測器の学習プログラム(lstm_with_baseshift.pyから学習部分だけ抜き出したもの)
- 学習したジェスチャー認識器／予測器を使ってジェスチャー認識・予測をするプログラム
    - src/delta2_mlp_gesture_recognizer.py            # MLPでジェスチャーを認識するプログラム
    - src/delta2_mlp_with_lstm_gesture_recognizer.py  # LSTMでジェスチャーを予測し、MLPで認識するプログラム
	- src/gesture_recognizer.py                       # MLP、LeNet、LSTM各々を切り替えて使えるジェスチャー認識・予測プログラム

## データ拡張方法(ジェスチャーパターン生成方法)
src/gesture_painter.pyで記録したオリジナルのデータに対して、make_datafiles.pyでhomography変換とガウス関数による空間歪曲を複数回適用して拡張します。具体的な生成方法は各ソースコードをご確認ください。  

## データセット作成方法
src/gesture_painter.pyとsrc/make_datafiles.pyでジェスチャーパターンを生成後、以下のスクリプトによって各データセットを生成します。詳細は各スクリプトの内容をご確認ください。

- 点列データセット生成スクリプト
	- src/make-points-dataset.sh
- 画像データセット生成スクリプト
	- src/make-image-dataset.sh
- 点の数が一定の点列データセット生成スクリプト
	- src/make-sampled_points-dataset.sh

## 学習済みモデルパラメータファイル・クラスラベルファイル
ジェスチャー予測・認識プログラムの動作確認用の学習済みモデルパラメータファイルとクラスラベルファイルは以下にあります。
なお、これらはあくまで動作確認用のサンプルです。性能は最適化していません。

- 学習済みモデルパラメータファイル
	- models/mlp-parameters.h5 (多層パーセプトロンのモデルパラメータファイル)
	- models/lenet-parameters.h5 (LeNetのモデルパラメータファイル)
	- models/lstm-with-baseshift-parameters.h5 (LSTMのモデルパラメータファイル)
- クラスラベルファイル
	- labels.txt

## ランチャースクリプト
以下は、上記の各スクリプトのコマンドライン引数を記述したランチャースクリプト（bash シェルスクリプト）です。

- 多層パーセプトロンの学習・評価用ランチャースクリプト
	- src/mlp-train.sh (学習用)
	- src/mlp-evaluate.sh (評価用)
	- src/mlp-infer.sh (推論関数の動作確認用)
- LeNetの学習・評価用ランチャースクリプト
	- src/lenet-train.sh (学習用)
	- src/lenet-evaluate.sh (評価用)
	- src/lenet-infer.sh (推論関数の動作確認用)
- LSTMの学習・評価用ランチャースクリプト
	- src/lstm-train.sh (学習用)
	- src/lstm-evaluate.sh (評価用)
	- src/lstm-infer.sh (推論関数の動作確認用)
    - src/lstm-with-baseshift-train.sh（lstm_with_baseshift.pyの学習処理を実行）
    - src/run-delta2-lstm-trainer.sh（delta2_lstm_trainer.pyを実行）
- 多層パーセプトロンを用いてジェスチャー認識するランチャースクリプト
    - src/run-delta2-mlp-gesture-recognizer.sh
	- src/run-mlp-gesture-recognizer.sh
- LeNetを用いてジェスチャー認識するランチャースクリプト
	- src/run-lenet-gesture-recognizer.sh
- LSTMを用いてジェスチャーの軌跡を予測し、多層パーセプトロンで認識するランチャースクリプト
    - src/run-delta2-mlp-with-lstm-gesture-recognizer.sh
	- src/run-mlp-with-lstm-gesture-recognizer.sh

## メンテナンス
フィードバック歓迎します。特にバグがある場合はご連絡いただけるとうれしいです。
