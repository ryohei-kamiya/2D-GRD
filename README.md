# 2D-GRD
## 概要
2D-GRD(2 Dimentional Gesture Recognition Dataset)は、一つなぎの平面上の点列で表現される時系列のデータセットです。マウスのドラッグ操作で26文字のアルファベットに似た形（ジェスチャーと呼ぶ。）を一筆書きし、一筆書きの開始点から終了点までの軌跡を記録して作成しました。  
各ジェスチャーには、形状が類似するアルファベットをクラス名として割り当てています。26文字のアルファベットに対応する26個のクラスがあり、1クラスあたり10,569個のジェスチャーパターンが含まれます。各クラスのパターンは、オリジナルの10パターンと、オリジナルを非線形に変換して生成した10,559個のパターンが含まれます。

- ジェスチャーパターン
	- データアーカイブ：https://goo.gl/BEC1U3 ( 左記のURLからダウンロードしてdata/generated/に展開してください )
	- クラス数：26クラス
	- 1クラスあたりデータ数：10,569パターン
	- 1クラスあたりオリジナルデータ数：10パターン ( data/original/points に格納 )

上記のジェスチャーパターンから、以下のデータセットを作成しました。  

- 点列データセット:
	- data/generated/grd-points-training-[sml].csv
	- data/generated/grd-points-validataion-[sml].csv
	- data/generated/grd-points-test-[sml].csv
- 画像データセット:
	- data/generated/grd-image-training-[sml].csv
	- data/generated/grd-image-validataion-[sml].csv
	- data/generated/grd-image-test-[sml].csv
- 規格化済みの点列データセット
	- data/generated/grd-adjusted_points-training-[sml].csv
	- data/generated/grd-adjusted_points-validataion-[sml].csv
	- data/generated/grd-adjusted_points-test-[sml].csv
- 規格化済みの画像データセット
	- data/generated/grd-adjusted_image-training-[sml].csv
	- data/generated/grd-adjusted_image-validataion-[sml].csv
	- data/generated/grd-adjusted_image-test-[sml].csv

[sml]はsまたはmまたはlのいずれかを表しています。sはsmallの略で、データ数を少量に絞ったデータセットを意味します。
m、lの順にデータ数が多くなり、lは全てのデータを含みます。また上記説明における「規格化」は、中心座標を(127,127)として、縦横が0～255に収まる最大サイズに拡大縮小し、点の数を64点にサンプリング(アップサンプリングまたはダウンサンプリング)する操作を意味しており、src/pointsbuffer.pyのadjust()関数を使用しています。

## データ拡張方法(ジェスチャーパターン生成方法)
各ジェスチャーパターンは、src/gesture-painter.pyで記録したオリジナルのデータに対して、make-datafiles.pyでhomography変換とガウス関数による空間歪曲を複数回適用して生成しました。具体的な生成方法は各ソースコードをご確認ください。  

- ジェスチャーパターン記録用スクリプト (Python3のTkinterを使用)
	- src/gesture-painter.py
- データ拡張用スクリプト
	- src/make-datafiles.py
- 点列の規格化などの処理用スクリプト
	- src/pointsbuffer.py

## データセット作成方法
src/gesture-painter.pyとsrc/make-datafiles.pyでジェスチャーパターンを生成後、以下のスクリプトによって各データセットを生成しました。詳細は各スクリプトの内容をご確認ください。

- 点列データセット生成スクリプト
	- src/make-points-dataset.sh
- 画像データセット生成スクリプト
	- src/make-image-dataset.sh
- 規格化済みの点列データセット生成スクリプト
	- src/make-adjusted_points-dataset.sh
- 規格化済みの画像データセット生成スクリプト
	- src/make-adjusted_image-dataset.sh

## データセット利用例
これらのデータセットの利用例として以下のサンプルスクリプトを作成しました。

- ソニーのNeural Network Librariesを使ったジェスチャー認識スクリプト
	- src/gesture-recognizer.py
- Neural Network Librariesによる多層パーセプトロンの実装
	- src/mlp.py
- Neural Network LibrariesによるLeNet(BatchNormalizationを利用した改変版)の実装
	- src/lenet.py
- Neural Network LibrariesによるLSTMの実装
	- src/lstm.py
- Neural Network Librariesから時系列のデータセット(可変長)を読み込むためのユーティリティスクリプト
	- src/timeseries_data.py

詳細はスクリプトの内容をご確認ください。

## 学習済みモデルパラメータファイル・クラスラベルファイル
src/gesture-recognizer.pyから読み込む学習済みのモデルパラメータファイルとクラスラベルファイルは以下にあります。

- 学習済みモデルパラメータファイル
	- models/mlp-parameters.h5 (多層パーセプトロン用のモデルパラメータファイル)
	- models/lenet-parameters.h5 (LeNet用のモデルパラメータファイル)
	- models/lstm-parameters.h5 (LSTM用のモデルパラメータファイル)
- クラスラベルファイル
	- labels.txt

なお、これらはあくまで動作確認用のサンプルです。性能は最適化していませんのでご注意ください。

## ランチャースクリプト
上記の各スクリプトは、コマンドラインで与える引数が多く、開発中の動作確認のための入力が煩わしかったため、予め引数を記述したランチャースクリプトを作成しました。手っ取り早く動かしてみたい人向けに、これらのスクリプトも上げておきます。

- 多層パーセプトロンの学習・評価用ランチャースクリプト
	- src/mlp-train.sh (学習用)
	- src/mlp-evaluate.sh (評価用)
	- src/mlp-predict.sh (推論関数の動作確認用)
- LeNetの学習・評価用ランチャースクリプト
	- src/lenet-train.sh (学習用)
	- src/lenet-evaluate.sh (評価用)
	- src/lenet-predict.sh (推論関数の動作確認用)
- LSTMの学習・評価用ランチャースクリプト
	- src/lstm-train.sh (学習用)
	- src/lstm-evaluate.sh (評価用)
	- src/lstm-predict.sh (推論関数の動作確認用)
- 多層パーセプトロンを用いてジェスチャー認識するランチャースクリプト
	- src/run-mlp-gesture-recognizer.sh
- LeNetを用いてジェスチャー認識するランチャースクリプト
	- src/run-lenet-gesture-recognizer.sh
- LSTMを用いてジェスチャーの軌跡を予測し、多層パーセプトロンで認識するランチャースクリプト
	- src/run-mlp-with-lstm-gesture-recognizer.sh

## メンテナンス
フィードバック歓迎します。特にバグがある場合はご連絡いただけるとうれしいです。
