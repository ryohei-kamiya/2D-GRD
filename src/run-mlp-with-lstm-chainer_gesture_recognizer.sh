#!/bin/bash
python3 chainer_gesture_recognizer.py -n mlp-with-lstm -mlp ../models/chainer_mlp.npz -lstm ../models/chainer_lstm_with_baseshift.npz -l ../models/labels.txt -xl 64 -xil 16 -xss 1 -lstmu 32
