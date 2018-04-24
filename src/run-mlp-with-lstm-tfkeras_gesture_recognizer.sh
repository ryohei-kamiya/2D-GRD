#!/bin/bash
python3 tfkeras_gesture_recognizer.py -n mlp-with-lstm -mlp ../models/tfkeras_mlp.h5 -lstm ../models/tfkeras_lstm_with_baseshift.h5 -l ../models/labels.txt -xl 64 -xil 16 -xss 1 -lstmu 32
