#!/bin/bash
python3 chainer_gesture_recognizer.py -n mlp -mlp ../models/chainer_mlp.npz -l ../models/labels.txt -xl 64
