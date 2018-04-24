#!/bin/bash
python3 chainer_gesture_recognizer.py -n lenet -lenet ../models/chainer_lenet.npz -l ../models/labels.txt --width 28 --height 28
