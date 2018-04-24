#!/bin/bash
python3 tfkeras_gesture_recognizer.py -n lenet -lenet ../models/tfkeras_lenet.h5 -l ../models/labels.txt --width 28 --height 28
