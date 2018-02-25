#!/bin/bash
# Copyright 2018 Ryohei Kamiya <ryohei.kamiya@lab2biz.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


python3 delta2_lstm_trainer.py \
  -b 100 -r 0.001 -i 10000 -v 100 -j 1 -w 0.0 \
  -mon ../tmp -m ../models/lstm-with-baseshift-parameters.h5 \
  -dt ../data/generated/grd-points-training-s.csv \
  -dv ../data/generated/grd-points-validation-s.csv \
  -xil 16 -xol 1 -xss 1 -cs 2
