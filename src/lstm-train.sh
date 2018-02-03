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


python3 lstm.py -b 1000 -r 0.001 -i 10000 -v 100 -j 1 -w 0.0 -c cpu \
  -mon ../tmp -m ../models/lstm-parameters.h5 \
  -dt ../data/generated/grd-points-training-l.csv \
  -dv ../data/generated/grd-points-validation-l.csv \
  -de ../data/generated/grd-points-test-l.csv \
  -p train -xil 32 -xol 16 -xss 16 -cs 2
