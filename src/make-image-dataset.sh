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


DATASET='image';

rm_if_exist () {
	if [ -f "$1" ]; then
		rm -rf "$1";
	fi
}

label_num=$((0));
for dname in `ls "./${DATASET}"`; do
	if [ -d "./${DATASET}/$dname" ]; then
		tmp="grd-${DATASET}-${dname}.tmp";
		rm_if_exist "$tmp"
		for fname in `ls "./${DATASET}/${dname}"`; do
			echo "./${DATASET}/${dname}/${fname},${label_num}" >> "$tmp";
		done
		ofile="grd-${DATASET}-${dname}.csv";
		shuf "$tmp" > "$ofile";
		label_num=$((label_num+1));
	fi
done
estmp="grd-${DATASET}-test-s.tmp";
vstmp="grd-${DATASET}-validation-s.tmp";
tstmp="grd-${DATASET}-training-s.tmp";
emtmp="grd-${DATASET}-test-m.tmp";
vmtmp="grd-${DATASET}-validation-m.tmp";
tmtmp="grd-${DATASET}-training-m.tmp";
eltmp="grd-${DATASET}-test-l.tmp";
vltmp="grd-${DATASET}-validation-l.tmp";
tltmp="grd-${DATASET}-training-l.tmp";
rm_if_exist "$estmp";
rm_if_exist "$vstmp";
rm_if_exist "$tstmp";
rm_if_exist "$emtmp";
rm_if_exist "$vmtmp";
rm_if_exist "$tmtmp";
rm_if_exist "$eltmp";
rm_if_exist "$vltmp";
rm_if_exist "$tltmp";
total_labels=${label_num};
for fname in `ls grd-${DATASET}-*.csv`; do
	total_patterns=`wc -l $fname | cut -f1 -d' '`;
	eldata_size=$((total_patterns / 10))
	vldata_size=$((total_patterns * 2 / 10))
	tldata_size=$((total_patterns - eldata_size - vldata_size))
	head -n $eldata_size "$fname" >> "$eltmp";
	head -n $((eldata_size + vldata_size)) "$fname" | tail -n $vldata_size >> "$vltmp";
	tail -n $tldata_size "$fname" >> "$tltmp";
	emdata_size=$((eldata_size / 2))
	vmdata_size=$((vldata_size / 2))
	tmdata_size=$((tldata_size / 2))
	head -n $emdata_size "$fname" >> "$emtmp";
	head -n $((emdata_size + vmdata_size)) "$fname" | tail -n $vmdata_size >> "$vmtmp";
	tail -n $tmdata_size "$fname" >> "$tmtmp";
	esdata_size=$((eldata_size / 4))
	vsdata_size=$((vldata_size / 4))
	tsdata_size=$((tldata_size / 4))
	head -n $esdata_size "$fname" >> "$estmp";
	head -n $((esdata_size + vsdata_size)) "$fname" | tail -n $vsdata_size >> "$vstmp";
	tail -n $tsdata_size "$fname" >> "$tstmp";
done
esdata="grd-${DATASET}-test-s.csv";
vsdata="grd-${DATASET}-validation-s.csv";
tsdata="grd-${DATASET}-training-s.csv";
emdata="grd-${DATASET}-test-m.csv";
vmdata="grd-${DATASET}-validation-m.csv";
tmdata="grd-${DATASET}-training-m.csv";
eldata="grd-${DATASET}-test-l.csv";
vldata="grd-${DATASET}-validation-l.csv";
tldata="grd-${DATASET}-training-l.csv";
shuf "$estmp" > "$esdata"; sed -i "1i x,y" "$esdata";
shuf "$vstmp" > "$vsdata"; sed -i "1i x,y" "$vsdata";
shuf "$tstmp" > "$tsdata"; sed -i "1i x,y" "$tsdata";
shuf "$emtmp" > "$emdata"; sed -i "1i x,y" "$emdata";
shuf "$vmtmp" > "$vmdata"; sed -i "1i x,y" "$vmdata";
shuf "$tmtmp" > "$tmdata"; sed -i "1i x,y" "$tmdata";
shuf "$eltmp" > "$eldata"; sed -i "1i x,y" "$eldata";
shuf "$vltmp" > "$vldata"; sed -i "1i x,y" "$vldata";
shuf "$tltmp" > "$tldata"; sed -i "1i x,y" "$tldata";
rm -f *.tmp
