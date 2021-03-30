#!/bin/bash

source /etc/profile

cd ../src

declare -a exp_ids=("499" "500" "501" "496" "497" "498" "493" "494" "495")
#declare -a exp_ids=("test")

for exp_id in "${exp_ids[@]}"
do
  echo "Running inference on $exp_id ..."
  python3 run_train.py \
          --conf='../conf/config.yaml' \
          --exp_id=$exp_id
done
