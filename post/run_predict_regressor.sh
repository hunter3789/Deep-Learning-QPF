#!/bin/bash

start="00:00 2024-06-01"
end="00:00 2024-08-31"
increment="+24 hours"

while (( $(date -d "${start}" "+%s") <= $(date -d "${end}" "+%s") ))
do
    mkdir -p /home/cloud-user/clee/post/output/$(date -d "${start}" "+%Y%m")/$(date -d "${start}" "+%d")
    echo $(date -d "${start}" "+%Y%m%d%H") $fhr
    #python predict_regressor_nc.py --case $(date -d "${start}" "+%Y%m%d%H") --epoch 44
    python predict_regressor.py --case $(date -d "${start}" "+%Y%m%d%H") --epoch 44

    start=$(date -d "${start} ${increment}" "+%H:%M %Y-%m-%d")
done
