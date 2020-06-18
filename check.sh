#!/bin/bash/

ch=$(cat check.txt)

che=$((ch*100))

if [[ $che -lt 85 ]] ;then echo`python3 retrain_mlops.py` ;else echo `exit 0`;fi