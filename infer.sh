#!/bin/bash


VERSION=v0
DEVICE=cuda:0

DIR_IN=/home/data/kbh/lip/LRS3/trainval
DIR_OUT=/home/data/kbh/lip/LRS3/${VERSION}_trainval
python src/inference.py -c ./config/${VERSION}.yaml --default ./config/default.yaml --chkpt /home/nas/user/kbh/AE_lip/chkpt/${VERSION}/bestmodel.pt  -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}


DIR_IN=/home/data/kbh/lip/LRS3/test
DIR_OUT=/home/data/kbh/lip/LRS3/${VERSION}_test
python src/inference.py -c ./config/${VERSION}.yaml --default ./config/default.yaml --chkpt /home/nas/user/kbh/AE_lip/chkpt/${VERSION}/bestmodel.pt  -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}
