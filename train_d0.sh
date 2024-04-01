#!/bin/bash

DEVICE=cuda:0


VERSION=v0
python src/train.py -c config/${VERSION}.yaml --default config/default.yaml -v ${VERSION} -d ${DEVICE}


#python src/train.py -c config/${VERSION}.yaml --default config/default.yaml -v ${VERSION} -d ${DEVICE} --chkpt <TODO:root>/chkpt/${VERSION}/bestmodel.pt -s 0
