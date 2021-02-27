#!/usr/bin/env bash

EX_SCRIPT=$1
EX_PREFIX=${EX_SCRIPT::-3}
EX_TIMESTAMP=`date +%Y%m%d%H%M%S`
bash ${EX_SCRIPT} > ${EX_PREFIX}_${EX_TIMESTAMP}.log 2>&1
