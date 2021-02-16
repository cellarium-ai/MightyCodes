#!/bin/bash

BRANCH=main
BUCKET_PATH=gs://fc-secure-edbbad6f-85d8-45a4-8e64-3f4a0ac501ff/mighty_codes_tarballs
PWD=$(pwd)
TMP_PATH=${PWD}/__tmp__

mkdir -p ${TMP_PATH}
cd ${TMP_PATH}
git clone git@github.com:broadinstitute/MightyCodes.git
cd MightyCodes
git fetch --all
git checkout main
COMMIT_ID=$(git rev-parse --short HEAD)
cd ..
tar cvzf ./MightyCodes-${COMMIT_ID}.tar.gz ./MightyCodes/
gsutil cp ./MightyCodes-${COMMIT_ID}.tar.gz ${BUCKET_PATH}
cd ${PWD}
rm -rf ${TMP_PATH}


