#!/bin/bash

set -xe
TARGET=root@$1

cd "$(dirname $0)"
ssh $TARGET 'mkdir -p sideload'
scp test-job.json test-job.sh linux.tar.gz \
    dlxu-memory-growth.py dlxu-memory-growth.sh dlxu-memory-growth.json \
    $TARGET:sideload
