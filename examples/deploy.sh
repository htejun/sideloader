#!/bin/bash

set -xe
TARGET=root@$1

cd "$(dirname $0)"
ssh $TARGET 'mkdir -p sideload'
scp build-linux.json build-linux.sh linux.tar.gz \
    dlxu-memory-growth.py dlxu-memory-growth.sh dlxu-memory-growth.json \
    io-bomb.sh io-bomb.json \
    $TARGET:sideload
