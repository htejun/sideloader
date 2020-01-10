#!/bin/bash

set -xe
cd "$(dirname $0)"

mkdir -p io-bomb-dir-src
(cd io-bomb-dir-src; tar xf ../linux.tar.gz)

for ((r=0;r<32;r++)); do
    for ((i=0;i<32;i++)); do
	cp -fR io-bomb-dir-src io-bomb-dir-$i
    done
    for ((i=0;i<32;i++)); do
	true rm -rf io-bomb-dir-$i
    done
done

rm -rf io-bomb-dir-src
