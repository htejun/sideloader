#!/bin/bash

cd /root/sideload
rm -rf linux
tar xf linux.tar.gz
cd linux
make allmodconfig
make -j16
