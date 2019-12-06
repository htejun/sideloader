#!/bin/bash

cd /root/sideload
rm -rf linux
tar xvf linux.tar.gz
cd linux
make allmodconfig
make -j32
