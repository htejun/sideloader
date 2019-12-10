#!/bin/bash

set -xe
yum -y install openssl-devel
cd /root/sideload
rm -rf linux
tar xvf linux.tar.gz
cd linux
make allmodconfig
make -j32
