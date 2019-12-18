#!/bin/bash

set -xe
cd "$(dirname $0)"

yum -y install gcc bison flex openssl-devel elfutils-libelf-devel
rm -rf linux
tar xvf linux.tar.gz
cd linux
make allmodconfig
make -j64
