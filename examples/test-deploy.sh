#!/bin/bash

DIR=$(dirname $0)
TARGET=root@$1

cd $DIR
scp ../sideloader.py config.json test-job.json test-job.sh linux.tar.gz $TARGET:
ssh $TARGET 'mkdir -p /var/lib/sideloader/jobs.d && mkdir -p sideload && mv config.json /var/lib/sideloader/ && mv test-job.json test-job.sh linux.tar.gz sideload/'
