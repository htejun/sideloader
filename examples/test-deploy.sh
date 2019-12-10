#!/bin/bash

DIR=$(dirname $0)
TARGET=root@$1

cd $DIR
scp ../sideloader.py ../config.json ../sideloader.service \
    test-job.json test-job.sh linux.tar.gz $TARGET:
ssh $TARGET 'mkdir -p /var/lib/sideloader/jobs.d && mkdir -p sideload && mv sideloader.py config.json /var/lib/sideloader/ && mv test-job.json test-job.sh linux.tar.gz sideload/ && mv sideloader.service /etc/systemd/system && systemctl daemon-reload && systemctl enable sideloader && systemctl start sideloader'
