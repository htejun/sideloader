#!/bin/bash

set -xe
TARGET=root@$1

cd $(dirname $0)
ssh $TARGET 'mkdir -p /var/lib/sideloader/jobs.d'
scp sideloader.py config.json $TARGET:/var/lib/sideloader
scp sideloader.service $TARGET:/etc/systemd/system
ssh $TARGET 'systemctl daemon-reload && systemctl enable sideloader && systemctl restart sideloader'
