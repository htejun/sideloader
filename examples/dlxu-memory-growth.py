#!/bin/python

import datetime
import gc
import os
import resource
import time

PAGE_SIZE = resource.getpagesize()
ALLOCATION = int(os.environ.get('ALLOCATION', 5 * (10 ** 9)))
FRAC_PER_LOOP = float(os.environ.get('FRAC_PER_LOOP', 1 / float(10)))

def get_memory_usage():
    return int(open("/proc/self/statm", "rt").read().split()[1]) * PAGE_SIZE

def bloat(size):
    l = []
    mem_usage = get_memory_usage()
    target_mem_usage = mem_usage + size
    while get_memory_usage() < target_mem_usage:
        l.append(b"g" * (10 ** 6))
    return l

def run():
    arr = []  # prevent GC
    prev_time = datetime.datetime.now()
    while True:
        # allocate some memory
        if get_memory_usage() < ALLOCATION:
            l = bloat(ALLOCATION * FRAC_PER_LOOP)
            arr.append(l)
            now = datetime.datetime.now()
            print("{} -- RSS = {} bytes. Delta = {}".format(now, get_memory_usage(), (now - prev_time).total_seconds()))
            prev_time = now
        else:
            break

        time.sleep(1)

    print('{} -- Done with workload'.format(datetime.datetime.now()))

if __name__ == "__main__":
    run()
