#!/bin/env python3

import argparse
import os
import sys
import time
import math
import pathlib
import json
import re
import subprocess
import datetime
import signal
import multiprocessing

# Iterate every second
interval = 1

USER_HZ = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
CGRP_BASE = '/sys/fs/cgroup'
SL_BASE = '/var/lib/sideloader'
SL_PREFIX = 'sideload-'
SVC_SUFFIX = '.service'
dfl_cfg_file = SL_BASE + '/config.json'
dfl_job_dir = SL_BASE + '/jobs.d'
dfl_status_file = SL_BASE + '/status.json'

description = '''
Resource control side-workload manager. See the following for details.

   https://fb.quip.com/qYC2Ay7SyyO7

'''

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=description)
parser.add_argument('--config', default=dfl_cfg_file,
                    help='Config file (default: %(default)s)')
parser.add_argument('--jobdir', default=dfl_job_dir,
                    help='Job input directory (default: %(default)s)')
parser.add_argument('--status', default=dfl_status_file,
                    help='Status file (default: %(default)s)')
parser.add_argument('--verbose', '-v', action='count')

args = parser.parse_args()

#
# Utility functions
#
def ddbg(s):
    global args
    if args.verbose and args.verbose >= 2:
        print(f'DBG: {s}', flush=True)

def dbg(s):
    global args
    if args.verbose:
        print(f'DBG: {s}', flush=True)

def log(s):
    print(s, flush=True)

def warn(s):
    print(f'WARN: {s}', file=sys.stderr, flush=True)

def err(s):
    print(f'ERR: {s}', file=sys.stderr, flush=True)
    sys.exit(1)

def parse_size(s):
    units = { 'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30, 'T': 1 << 40 }
    split = re.sub(r'([kKmMgGtT])', r' \1 ', s).split()
    size = 0
    for i in range(0, len(split), 2):
        try:
            num = float(split[i])
        except:
            num = float('nan')
        if not math.isfinite(num) or (i + 1 < len(split) and split[i + 1] not in units):
            raise Exception(f'invalid size "{s}"')
        if i + 1 < len(split):
            size += num * units[split[i + 1]]
        else:
            size += num
    return int(size)

# "1.5G"   - 1.5 gigabytes, returns 1610612736 (bytes)
# "1G128M" - 1 gigabyte and 128 megabytes, returns 1207959552 (bytes)
# "35.7%"  - 35.7% of whole
def parse_size_or_pct(s, whole):
    s = s.strip()
    if s.endswith('%'):
        return int(whole * float(s[:-1]) / 100)
    else:
        return parse_size(s)

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
        if len(lines) == 1 and not len(lines[0]):
            return []
        return lines

def read_first_line(path):
    return read_lines(path)[0]

def read_cpu_idle():
        toks = read_first_line('/proc/stat').split()[1:]
        idle = int(toks[3]) + int(toks[4])
        total = 0
        for tok in toks:
            total += int(tok)
        return idle, total

def read_mem_swap():
    mem_total = None
    swap_total = None
    swap_free = None

    with open('/proc/meminfo', 'r', encoding='utf-8') as f:
        for line in f:
            toks = line.split()
            if toks[0] == 'MemTotal:':
                mem_total = int(toks[1]) * 1024
            elif toks[0] == 'SwapTotal:':
                swap_total = int(toks[1]) * 1024
            elif toks[0] == 'SwapFree:':
                swap_free = int(toks[1]) * 1024

    return mem_total, swap_total, swap_free

def read_cgroup_keyed(path):
    content = {}
    for line in read_lines(path):
        toks = line.split()
        key = toks[0]
        content[key] = toks[1]
    return content

def read_cgroup_nested_keyed(path):
    content = {}
    for line in read_lines(path):
        toks = line.split()
        key = toks[0]
        content[key] = {}
        for tok in toks[1:]:
            nkey, val = tok.split('=')
            content[key][nkey] = val
    return content

def float_or_max(v, max_val):
    if v == 'max':
        return max_val
    return float(v)

def dump_json(data, path):
    dirname, basename = os.path.split(path)
    tf = open(os.path.join(dirname, 'P{}-{}.tmp'.format(os.getpid(), basename)), 'w')
    tf.write(json.dumps(data, sort_keys=True, indent=4))
    tf.close()
    os.rename(tf.name, path)

def svc_to_jobid(svc):
    assert svc.startswith(SL_PREFIX) and svc.endswith(SVC_SUFFIX)
    return svc[len(SL_PREFIX):-len(SVC_SUFFIX)]

def time_interval_str(at, now):
    if at is None:
        return '0'
    else:
        intv = max(int(now - at), 1)
        return f'{intv:.2f}'

#
# Classes
#
class Config:
    def __init__(self, cfg):
        mem_total, swap_total, swap_free = read_mem_swap()

        self.main_slice = cfg['main-slice']
        self.side_slice = cfg['side-slice']
        self.cpu_period = float(cfg['cpu-period'])
        self.cpu_weight = int(cfg['cpu-weight'])
        self.cpu_headroom = float(cfg['cpu-headroom'])
        self.cpu_min_avail = float(cfg['cpu-min-avail'])
        self.cpu_throttle_period = float(cfg['cpu-throttle-period'])
        self.memory_high = parse_size_or_pct(cfg['memory-high'], mem_total)
        self.io_weight = int(cfg['io-weight'])

        self.ov_memp_thr = float(cfg['overload-mempressure-threshold'])
        self.ov_hold = float(cfg['overload-hold'])
        self.ov_hold_max = float(cfg['overload-hold-max'])
        self.ov_hold_decay = float(cfg['overload-hold-decay-rate'])

        self.crit_swapfree_thr = \
            parse_size_or_pct(cfg['critical-swapfree-threshold'], swap_total)
        self.crit_memp_thr = float(cfg['critical-mempressure-threshold'])
        self.crit_iop_thr = float(cfg['critical-iopressure-threshold'])

class JobFile:
    def __init__(self, ino, path, fh):
        self.ino = ino
        self.path = path
        self.fh = fh

    def __repr__(self):
        return f'{self.ino}:{self.path}'

class Job:
    def __init__(self, cfg, jobfile):
        jobid = cfg['id']
        if not re.search('^[A-Za-z0-9-_.]*$', jobid):
            raise Exception(f'"{jobid}" is not a valid identifier')

        frozen_exp = None
        if 'frozen-expiration' in cfg:
            frozen_exp = float(cfg['frozen-expiration'])

        self.jobfile = jobfile
        self.jobid = jobid
        self.cmd = cfg['cmd']
        self.frozen_exp = frozen_exp
        self.frozen_at = None
        self.done = False
        self.kill_why = None
        self.killed = False
        self.svc_name = f'{SL_PREFIX}{jobid}{SVC_SUFFIX}'
        self.svc_status = None

    def update_frozen(self, freeze, now):
        changed = False
        if not self.frozen_at and freeze:
            self.frozen_at = now
            changed = True
        elif self.frozen_at and not freeze:
            self.frozen_at = None
            changed = True

        path = pathlib.Path(f'{CGRP_BASE}/{config.side_slice}/{self.svc_name}/cgroup.freeze')
        if not path.exists():
            if changed:
                warn(f'Failed to freeze {self.jobid}')
            return

        if int(freeze) == int(read_first_line(path)):
            return

        with path.open('w') as f:
            f.write(str(int(freeze)))

    def maybe_kill(self):
        if not self.kill_why:
            return

        path = pathlib.Path(f'{CGRP_BASE}/{config.side_slice}/{self.svc_name}/cgroup.procs')
        if not path.exists():
            return

        pids = read_lines(path)
        if len(pids):
            dbg(f'killing {self.jobid}: {pids}')
            for pid in pids:
                os.kill(int(pid), signal.SIGKILL)
            log(f'JOB: Attempted to kill {self.jobid} ({len(pids)} processes)')

    def kill(self, why):
        dbg(f'kill requested for {self.jobid} why="{why}" cur_why="{self.kill_why}"')
        if not self.kill_why:
            self.kill_why = why
        self.maybe_kill()

    def refresh_status(self, now):
        self.svc_status = "<UNKNOWN>"

        out = subprocess.run(['systemctl', 'status', self.svc_name],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout.decode('utf-8')
        for line in out.split('\n'):
            toks = line.split(maxsplit=1)
            if len(toks) == 2 and toks[0] == 'Active:':
                self.svc_status = toks[1]
                break

        if '(exited)' in self.svc_status:
            self.done = True

        if 'failed' in self.svc_status:
            self.done = True
            self.killed = True


class Sysinfo:
    def __init__(self, pressure_dir, cpu_idle_slots):
        self.pressure_dir = pressure_dir
        self.cpu_idle_hist = [None] * cpu_idle_slots
        self.cpu_side_hist = [None] * cpu_idle_slots
        self.cpu_total_hist = [None] * cpu_idle_slots
        self.cpu_idle_pct_hist = [0] * cpu_idle_slots
        self.cpu_hist_idx = 0
        self.cpu_min_idle = 0
        self.cpu_avg_idle = 0
        self.cpu_avg_side = 0
        self.cpu_avail = 0
        self.memp_1min = 0
        self.memp_5min = 0
        self.iop_1min = 0
        self.iop_5min = 0
        self.swap_free_pct = 100
        self.critical = False
        self.critical_why = None
        self.overload = False
        self.overload_why = None

    def update(self):
        global config

        # cpu stats
        self.main_cpu_busy = 0
        cpu_idle, cpu_total = read_cpu_idle()
        cpu_idle = cpu_idle / USER_HZ * 1_000_000
        cpu_total = cpu_total / USER_HZ * 1_000_000
        cpu_stat = read_cgroup_keyed(f'{CGRP_BASE}/{config.side_slice}/cpu.stat')
        cpu_side = float(cpu_stat['usage_usec'])

        prev_idx = (self.cpu_hist_idx - 1) % len(self.cpu_idle_hist)
        next_idx = (self.cpu_hist_idx + 1) % len(self.cpu_idle_hist)
        prev_idle = self.cpu_idle_hist[prev_idx];
        prev_total = self.cpu_total_hist[prev_idx];
        oldest_idle = self.cpu_idle_hist[next_idx];
        oldest_side = self.cpu_side_hist[next_idx];
        oldest_total = self.cpu_total_hist[next_idx];

        if prev_total is not None and cpu_total > prev_total:
            self.cpu_idle_pct_hist[self.cpu_hist_idx] = \
                max(min((cpu_idle - prev_idle) /
                        (cpu_total - prev_total), 1), 0) * 100

        if oldest_total is not None and cpu_total > oldest_total:
            self.cpu_avg_idle = max(min((cpu_idle - oldest_idle) /
                                        (cpu_total - oldest_total), 1), 0) * 100
            self.cpu_avg_side = max(min((cpu_side - oldest_side) /
                                        (cpu_total - oldest_total), 1), 0) * 100

        self.cpu_min_idle = min(self.cpu_idle_pct_hist)
        self.cpu_idle_hist[next_idx] = cpu_idle
        self.cpu_side_hist[next_idx] = cpu_side
        self.cpu_total_hist[next_idx] = cpu_total
        self.cpu_hist_idx = next_idx

        self.cpu_avail = max(self.cpu_avg_side + self.cpu_min_idle - config.cpu_headroom,
                             config.cpu_min_avail)

        # memory and io pressures
        pres = read_cgroup_nested_keyed(self.pressure_dir + '/memory.pressure')
        self.memp_1min = float(pres['full']['avg60'])
        self.memp_5min = float(pres['full']['avg300'])

        pres = read_cgroup_nested_keyed(self.pressure_dir + '/io.pressure')
        self.iop_1min = float(pres['full']['avg60'])
        self.iop_5min = float(pres['full']['avg300'])

        # swap
        mem_total, self.swap_total, self.swap_free = read_mem_swap()
        self.swap_free_pct = 100
        if self.swap_total:
            self.swap_free_pct = self.swap_free / self.swap_total * 100

        # is critical?
        self.critical = True
        if self.swap_free <= config.crit_swapfree_thr:
            self.critical_why = (f'swap-free {self.swap_free>>20}MB is lower than '
                                 f'critical threshold {config.crit_swapfree_thr>>20}MB')
        elif self.memp_5min >= config.crit_memp_thr:
            self.critical_why = (f'5min memory pressure {self.memp_5min:.2f} is higher than '
                                 f'critical threshold {config.crit_memp_thr:.2f}')
        elif self.iop_5min >= config.crit_iop_thr:
            self.critical_why = (f'5min io pressure {self.iop_5min:.2f} is higher than '
                                 f'critical threshold {config.crit_iop_thr:.2f}')
        else:
            self.critical = False
            self.critical_why = False

        # is overloaded?
        side_margin = max(self.cpu_avg_side + self.cpu_avg_idle - config.cpu_headroom, 0)
        if side_margin < config.cpu_min_avail:
            self.overload = True
            self.overload_why = (f'cpu margin {side_margin:.2f} is too low')
        elif self.memp_1min >= config.ov_memp_thr:
            self.overload = True
            self.overload_why = (f'1min memory pressure {self.memp_1min:.2f} is over '
                                 f'the threshold {config.ov_memp_thr:.2f}')
        else:
            self.overload = False
            self.overload_why = None

#
# Implementation
#
def list_side_services():
    global config

    out = subprocess.run(['systemctl', 'list-units', '-l', SL_PREFIX + '*'],
                         stdout=subprocess.PIPE).stdout.decode('utf-8')
    svcs = []
    for line in out.split('\n'):
        toks = line[2:].split();
        if len(toks) and \
           toks[0].startswith(SL_PREFIX) and toks[0].endswith(SVC_SUFFIX):
            svcs.append(toks[0])
    return svcs

def process_job_dir(jobfiles, jobs, now):
    global args

    job_dir_path = pathlib.Path(args.jobdir)
    input_jobfiles = {}

    # Open all job files.
    for path in job_dir_path.glob('*'):
        try:
            if path.is_symlink() or not path.is_file():
                raise Exception('Invalid file type')
            fh = path.open('r', encoding='utf-8')
            ino = os.fstat(fh.fileno()).st_ino
            input_jobfiles[ino] = JobFile(ino, str(path), fh)
        except Exception as e:
            warn(f'Failed to open {path} ({e})')

    # Let's find out which files are gone and which are new.
    gone_jobfiles = []
    new_jobfiles = []

    for ino, jf in jobfiles.items():
        if jf.ino not in input_jobfiles:
            gone_jobfiles.append(jf)
    for ino, jf in input_jobfiles.items():
        if jf.ino not in jobfiles:
            new_jobfiles.append(jf)

    if len(gone_jobfiles):
        ddbg(f'gone_jobfiles: {[jf.path for jf in gone_jobfiles]}')
    if len(new_jobfiles):
        ddbg(f'new_jobfiles: {[jf.path for jf in new_jobfiles]}')

    for jf in gone_jobfiles:
        del jobfiles[jf.ino]

    # Collect active jobids and determine jobs to kill.
    jobids = set()
    jobs_to_kill = {}

    for i, job in jobs.items():
        if job.jobfile.ino in jobfiles:
            jobids.add(job.jobid)
        else:
            jobs_to_kill[job.jobid] = job

    if len(jobs_to_kill):
        ddbg(f'jobs_to_kill: {[jobid for jobid in jobs_to_kill]}')

    # Load new job files
    jobs_to_start = {}

    for jf in new_jobfiles:
        jf_jobids = set()
        jf_jobs = {}
        try:
            parsed = json.load(jf.fh)
            for ent in parsed['sideloader-jobs']:
                job = Job(ent, jf)
                if job.jobid in jobids or job.jobid in jf_jobids:
                    raise Exception(f'Duplicate job id {job.jobid}')
                jf_jobids.add(job.jobid)
                jf_jobs[job.jobid] = job
        except Exception as e:
            warn(f'Failed to load {path} ({e})')
        else:
            jobfiles[jf.ino] = jf
            jobids = jobids.union(jf_jobids)
            jobs_to_start.update(jf_jobs)

    if len(jobs_to_start):
        ddbg(f'jobs_to_start: {[jobid for jobid in jobs_to_start]}')

    return jobs_to_kill, jobs_to_start

def verify_sysconfig():
    global config

    warns = []

    root_part = None
    root_dev = None
    root_maj = None
    root_min = None

    # is root btrfs?
    for line in read_lines('/proc/mounts'):
        toks = line.split()
        if toks[1] == '/':
            if toks[2] != 'btrfs':
                warns.append('root filesystem is not btrfs')
            if toks[0].startswith('/dev/'):
                root_part = toks[0][len('/dev/'):]
            break

    # is iocost enabled?
    if root_part is None:
        warns.append('failed to find root device')
    else:
        if root_part.startswith('sd'):
            root_dev = re.sub(r'^(sd[^0-9]*)[0-9]*$', r'\1', root_part)
        elif root_part.startswith('nvme'):
            root_dev = re.sub(r'^(nvme[^p]*)(p[0-9])?$', r'\1', root_part)
        else:
            raise Exception(f'unknown device')

        try:
            out = subprocess.run(['stat', '-c', '0x%t 0x%T', f'/dev/{root_dev}'],
                                 stdout=subprocess.PIPE).stdout.decode('utf-8')
            toks = out.split()
            root_maj = int(toks[0], 0)
            root_min = int(toks[1], 0)
        except Exception as e:
            warns.append(f'failed to find devnr for {root_part} ({e})')
        else:
            try:
                enabled = False
                for line in read_lines('/sys/fs/cgroup/io.cost.qos'):
                    toks = line.split()
                    devnr = toks[0].split(':')
                    if root_maj == int(devnr[0]) and root_min == int(devnr[1]):
                        for t in toks:
                            if t == 'enable=1':
                                enabled = True
                                break
                        break

                if not enabled:
                    warns.append(f'iocost not enabled on {root_dev}')
            except Exception as e:
                warns.append(f'failed to verify iocost for {root_dev} ({e})')

    # is freezer available
    if not os.path.exists(f'{CGRP_BASE}/{config.side_slice}/cgroup.freeze'):
        warns.append('freezer is not available')

    # is cpu controller enabled?
    if 'cpu' not in read_first_line(f'{CGRP_BASE}/cgroup.subtree_control'):
        warns.append('cpu controller not enabled at root, fixing...');
        try:
            subprocess.run(['systemctl', 'set-property', '--', '-.slice', 'DisableControllers='])
            with open(f'{CGRP_BASE}/cgroup.subtree_control', 'w') as f:
                f.write('+cpu')
        except Exception as e:
            warns.append(f'failed to enable CPU controller in the root cgroup ({e})')

    # verify swap and swappiness
    mem_total, swap_total, swap_free = read_mem_swap()
    if swap_total < 0.9 * (mem_total / 2):
        warns.append('swap is smaller than half of physical memory')

    swappiness = int(read_first_line('/proc/sys/vm/swappiness'))
    if swappiness < 60:
        warns.append('swappiness ({swappiness}) is lower than default 60')

    # verify resource configs in main workload
    try:
        weight = int(read_first_line(f'{CGRP_BASE}/{config.main_slice}/cpu.weight'))
        if weight < config.cpu_weight * 4:
            warns.append(f'{config.main_slice} cpu.weight is lower than 4x {config.side_slice}')
    except Exception as e:
        warns.append(f'failed to check {config.main_slice} cpu.weight ({e})')
    try:
        weight = int(read_first_line(f'{CGRP_BASE}/{config.main_slice}/io.weight').split()[1])
        if weight < config.io_weight * 4:
            warns.append('{config.main_slice} io.weight is lower than 4x {config.side_slice}')
    except Exception as e:
        warns.append(f'failed to check {config.main_slice} io.weight ({e})')

    main_memory_low = None
    try:
        main_path = pathlib.Path(f'{CGRP_BASE}/{config.main_slice}')
        for subdir in ('', 'workload-tw.slice/', 'workload-tw.slice/*.task/',
                       'workload-tw.slice/*.task/task/'):
            for path in main_path.glob(f'{subdir}memory.low'):
                low = int(float_or_max(read_first_line(path), mem_total))
                if low < mem_total / 3:
                    if main_memory_low:
                        warns.append(f'{str(path)} is lower than a third of system '
                                     f'memory, configuring to {main_memory_low}')
                        try:
                            with path.open('w') as f:
                                f.write(f'{main_memory_low}')
                        except Exception as e:
                            warns.append(f'failed to set {str(path)} to {main_memory_low} ({e})')
                    else:
                        warns.append(f'{str(path)} is lower than a third of system '
                                     f'memory, no idea what to config')
                else:
                    main_memory_low = low
    except Exception as e:
        warns.append(f'failed to check {config.main_slice}/* memory.low ({e})')

    # verify resource configs in side workload
    side_cgrp = f'{CGRP_BASE}/{config.side_slice}'
    fix_side = False
    try:
        weight = int(read_first_line(f'{side_cgrp}/cpu.weight'))
        if weight != config.cpu_weight:
            warns.append(f'{config.side_slice} cpu.weight is not {config.cpu_weight}, fixing...')
            fix_side = True
    except Exception as e:
        warns.append(f'failed to check {config.main_slice} cpu.weight ({e})')
    try:
        high = int(read_first_line(f'{side_cgrp}/memory.high'))
        if high != config.memory_high:
            warns.append(f'{config.side_slice} memory.high is not {config.memory.high}, fixing...')
            fix_side = True
    except Exception as e:
        warns.append(f'failed to check {config.side_slice} memory.high ({e})')
    try:
        weight = int(read_first_line(f'{side_cgrp}/io.weight').split()[1])
        if weight != config.io_weight:
            warns.append(f'{config.main_slice} io.weight is not {config.io_weight}, fixing...')
            fix_side = True
    except Exception as e:
        warns.append(f'failed to check {config.side_slice} io.weight ({e})')

    if fix_side:
        try:
            subprocess.call(['systemctl', 'set-property', config.side_slice,
                             f'CPUWeight={config.cpu_weight}',
                             f'MemoryHigh={config.memory_high}',
                             f'IOWeight={config.io_weight}'])
            with open(f'{side_cgrp}/cpu.weight', 'w') as f:
                f.write(f'{config.cpu_weight}')
            with open(f'{side_cgrp}/memory.high', 'w') as f:
                f.write(f'{config.memory_high}')
            with open(f'{side_cgrp}/io.weight', 'w') as f:
                f.write(f'{config.io_weight}')
        except Exception as e:
            warns.append(f'failed to set {config.side_slice} resource configs ({e})')

    return warns

def config_cpu_max(pct):
    global config

    cpu_max_file = f'{CGRP_BASE}/{config.side_slice}/cpu.max'
    period = int(config.cpu_throttle_period * 1_000_000)
    quota = int(multiprocessing.cpu_count() * period * pct / 100)

    try:
        cur_quota, cur_period = read_first_line(cpu_max_file).split()
        cur_quota = 100 if cur_quota == 'max' else int(cur_quota)
        cur_period = int(cur_period)
        if period == cur_period and quota == cur_quota:
            return

        with open(cpu_max_file, 'w') as f:
            f.write(f'{quota} {period}')
    except Exception as e:
        warn(f'Failed to configure {cpu_max_file} ({e})')

# Run
config = Config(json.load(open(args.config, 'r'))['sideloader-config'])
dbg(f'Config: {config.__dict__}')
log(f'INIT: sideloads in {config.side_slice}, main workloads in {config.main_slice}')

job_queue = {}
critical_at = None
overload_at = None
overload_hold_from = 0
overload_hold = 0
jobfiles = {}
jobs = {}
now = time.time()
syscfg_warns = {}
last_syscfg_at = 0

sysinfo = Sysinfo(f'{CGRP_BASE}/{config.side_slice}',
                  math.ceil(config.cpu_period / interval))

# Init sideload.slice
subprocess.call(['systemctl', 'set-property', config.side_slice,
                 f'CPUWeight={config.cpu_weight}',
                 f'MemoryHigh={config.memory_high}',
                 f'IOWeight={config.io_weight}'])

# List sideload.slice and kill everything which isn't in the jobdir.
# Don't worry about matching or missing ones, the main loop will
# handle them.
svcs = list_side_services()
jobs_to_kill, jobs_to_start = process_job_dir({}, {}, now)
svcs_to_stop = []

for svc in svcs:
    if svc_to_jobid(svc) not in jobs_to_start:
        svcs_to_stop.append(svc)

if len(svcs_to_stop):
    log(f'JOB: Stopping stray services {svcs_to_stop}')
    subprocess.run(['systemctl', 'stop'] + svcs_to_stop)
    subprocess.run(['systemctl', 'reset-failed'] + svcs_to_stop)

# The main loop
while True:
    last = now
    now = time.time()

    # Handle job starts and stops
    jobs_to_kill, jobs_to_start = process_job_dir(jobfiles, jobs, now)
    svcs_to_stop = []
    for jobid, job in jobs_to_kill.items():
        log(f'JOB: Stopping {job.svc_name}')
        subprocess.run(['systemctl', 'stop', job.svc_name])
        subprocess.run(['systemctl', 'reset-failed', job.svc_name])
        del jobs[jobid]
    # Start new jobs iff not overloaded
    job_queue.update(jobs_to_start)
    if not overload_at:
        for jobid, job in job_queue.items():
            log(f'JOB: Starting {job.svc_name}')
            jobs[jobid] = job
            subprocess.call(f'systemd-run -r --slice {config.side_slice} '
                            f'--unit {job.svc_name} {job.cmd}', shell=True)
        job_queue = {}

    # Do syscfg check every 10 secs if there are jobs; otherwise, every 60s
    if now - last_syscfg_at >= (10 if len(jobs) else 60):
        last_syscfg_at = now
        warns = verify_sysconfig()
        if syscfg_warns != warns:
            if len(warns):
                i = 0
                for w in warns:
                    warn(f'SYSCFG[{i}]: {w}')
                    i += 1
            else:
                log(f'SYSCFG: All good')
        syscfg_warns = warns

    # Read the current system state
    sysinfo.update()

    # Configure side's cpu.max
    if len(jobs):
        config_cpu_max(sysinfo.cpu_avail)

    # Handle critical condition
    if sysinfo.critical:
        if critical_at is None:
            crtical_at = now
        if overload_at is None:
            overload_at = now
        overload_reason = 'resource critical'
        overload_hold = config.ov_hold_max
        for jobid, job in jobs.items():
            job.kill('resource critical')
    else:
        critical_at = None

    # Handle overload condition
    if sysinfo.overload:
        # Log if we're just getting overloaded
        if not overload_at:
            log(f'OVERLOAD: {sysinfo.overload_why}, hold={int(overload_hold)}s')
            overload_at = now
            overload_hold = min(config.ov_hold + overload_hold, config.ov_hold_max)
        overload_hold_from = now
    elif overload_at:
        if now > overload_hold_from + overload_hold:
            log('OVERLOAD: end, resuming normal operation')
            overload_at = None

    if overload_at:
        for jobid, job in jobs.items():
            job.update_frozen(True, now)
            ddbg(f'{jobid} frozen for {int(now - job.frozen_at)}s exp={job.frozen_exp}')
            if now - job.frozen_at >= job.frozen_exp:
                job.kill('frozen for too long')
    else:
        overload_hold = max(overload_hold - config.ov_hold_decay, 0)
        for jobid, job in jobs.items():
            job.update_frozen(False, now)

    # Update job status and report
    for jobid, job in jobs.items():
        job.maybe_kill()
        job.refresh_status(now)

    status = {
        'sideloader-status': {
            'now': str(datetime.datetime.fromtimestamp(now)),
            'sysconfig-warnings-at': str(datetime.datetime.fromtimestamp(last_syscfg_at)),
            'sysconfig-warnings': syscfg_warns,
            'jobs': [ { 'id': jobid,
                        'path' : job.jobfile.path,
                        'service-name': job.svc_name,
                        'service-status': job.svc_status,
                        'frozen-for': time_interval_str(job.frozen_at, now),
                        'is-killed': f'{int(job.killed)}',
                        'is-done': f'{int(job.done)}',
                        'kill-why': f'{job.kill_why if job.kill_why else ""}',
                      } for jobid, job in jobs.items() ],
            'sysinfo': {
                'cpu-min-idle': f'{sysinfo.cpu_min_idle:.2f}',
                'cpu-avg-idle': f'{sysinfo.cpu_avg_idle:.2f}',
                'cpu-avg-side': f'{sysinfo.cpu_avg_side:.2f}',
                'cpu-avail': f'{sysinfo.cpu_avail:.2f}',
                'mempressure-1min': f'{sysinfo.memp_1min:.2f}',
                'mempressure-5min': f'{sysinfo.memp_5min:.2f}',
                'iopressure-1min': f'{sysinfo.iop_1min:.2f}',
                'iopressure-5min': f'{sysinfo.iop_5min:.2f}',
                'swap-free-pct': f'{sysinfo.swap_free_pct:.2f}',
            },
            'overload': {
                'critical-for': time_interval_str(critical_at, now),
                'overload-for': time_interval_str(overload_at, now),
                'overload-hold': f'{max(overload_hold_from + overload_hold - now, 0):.2f}',
                'critical-why': f'{sysinfo.critical_why if sysinfo.critical_why else ""}',
                'overload-why': f'{sysinfo.overload_why if sysinfo.overload_why else ""}',
            }
        }
    }
    dump_json(status, args.status)

    time.sleep(interval)
