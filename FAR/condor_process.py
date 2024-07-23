import os
import argparse
import configparser
import numpy as np
# Define functions

def shfile(condor_dir, namesh):

    lines = ['#!/bin/bash',
             'source ~/.bashrc',
             'conda init bash',
             'conda activate tracks_env',
             'python3 /data/gravwav/lopezm/Projects/GlitchBank/git_new/computational-aspects-of-machine-learning-project-3/FAR/far_eqmatch.py --runs=${runs} --n=${n}']

    with open("%s.sh" % (namesh), 'w') as f:
        f.write('\n'.join(lines))
    f.close()


def subfile(condor_dir, namesh, namesub, logdir, request_memory, request_disk):

    lines = ['+UseOS           = "el9"',
             '+JobCategory     = "short"',
             'request_memory   = %i M' % (request_memory),
             'request_disk     = %i M' % (request_disk),
             'executable       = %s.sh' % (namesh),
             'environment      = runs=$(runs);n=$(n)',
             'output           = %s/$(PID).out' % (logdir),
             'error            = %s/$(PID).err' % (logdir),
             'log              = CreateJobs.log',
             'notification     = never',
             'rank             = memory',
             'queue 1']

    with open("%s.sub" % (namesub), 'w') as f:
        f.write('\n'.join(lines))
    f.close()


def dagfile(condor_dir, namedag, namesub, runs, n, ids):

    count = 1
    lines = list()

    for i, run in zip(ids, runs):

        line1 = 'JOB A%i %s' % (count, "%s.sub" % (namesub))
        line2 = 'VARS A%i PID="%i" runs="%i" n="%i"' % (count, count, run, n)
        line3 = 'RETRY A%i 3' % (count)

        lines.append(line1)
        lines.append(line2)
        lines.append(line3)

        count += 1
    with open("%s.dag" % (namedag), 'w') as f:
        f.write('\n'.join(lines))
    f.close()

if not os.path.exists('./logs'):
    os.mkdir('./logs')


condor_dir = '.'
logdir = './logs'
request_memory   = 3000
request_disk     = 3000
run_name = 'test'
n = 2
runs = np.arange(0, 2100, n)
ids = np.arange(len(runs))

dagfile(condor_dir, run_name + "_dag", run_name + "_sub", runs, n, ids)
subfile(condor_dir, run_name + "_sh", run_name + "_sub", logdir, request_memory, request_disk)
shfile(condor_dir, run_name + "_sh")
os.system("chmod +x "+ run_name + "_sh.sh")
print('done')
