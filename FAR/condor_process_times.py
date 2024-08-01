import os
import argparse
import configparser
import numpy as np

# Define functions

def shfile(condor_dir, namesh):

    lines = ['#!/bin/bash',
             'source ~/.bashrc && conda activate tracks_env',
             'cd /data/gravwav/lopezm/Projects/GlitchBank/git_new/computational-aspects-of-machine-learning-project-3/',
             'export PYTHONPATH=$(pwd)',
             'echo ${job_start}',
             'echo ${N}',
             'python3 /data/gravwav/lopezm/Projects/GlitchBank/git_new/computational-aspects-of-machine-learning-project-3/FAR/computeTimes.py --job_start ${job_start} --N ${N}']

    with open(condor_dir+"%s.sh" % (namesh), 'w') as f:
        f.write('\n'.join(lines))
    f.close()


def subfile(condor_dir, namesh, namesub, logdir, request_memory, request_disk):

    lines = ['+UseOS           = "el9"',
             '+JobCategory     = "short"',
             'request_memory   = %i M' % (request_memory),
             'request_disk     = %i M' % (request_disk),
             'executable       = %s.sh' % (namesh),
             'environment      = PID=$(PID);job_start=$(job_start);N=$(N)', 
             'output           = %s/$(PID).out' % (logdir),
             'error            = %s/$(PID).err' % (logdir),
             'log              = CreateJobs.log',
             'notification     = never',
             'rank             = memory',
             'queue 1']

    with open(condor_dir + "%s.sub" % (namesub), 'w') as f:
        f.write('\n'.join(lines))
    f.close()


def dagfile(condor_dir, namedag, namesub):

    count = 1
    lines = list()
    N = 12
    ids = np.arange(1, 6100, N)
    for i in ids:

        line1 = 'JOB A%i %s' % (count, "%s.sub" % (namesub))
        line2 = 'VARS A%i PID="%i" jobs_start="%i" N="%i"' % (count, count, int(i), N)
        line3 = 'RETRY A%i 3' % (count)

        lines.append(line1)
        lines.append(line2)
        lines.append(line3)

        count += 1
    with open(condor_dir + "%s.dag" % (namedag), 'w') as f:
        f.write('\n'.join(lines))
    f.close()

run_name = 'times_real'
request_memory = 4000 
request_disk = 4000
condor_dir = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/dag/'
logdir = condor_dir + 'logs'
if not os.path.exists(logdir):
    os.mkdir(logdir)
os.system(f'rm {condor_dir}logs/* {condor_dir}times_real_dag.dag.*')
dagfile(condor_dir, run_name + "_dag", run_name + "_sub")
subfile(condor_dir, run_name + "_sh", run_name + "_sub", logdir, request_memory, request_disk)
shfile(condor_dir, run_name + "_sh")

os.system("chmod +x "+ condor_dir + run_name + "_sh.sh")
print('cd '+condor_dir)
print('condor_submit_dag '+run_name+'_dag.dag')
 --job_start $(job_start) --N $(N)',  # Ensure these match the DAG variables
