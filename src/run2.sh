#!/usr/bin/bash
source ~/.bashrc
conda activate /data/gravwav/lopezm/cmb_env/miniforge3/envs/ml_course
python3 /data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/src/unknown_coinc.py --tw=0.05 --ifos='L1V1'
