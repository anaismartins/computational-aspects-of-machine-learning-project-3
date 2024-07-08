import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../../computational-aspects-of-machine-learning-project-3')
from src.torch_utils import load_model, predictions
#from FAR.utils import measure, read
from FAR.utils.read import RecoveryData, LogitToStat, DataProcessor, Merge
from FAR.utils.measure import Measure, Binning, CoincMass, SNRveto
from FAR.utils.plot import Plotting as pl
from FAR.utils.background import BackgroundFineTuner
from FAR.utils.utilities import rename_columns
import warnings
from itertools import product
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Process interferometer configurations.')
parser.add_argument('--ifos', nargs='+', default=None,
                    help='List of interferometer configurations')
args = parser.parse_args()
ifos = args.ifos[0]
# Some variables
path = '/data/gravwav/lopezm/Projects/GlitchBank/'
path_store = path + 'runs/background/results/'
timeslides_file = 'computational-aspects-of-machine-learning-project-3/FAR/timeslides_1132.9y.csv'
path_inj = '/data/gravwav/lopezm/Projects/GlitchBank/runs/injections/'
if len(ifos) == 4:
    inj = pd.read_pickle(path_inj + 'inj_coinc2_pred_'+ifos.lower().replace("1", "")+'.csv')
if len(ifos) == 6:
    inj = pd.read_pickle(path_inj + 'inj_coinc3_pred_hlv.csv')
print(len(inj))


merge, binning = Merge(path_store, path + timeslides_file), Binning() 
#t_search = pd.read_csv('timeslides_1132.9y.csv')[['time']]
if ifos == 'H1L1': t_search = 0.561999936580416
if ifos == 'H1V1': t_search = 0.67373829908675
if ifos == 'L1V1': t_search = 0.6958418315575849
if ifos == 'H1L1V1': t_search = 0.5619617579908676

# Call the background
t_bkg, bkg = merge.mergeTriggers(ifos) # FIXME

aucs_snr, aucs_m = [], []

for xlabel, auc_list, name in zip([r'$SNR$', r'$\Delta m$'],
                                      [aucs_snr, aucs_m], ['snr', 'mass']):
#for xlabel, auc_list, name in product([r'$SNR$'],
#                                      [aucs_snr], ['snr']): 
    tprs, fprs, aucs, vetoes = BackgroundFineTuner(ifos,
                                                   t_search, t_bkg).fine_tune_bkg(inj,
                                                                                  bkg, xlabel)
    auc_list.append(aucs)
np.save('./closedbox/aucs/auc_'+ifos+'_'+name+'.csv', auc_list)
