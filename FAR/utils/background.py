import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from sklearn import metrics
import sys
sys.path.insert(1, '../')
from FAR.utils.measure import Measure, CoincMass, SNRveto, Binning

class BackgroundFineTuner:
    PLOT_DPI = 300
    IFOS_CONFIG_6 = ['H1', 'L1', 'V1']

    def __init__(self, ifos, t_search, t_bkg):
        self.ifos = ifos
        self.t_search = t_search
        self.t_bkg = t_bkg
        self.dt = np.concatenate([np.arange(0.01, 0.9, 0.01),
                                  np.asarray([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999,0.9999999])])

    def apply_veto(self, signal, condition, s1):
        if condition == r'$SNR$':
            if len(self.ifos) == 4:
                return SNRveto(s1, signal, [self.ifos[:2], self.ifos[2:]])
            if len(self.ifos) == 6:
                return SNRveto(s1, signal, self.IFOS_CONFIG_6)
        elif condition == r'$\Delta m$':
            if len(self.ifos) == 4:
                return CoincMass(signal, [self.ifos[:2], self.ifos[2:]], s1, s1).filterCoincMass()
            if len(self.ifos) == 6:
                return CoincMass(signal, self.IFOS_CONFIG_6, s1, s1).filterCoincMass()
        return signal

    def calculate_tprs_fprs(self, binning, pos, neg, xlabel):
        print(neg.columns)

        x, y_obs, _, _ = Measure.getFARandThreshold(neg, self.ifos,
                                                    self.t_bkg, self.t_search, FAR=0.1)

        tprs, fprs = [], []
        for d in self.dt:
            p_star = -np.log(1 - d + 1e-20)
            # We only consider post-GstLAL
            tp, p, _, _ = Measure.TPR(pos, self.ifos, p_star=p_star)
            tpr = tp / p * 100

            fpr = Measure.FPR(neg, self.ifos, p_star=p_star)
            tprs.append(tpr)
            fprs.append(fpr)
        return np.asarray(tprs), np.asarray(fprs)

    def plot_tpr_fpr(self, fprs, tprs, aucs, vetoes, xlabel):
        colors = plt.cm.viridis(np.linspace(1, 0, len(vetoes)))[::-1]
        cmap = LinearSegmentedColormap.from_list('custom', colors[::-1], N=len(vetoes))
        
        plt.figure()
        plt.loglog(fprs[0], tprs[0], marker='o', c='darkorange', markersize=2, label=f'{vetoes[0]}', zorder=100)
        for fpr, tpr, auc, v, c in zip(fprs[1:], tprs[1:], aucs[1:], vetoes[1:], colors[1:]):
            plt.loglog(fpr, tpr, marker='o', c=c, markersize=2)
        
        sm = ScalarMappable(cmap=cmap)
        sm.set_array(vetoes[1:])
        cbar = plt.colorbar(sm, label=xlabel)
        plt.ylim(10, 100)
        labels = [100, 50, 20, 5, 1]
        plt.xticks(labels, labels), plt.yticks(labels, labels)
        plt.xlabel('FPR'), plt.ylabel('TPR'), plt.legend()
        name = 'snr' if xlabel == r'$SNR$' else 'mass'
        plt.savefig('../FAR/closedbox/roc_'+name+'_'+self.ifos+'.png', dpi=self.PLOT_DPI)
        plt.savefig('../FAR/closedbox/roc_'+name+'_'+self.ifos+'.pdf')

    def fine_tune_bkg(self, all_tmps, bkg, xlabel):

        columns_to_check = all_tmps.columns[all_tmps.columns.str.contains('Cluster time_')]
        all_tmps = all_tmps.dropna(subset=columns_to_check)
        
        binning = Binning() # FIXME 3.9
        vetoes = [None] + (np.arange(6, 3.9, -0.1).tolist() if xlabel == r'$SNR$' else np.arange(100, 5, -5).tolist())
        fprs, tprs, aucs = [], [], []
        
        for v in vetoes:
            if v is None:
                pos, neg = all_tmps.copy(), bkg.copy()
            else:
                pos = self.apply_veto(all_tmps.copy(), xlabel, v)
                neg = self.apply_veto(bkg.copy(), xlabel, v)
                
            tpr, fpr = self.calculate_tprs_fprs(binning, pos, neg, xlabel)

            tprs.append(tpr)
            fprs.append(fpr)
            
            # We need to correct the AUC to compute the full area
            correction = (100 - max(fpr)) * max(tpr)
            auc = np.round(metrics.auc(fpr, tpr) + correction, 3)
            aucs.append(auc)
            print(v, auc, len(pos), len(neg))
        self.plot_tpr_fpr(fprs, tprs, aucs, vetoes, xlabel)
        return tprs, fprs, aucs, vetoes
