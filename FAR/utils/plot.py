import numpy as np
import sys
sys.path.insert(1, '../')
from FAR.utils.measure import Measure, CoincMass, SNRveto, Binning
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn import metrics
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter

class Plotting:
    def __init__(self):
        pass

    @staticmethod
    def plotSigma(x_obs, x_teo, FAR_obs, FAR_teo, color, dark_color, ax, label, ylabel, ylim, desired_far=0.01):
        ax.plot(x_obs, FAR_obs, c=dark_color, label='Obs. ' + label)

        ax.plot(x_teo, FAR_teo, c=color, label='Mod. BKG')
        for sigma, alpha in zip(np.arange(1, 6), 1 - np.linspace(0.75, 0.9, 5)):

            if sigma == 1:
                label_sigma = r'$\pm \sigma$'
            else:
                label_sigma = r'$\pm '+str(sigma)+' \sigma$'
            ax.fill_between(x_teo, 
                             FAR_teo - sigma*np.sqrt(FAR_teo),
                             FAR_teo + sigma*np.sqrt(FAR_teo), 
                             color=color, alpha=alpha)

        ax.set_xlabel(r'$-\log{(1 - \bar{P}_{inj})}$')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')

        # Set y-ticks within ylim range
        yticks = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        yticks = [tick for tick in yticks if ylim[0] <= tick <= ylim[1]]
        ax.set_yticks(yticks)
        
        # Set y-axis limit
        ax.set_ylim(ylim)
        ax.grid(True)
        desired_far = 0.01
        desired_stat = Measure.FAR2stat(desired_far, x_obs, FAR_obs)
        ax.scatter(desired_stat, desired_far, 
                    label='FAR at 0.01 at ' + str(np.round(desired_stat, 2)), c='grey', marker='x', zorder=10)
        ax.legend(fontsize=10)
        return desired_stat

    @staticmethod
    def FARvsEvents(N, t_bkg, t_search, color, ax, i, j, label):
        iFAR = 1/(N/t_bkg)
        events = N*t_search/t_bkg
        ax.plot(iFAR,
                 N*t_search/t_bkg, c=color, label=label)

        for sigma, alpha in zip(np.arange(1, 6), 1 - np.linspace(0.75, 0.9, 5)):

            if sigma == 1:
                label_sigma = r'$\sigma$'
            else:
                label_sigma = str(sigma) + r'$\sigma$'
            ax.fill_between(iFAR, 
                             events - sigma*np.sqrt(events),
                             events + sigma*np.sqrt(events), 
                             color=color, alpha=alpha)
        if i != 0:
            ax.set_xlabel(r'Inversed False Alarm Rate (y)')
        if j == 0:
            ax.set_ylabel(r'Number of events')    
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(1e-4, 10)
        ax.legend()

    @staticmethod
    def PlotGrapes(cat, ifo):
        cat_det = cat[cat['FAR_'+ifo] <= 0.1]
        catalogs = ['GWOSC', 'Nitz', 'Princeton']
        cmap1 = LinearSegmentedColormap.from_list(name='test', colors=['darkslateblue', 'thistle'])
        cmap2 = LinearSegmentedColormap.from_list(name='test', colors=['sienna', 'bisque'])
        cmap3 = LinearSegmentedColormap.from_list(name='test', colors=['navy', 'cornflowerblue'])
        
        fig = plt.figure(figsize=(10, 6))
        for catalog, cmap, color in zip(catalogs, [cmap1, cmap2, cmap3], ['thistle', 'bisque', 'cornflowerblue']):
            
            cat_tmp = cat[cat['Catalog'] == catalog]
            cat_tmp_det = cat_tmp[cat_tmp['FAR_'+ifo] <= 0.1]
            plt.scatter(cat_tmp['SNR'], cat_tmp['Chirp mass'], c=color, s=1)
            #In Hull IMBH
            cat_tmp_inhull = cat_tmp[cat_tmp['inHUll'] == 'p']
            plt.scatter(cat_tmp_inhull['SNR'], cat_tmp_inhull['Chirp mass'], s=35, color=color,
                        alpha=0.8, lw=0.5, marker=r'$\odot$', facecolors='none', edgecolors=color)
        
            x, y, s = list(), list(), list()
            for i in range(len(cat_tmp_det)):
                if cat_tmp_det['inHUll'].iloc[i] == 'p':
                    x.append(cat_tmp_det['SNR'].iloc[i])
                    y.append(cat_tmp_det['Chirp mass'].iloc[i])
            plt.scatter(x, y, s=5, marker='o', color='white', zorder=10)
            
            plt.scatter(cat_tmp_det['SNR'], cat_tmp_det['Chirp mass'],
                        c=cat_tmp_det['FAR_'+ifo], cmap=cmap, s=40, zorder=5)
            
            cbar = plt.colorbar()
            cbar.set_label(catalog + ' ' r'FAR (yr$^{-1}$)', fontsize=11)  # Adjust font size of label
            cbar.ax.tick_params(labelsize=10)  # Adjust font size of ticks
        
            plt.clim(0.01, 0.1)
            plt.xlabel(r'Network SNR ($\rho_{net}$)'), plt.ylabel(r'Chirp mass $(\mathcal{M}$)')
        
            # Detection
            g1 = plt.scatter([],[], s=1, marker='o', color='silver')
            g4 = plt.scatter([],[], s=100, marker='o', color='silver')
            g2 = plt.scatter([],[], s=100, color='silver', marker=r'$\odot$', facecolors='none',lw=0.5,)
            plt.legend((g1, g4, g2),  ('Not detected', 'Triple coincidence', 'Outside template bank'),scatterpoints=1, ncol=1, fontsize=12)
            plt.tight_layout()
    
    @staticmethod
    def plotBKGvetoes(ts_data, ifo, t_bkg, t_search, snr_veto, delta_m):
        x, y_obs, p_star, ylabel = Measure.getFARandThreshold(ts_data, ifo, t_bkg, t_search)
        plt.plot(x, y_obs, c='cornflowerblue', label=r'No vetoes', alpha=0.5)
        plt.scatter(p_star, 0.01, marker='*', c='cornflowerblue', zorder=10,
                    label='FAR= 0.01 at '+str(np.round(p_star, 2)))

        if len(ifo) == 4:
            ts_data = SNRveto(snr_veto, ts_data, [ifo[:2].replace('1', ""), ifo[2:].replace('1', "")])
        if len(ifo) == 6:
            ts_data = SNRveto(snr_veto, ts_data, ['H1', 'L1', 'V1'])

        x, y_obs, p_star, _ = Measure.getFARandThreshold(ts_data, ifo, t_bkg, t_search)
        plt.plot(x, y_obs, c='darkorange', label=r'$SNR$ vetoes', alpha=0.5)
        plt.scatter(p_star, 0.01, marker='*', c='darkorange', zorder=10,
                   label='FAR= 0.01 at '+str(np.round(p_star, 2)))

        if len(ifo) == 4:
            ts_data = CoincMass(ts_data, [ifo[:2].replace('1', ""),
                                          ifo[2:].replace('1', "")], delta_m, delta_m).filterCoincMass()
        if len(ifo) == 6:
            ts_data = CoincMass(ts_data, ['H1', 'L1'], delta_m, delta_m).filterCoincMass()
            ts_data = CoincMass(ts_data, ['L1', 'V1'], delta_m, delta_m).filterCoincMass()
        x, y_obs, p_star, _ = Measure.getFARandThreshold(ts_data, ifo, t_bkg, t_search)
        plt.plot(x, y_obs, c='tomato', label=r'$\Delta m, SNR$ vetoes', alpha=0.5)
        plt.scatter(p_star, 0.01, marker='*', zorder=10,c='tomato',
                    label='FAR= 0.01 at '+str(np.round(p_star, 2)))
        plt.yscale('log')
        plt.xlabel(r'$-\log{(1 - \bar{P}_{inj})}$')
        plt.ylabel(ylabel)
        plt.legend()
        
    @staticmethod
    def FineTuneBKG(all_tmps, bkg, ifos, t_search, t_bkg, xlabel):
        binning = Binning()
        # Generate some random data for demonstration
        if xlabel == r'$SNR$':
            sigma13, ifnon =  [None] + np.arange(6, 4, -0.01).tolist(), 6.5
            cbarticks, cbarlabels = [ifnon, 6, 5.5, 4.5, 4], ['None', 6, 5.5, 4.5, 4]
            name, value = 'snr', 4.62
        if xlabel == r'$\Delta m$':
            sigma13, ifnon =  [None] + np.arange(20, 0.8, -0.1).tolist(), 30
            cbarticks, cbarlabels = [ifnon, 20, 15, 10, 5, 1], ['None', 20, 15, 10, 5, 1]
            name, value = 'mass', 12.7
        
        far = [0.5,0.25, 0.1, 5e-2, 1e-2,5e-3, 1e-3, 5e-4,1e-4]
        colors = plt.cm.viridis(np.linspace(1, 0, len(sigma13)))[::-1]
        cmap = LinearSegmentedColormap.from_list('custom', colors[::-1], N=len(sigma13))
        auc = []
        
        for c, s1 in zip(colors, sigma13):
            tprs, fprs = [], []
            pos, neg = all_tmps.copy(), bkg.copy()
        
            if s1 is not None:
                if xlabel == r'$SNR$':
                    if len(ifos) == 4:
                        pos = SNRveto(s1, pos, [ifos[:2], ifos[2:]])
                        neg = SNRveto(s1, neg, [ifos[:2].replace('1', ""),
                                                  ifos[2:].replace('1', "")]) #FIXME
                    if len(ifos) == 6:
                        pos = SNRveto(s1, pos, ['H1', 'L1', 'V1'])
                        neg = SNRveto(s1, neg, ['H1', 'L1', 'V1']) #FIXME
                if xlabel == r'$\Delta m$':
                    if len(ifos) == 4:
                        pos = CoincMass(pos, [ifos[:2], ifos[2:]], s1, s1).filterCoincMass()
                        neg = CoincMass(neg, [ifos[:2].replace('1', ""),
                                              ifos[2:].replace('1', "")], s1, s1).filterCoincMass()
                    if len(ifos) == 6:
                        pos = CoincMass(pos, ['H1', 'L1'], s1, s1).filterCoincMass()
                        pos = CoincMass(pos, ['L1', 'V1'], s1, s1).filterCoincMass()
                        neg = CoincMass(neg, ['H1', 'L1'], s1, s1).filterCoincMass()
                        neg = CoincMass(neg, ['L1', 'V1'], s1, s1).filterCoincMass()
            x, y, p = binning.fit_and_binning(data=neg['rank_stat_'+ifos], rank=3)
            y_obs, ylabel = binning.yVar(y, 'far', t_bkg, t_search)
            for f in far:
                p_star = Measure.FAR2stat(f, x, y_obs)
                tpr = Measure.TPR(pos, ifos, p_star=p_star)[0]
                fpr = Measure.FPR(neg, ifos, p_star=p_star)
                tprs.append(tpr)
                fprs.append(fpr)
            print(s1,np.round(metrics.auc(fprs, tprs),2),  p_star)
            plt.plot(fprs, tprs, marker='o', c=c, markersize=5,
                     label=str(s1)+' '+str(np.round(metrics.auc(fprs, tprs),2)))
            auc.append(np.round(metrics.auc(fprs, tprs),2))
        
        # Plot the color bar
        sm = ScalarMappable(cmap=cmap)
        sigma13[0] = ifnon
        sm.set_array(sigma13)
        cbar = plt.colorbar(sm)
        cbar.set_label(xlabel)
        cbar.set_ticks(cbarticks),cbar.set_ticklabels(cbarlabels)
        plt.xlabel(r'FPR'), plt.ylabel('TPR')
        
        path_img = '/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/FAR/closedbox/' 
        # Save the figure as PNG
        plt.savefig(path_img +'roc_'+name+'_'+ifos+'.png', dpi=300)  # Set dpi for high resolution
        plt.savefig(path_img +'roc_'+name+'_'+ifos+'.pdf')
        plt.show()
        return sigma13, auc

    @staticmethod
    def plotSpect2coinc(ifo1, ifo2, t1, t2, tw=2):
        fig, ax = plt.subplots(1, 2, dpi=120, figsize=(10, 6))
        for i, ifo, t_star in zip([0, 1], [ifo1, ifo2], [t1, t2]):
            strain = TimeSeries.fetch_open_data(ifo, t_star - tw, t_star + tw)
            # Perform q-transform
            q_scan = strain.q_transform(qrange=[4, 64], frange=[10, 2048],
                                     tres=0.002, fres=0.5, whiten=True)
    
            # Plot spectrogram
            ax[i].imshow(q_scan, cmap='viridis',label=ifo)
            ax[i].axvline(t_star, c='crimson', alpha=0.5, linewidth=0.5)
            ax[i].set_yscale('log', base=2)
            ax[i].set_xscale('linear')
            ax[0].set_ylabel('Frequency (Hz)', fontsize=14)
            ax[i].yaxis.set_major_formatter(ScalarFormatter())
            ax[i].title.set_text(str(t_star))
            ax[i].colorbar(clim=[0, 25.5])
            ax[1].grid(True)
            ax[1].set_yticklabels([])
        plt.show()