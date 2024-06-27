from torch_utils import load_model, predictions
from utils import prepareUnknown, prepareKnown
import numpy as np

run, dogs = 'O3a', False
"""
for run in ['O3b', 'O3a']:
    for tw in [0.05]:
        for ifo in ['H1', 'L1', 'V1']:
            print(tw, ifo)

            # Call models
            model = load_model(tw, ifo)
            
            # Prepare unknown data
            d_, d = prepareKnown(tw, ifo, run, dogs=dogs)
            print(d_.shape)
            # Predict
            #ypred, label_pred = predictions(d_, model)
            ypred = predictions(d_, model)
            for i in range(ypred.shape[1]):
                d['Prob'+str(i)] = ypred[:, i]

            d.to_csv('/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw'+str(tw)+'/predictions/pred_nodogs_known_'+ifo+'_'+run+'.csv')
"""


dogs = False
tw = 0.05
for tw in [0.05]:
    for ifo in ['H1', 'L1', 'V1']:
        print(tw, ifo)

        # Call models
        model = load_model(tw, ifo)

        # Prepare unknown data
        d_, d, t = prepareUnknown(tw, ifo)
        print(t.iloc[0])
        idx = np.where(t['Cluster time'] - 1242442967.4 > 0)[0][0]

        print(t.iloc[idx]['Cluster time']- 1242442967.4)
        # Predict
        #ypred, label_pred = predictions(d_, model)
        ypred = predictions(d_, model)

        for i in range(ypred.shape[1]):
            d['Prob'+str(i)] = ypred[:, i]
        d = d.join(t)
        print(d.columns)
        d.to_csv('/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw'+str(tw)+'/predictions/pred_logs_unknown_'+ifo+'.csv')

