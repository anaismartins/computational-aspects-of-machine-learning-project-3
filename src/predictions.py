from torch_utils import load_model, predictions
from utils import prepareUnknown, prepareKnown

run = 'O3b'
for tw in [0.05, 0.1, 0.2, 0.5]:
    for ifo in ['H1', 'L1', 'V1']:
        print(tw, ifo)

        # Call models
        model = load_model(tw, ifo)

        # Prepare unknown data
        d_, d = prepareKnown(tw, ifo, run)

        # Predict
        ypred, label_pred = predictions(d_, model)

        for i in range(ypred.shape[1]):
            d['Prob'+str(i)] = ypred[:, i]

        d.to_csv('/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw'+str(tw)+'/predictions/pred_known_'+ifo+'_'+run+'.csv')

"""
    
tw = 0.05
for tw in [0.05, 0.1, 0.2, 0.5]:
    for ifo in ['H1', 'L1', 'V1']:
        print(tw, ifo)

        # Call models
        model = load_model(tw, ifo)

        # Prepare unknown data
        d_, d, t = prepareUnknown(tw, ifo)

        # Predict
        ypred, label_pred = predictions(d_, model)

        for i in range(ypred.shape[1]):
            d['Prob'+str(i)] = ypred[:, i]
        d = d.join(t)
        d.to_csv('/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw'+str(tw)+'/predictions/pred_unknown_'+ifo+'.csv')
"""
