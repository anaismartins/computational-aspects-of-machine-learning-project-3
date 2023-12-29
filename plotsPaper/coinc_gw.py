import pandas as pd
import numpy as np

events = pd.read_csv('/data/gravwav/lopezm/Projects/GlitchBank/allevents.csv')
for ifos in ['H1L1', 'H1V1', 'L1V1']:
    ifo1, ifo2 = ifos[:2], ifos[2:]

    df = pd.read_csv('../output_new/tw0.05/predictions/pred_unknown_'+ifos+'.csv')
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df['Av num triggers'] = (df['Num triggers_'+ifo1]+df['Num triggers_'+ifo2])/2
    df['Av prob0'] = (df['Prob0_'+ifo1]+df['Prob0_'+ifo2])/2
    df = df.sort_values(by='Av num triggers', ascending=False)
    df = df.loc[(df['Av prob0']>0.9)]
    df = df.loc[(df['Av num triggers']>3)]

    gws, catalg_gws = list(), list()
    c = 0
    times1= df['Cluster time_'+ifo1]
    for i in range(len(df)):

        #tmp = df.iloc[i]
        t1 = times1.iloc[i]

        for e in range(len(events)):

            if (int(events.iloc[e]['GPS'])) - int(t1) == 0:
                gws.append(np.asarray(df.iloc[i].values))
                catalg_gws.append(events.iloc[e].values)
                #print(i, 'GW detected: ',events['commonName'][e],events['catalog.shortName'][e], tmp['Prob0_'+ifo1], tmp['Prob0_'+ifo2])
                c = c +1
    gws = pd.DataFrame(gws, columns=df.columns)
    catalg_gws = pd.DataFrame(catalg_gws, columns=events.columns)
    discovert_2coinc = pd.merge(gws, catalg_gws, left_index=True, right_index=True)
    discovert_2coinc = discovert_2coinc.drop_duplicates(subset=['Cluster time_'+ifo1]).reset_index(drop=True)
    discovert_2coinc.to_csv('gw_known_discovert_'+ifos+'.csv')