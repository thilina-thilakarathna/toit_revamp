
import numpy as np
import random



class Tampering:
    def __init__(self):
        pass
    def tamper_data(self, data, sp_percent, type, each_attribute=10, val=5,sig=[1/6,1/6,1/6,1/6,1/6,1/6]):
        sp_amount = len(data) * (sp_percent / 100)
        sampled_keys = random.sample(list(data.keys()), round(sp_amount))
        tampered_data = {}

        for key in sampled_keys:
            df = data[key]
            dftamper = df.copy()
            dftamper=dftamper.reset_index(drop=True)
        
            if type == "N":
                if not dftamper.empty:
                    num_rows_to_tamper = int(0.5 * len(dftamper))
                    rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
                    dftamper.loc[rows_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = 4.8
                    dftamper.loc[rows_to_tamper, 'true_label'] = 'T'


            elif type == "K":
                highest_3 = sorted(sig, reverse=True)[:3]
                result = [1 if value in highest_3 else 0 for value in sig]
                if not dftamper.empty:
                    for line in range(0, dftamper.shape[0]):
                        dftamper.loc[line, 'speed'] = min(round(dftamper.loc[line, 'speed'] * (1 + result[0]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'latency'] = min(round(dftamper.loc[line, 'latency'] * (1 + result[1]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'bandwidth'] = min(round(dftamper.loc[line, 'bandwidth'] * (1 + result[2]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'coverage'] = min(round(dftamper.loc[line, 'coverage'] * (1 + result[3]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'reliability'] = min(round(dftamper.loc[line, 'reliability'] * (1 + result[4]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'security'] = min(round(dftamper.loc[line, 'security'] * (1 + result[5]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'true_label'] = 'T'
    
            elif type == "S":
                if not dftamper.empty:
                    grouped = dftamper.groupby('providerid')
                    for providerid, group_df in grouped:
                        if len(group_df) >= 2:
                            lowest_index = group_df[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']].sum(axis=1).idxmin()

                            dftamper.loc[lowest_index, 'speed'] = group_df['speed'].mean()
                            dftamper.loc[lowest_index, 'latency'] = group_df['latency'].mean() 
                            dftamper.loc[lowest_index, 'bandwidth'] = group_df['bandwidth'].mean() 
                            dftamper.loc[lowest_index, 'coverage'] = group_df['coverage'].mean() 
                            dftamper.loc[lowest_index, 'reliability'] = group_df['reliability'].mean() 
                            dftamper.loc[lowest_index, 'security'] =  group_df['security'].mean() 
                            dftamper.loc[lowest_index, 'true_label'] = 'T'
            tampered_data[key] = dftamper


        untampered_data = {key: value for key, value in data.items() if key not in tampered_data}


        return {**untampered_data, **tampered_data}
    

        