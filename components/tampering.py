
import numpy as np
import random



class Tampering:
    def __init__(self):
        pass
    def tamper_data1(self, data, sp_percent, type, each_attribute=10, val=5,sig=[1/6,1/6,1/6,1/6,1/6,1/6]):
        sp_amount = len(data) * (sp_percent / 100)
        sampled_keys = random.sample(list(data.keys()), round(sp_amount))
        tampered_data = {}


        for key in sampled_keys:
            df = data[key]
            dftamper = df.copy()
            dftamper=dftamper.reset_index(drop=True)
            if type == "N1":
                if not dftamper.empty:
                    dftamper[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = 4.8
                    dftamper['true_label'] = 'T'
            elif type == "N2":
                if not dftamper.empty:
                    num_rows_to_tamper = int(0.5 * len(dftamper))
                    rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
                    dftamper.loc[rows_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = 4.8
                    dftamper.loc[rows_to_tamper, 'true_label'] = 'T'

            elif type == "K1":
                if not dftamper.empty:
                    dftamper[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] *= (1 + (each_attribute / 100))
                    dftamper[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = np.minimum(dftamper[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']], val)
                    dftamper['true_label'] = 'T'

            elif type == "K2":
                if not dftamper.empty:
                    unique_providers = dftamper['providerid'].unique()
                    num_providers_to_tamper = int(0.5 * len(unique_providers))
                    providers_to_tamper = random.sample(list(unique_providers), num_providers_to_tamper)
                    records_to_tamper = dftamper[dftamper['providerid'].isin(providers_to_tamper)].index
                    dftamper.loc[records_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] *= (1 + (each_attribute / 100))
                    dftamper.loc[records_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = np.minimum(dftamper.loc[records_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']], val)
                    dftamper.loc[records_to_tamper, 'true_label'] = 'T'
            
            elif type == "K3":
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
            elif type == "S1":
                if not dftamper.empty:
                    grouped = dftamper.groupby('providerid')
                    for providerid, group_df in grouped:
                        if len(group_df) >= 2:
                            lowest_index = group_df['TS'].idxmin()
                            highest_index = group_df['TS'].idxmax()
                            dftamper.loc[lowest_index, 'speed'] = dftamper.loc[highest_index, 'speed'] 
                            dftamper.loc[lowest_index, 'latency'] = dftamper.loc[highest_index, 'latency'] 
                            dftamper.loc[lowest_index, 'bandwidth'] = dftamper.loc[highest_index, 'bandwidth'] 
                            dftamper.loc[lowest_index, 'coverage'] = dftamper.loc[highest_index, 'coverage'] 
                            dftamper.loc[lowest_index, 'reliability'] = dftamper.loc[highest_index, 'reliability'] 
                            dftamper.loc[lowest_index, 'security'] = dftamper.loc[highest_index, 'security']  
                            dftamper.loc[lowest_index, 'true_label'] = 'T'
            elif type == "S2":

                if not dftamper.empty:
                    grouped = dftamper.groupby('providerid')
                    for providerid, group_df in grouped:
                        if len(group_df) >= 2:
                            lowest_index = group_df['TS'].idxmin()

                            dftamper.loc[lowest_index, 'speed'] = group_df['speed'].mean()
                            dftamper.loc[lowest_index, 'latency'] = group_df['latency'].mean() 
                            dftamper.loc[lowest_index, 'bandwidth'] = group_df['bandwidth'].mean() 
                            dftamper.loc[lowest_index, 'coverage'] = group_df['coverage'].mean() 
                            dftamper.loc[lowest_index, 'reliability'] = group_df['reliability'].mean() 
                            dftamper.loc[lowest_index, 'security'] =  group_df['security'].mean() 
                            dftamper.loc[lowest_index, 'true_label'] = 'T'
            tampered_data[key] = dftamper


        untampered_data = {key: value for key, value in data.items() if key not in tampered_data}

        # print("*************")
        # for key in untampered_data:
        #     print(key)
        #     df = data[key]
        #     print(df.shape[0])
        # print("tampered",len(tampered_data))
        # print("untampered",len(untampered_data))
        return {**untampered_data, **tampered_data}
    

        




