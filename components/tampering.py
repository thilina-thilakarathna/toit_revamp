
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
        return {**untampered_data, **tampered_data}
    

        

# import numpy as np
# import pandas as pd
# import random

# class Tampering:
#     def __init__(self):
#         self.cols = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

#     def tamper_data1(self, data, sp_percent, tamper_type, each_attribute=10, val=5, sig=None):
#         if sig is None:
#             sig = [1/6] * 6
            
#         sp_amount = round(len(data) * (sp_percent / 100))
#         sampled_keys = random.sample(list(data.keys()), sp_amount)
        
#         # Use a copy to avoid side-effects on the original data dictionary
#         final_data = data.copy()

#         for key in sampled_keys:
#             df = data[key].copy().reset_index(drop=True)
#             if df.empty:
#                 continue

#             if tamper_type == "N1":
#                 df[self.cols] = 4.8
#                 df['true_label'] = 'T'

#             elif tamper_type == "N2":
#                 rows_to_tamper = df.sample(frac=0.5).index
#                 df.loc[rows_to_tamper, self.cols] = 4.8
#                 df.loc[rows_to_tamper, 'true_label'] = 'T'

#             elif tamper_type == "K1":
#                 df[self.cols] *= (1 + (each_attribute / 100))
#                 df[self.cols] = df[self.cols].clip(upper=val)
#                 df['true_label'] = 'T'

#             elif tamper_type == "K2":
#                 unique_p = df['providerid'].unique()
#                 p_to_tamper = random.sample(list(unique_p), int(0.5 * len(unique_p)))
#                 mask = df['providerid'].isin(p_to_tamper)
#                 df.loc[mask, self.cols] *= (1 + (each_attribute / 100))
#                 df.loc[mask, self.cols] = df.loc[mask, self.cols].clip(upper=val)
#                 df.loc[mask, 'true_label'] = 'T'

#             elif tamper_type == "K3":
#                 # Vectorize the attribute selection
#                 threshold = sorted(sig, reverse=True)[2] # Get the 3rd highest value
#                 multipliers = [(1 + (each_attribute / 100)) if s >= threshold else 1 for s in sig]
                
#                 for col, mult in zip(self.cols, multipliers):
#                     if mult > 1:
#                         df[col] = (df[col] * mult).round().clip(upper=val)
#                 df['true_label'] = 'T'

#             elif tamper_type == "S1":
#                 # Vectorized Grouping: find indices of min/max TS per provider
#                 idx_min = df.groupby('providerid')['TS'].idxmin()
#                 idx_max = df.groupby('providerid')['TS'].idxmax()
                
#                 # Filter out providers that only have 1 record (min and max would be the same)
#                 valid_mask = idx_min != idx_max
#                 idx_min, idx_max = idx_min[valid_mask], idx_max[valid_mask]
                
#                 # Assign values from max index to min index
#                 df.loc[idx_min, self.cols] = df.loc[idx_max, self.cols].values
#                 df.loc[idx_min, 'true_label'] = 'T'

#             elif tamper_type == "S2":
#                 # Vectorized Mean: calculate means per provider
#                 means = df.groupby('providerid')[self.cols].transform('mean')
#                 idx_min = df.groupby('providerid')['TS'].idxmin()
                
#                 # Only update if group size > 1 (optional, based on your original logic)
#                 counts = df.groupby('providerid')['TS'].transform('count')
#                 mask = (df.index.isin(idx_min)) & (counts >= 2)
                
#                 df.loc[mask, self.cols] = means.loc[mask]
#                 df.loc[mask, 'true_label'] = 'T'

#             final_data[key] = df

#         return final_data


