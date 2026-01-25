
import numpy as np
import random

import pandas as pd


class Tampering:
    def __init__(self):

        pass

    def spa_tampering(self, data, type, each_attribute=10, val=5, sig=[1/6,1/6,1/6,1/6,1/6,1/6]):

 
        dftamper = data

        if type == "N":
            if not dftamper.empty:
                num_rows_to_tamper = int(0.5 * len(dftamper))
                rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
                dftamper.loc[rows_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = 4.8
                dftamper.loc[rows_to_tamper, 'true_label'] = 'T'
                dftamper.loc[rows_to_tamper, 'true_label_spa'] = 'T'
            
        elif type == "K":
            if not dftamper.empty:

                num_rows_to_tamper = int(0.5 * len(dftamper))
                rows_to_tamper = np.random.choice(
                    dftamper.index,
                    num_rows_to_tamper,
                    replace=False
                )

                highest_3 = sorted(sig, reverse=True)[:3]
                result = [1 if value in highest_3 else 0 for value in sig]

                for line in rows_to_tamper:
                    dftamper.loc[line, 'speed'] = min(
                        (dftamper.loc[line, 'speed'] * (1 + result[0]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'latency'] = min(
                        (dftamper.loc[line, 'latency'] * (1 + result[1]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'bandwidth'] = min(
                        (dftamper.loc[line, 'bandwidth'] * (1 + result[2]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'coverage'] = min(
                        (dftamper.loc[line, 'coverage'] * (1 + result[3]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'reliability'] = min(
                        (dftamper.loc[line, 'reliability'] * (1 + result[4]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'security'] = min(
                        (dftamper.loc[line, 'security'] * (1 + result[5]*(each_attribute / 100))), val
                    )

                    dftamper.loc[line, 'true_label'] = 'T'
                    dftamper.loc[line, 'true_label_spa'] = 'T'

        elif type == "S":
            if not dftamper.empty:

                p = 0.3

                dftamper = self.trust_astimation(dftamper.copy(),[0.3,0.1,0.2,0.1,0.1,0.2])

                grouped = dftamper.groupby('providerid')
                for providerid, group_df in grouped:
                        # need at least 2 records to compute meaningful statistics
                    if len(group_df) < 2:
                        continue
                        # median TS used to identify weak trust records we assume they have knowlege about the importance of attributes


                    ts_median = group_df['TS'].median()
                        # provider-level mean behavior (used for smoothing)
                    attr_means = group_df[['speed', 'latency', 'bandwidth','coverage', 'reliability', 'security']].mean()
                    for idx in group_df.index:
                            # probabilistically tamper only low-TS records
                        if (group_df.loc[idx, 'TS'] < ts_median and np.random.rand() < p):
                            for attr in attr_means.index:
                                dftamper.loc[idx, attr] = round((dftamper.loc[idx, attr] + attr_means[attr]) / 2, 2)
                                dftamper.loc[idx, 'true_label'] = 'T'
                                dftamper.loc[idx, 'true_label_spa'] = 'T'
                dftamper = dftamper.drop(columns=['TS'])



        return dftamper


    def bma_tampering(self, data, sp_percent, type, each_attribute=20, val=2.5, sig=[0.3,0.1,0.2,0.1,0.1,0.2]):

        
        result_df = data.copy()
        # unique_microcells = data['gen_microcell'].unique()
        # sp_amount = round(len(unique_microcells) * (sp_percent / 100))
        # # print(sp_amount)
        

        # sampled_microcells = set(random.sample(list(unique_microcells), sp_amount))
        


       
        unique_microcells = data['gen_microcell'].unique()
        sp_amount = round(len(unique_microcells) * (sp_percent / 100))

        # sort microcells by number of records (largest first)
        microcell_sizes = (
            data['gen_microcell']
            .value_counts()
            .sort_values(ascending=False)
        )
        sorted_microcells = microcell_sizes.index.tolist()

        # deterministic selection
        sampled_microcells = set(sorted_microcells[:sp_amount])

        tampered_data = pd.DataFrame()

        for key in sampled_microcells:
            dftamper = result_df[result_df['gen_microcell'] == key].copy()
            dftamper=dftamper.reset_index(drop=True)
  
            if type == "N":
                if not dftamper.empty:
                    num_rows_to_tamper = int(0.5 * len(dftamper))
                    rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
                    dftamper.loc[rows_to_tamper, ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']] = 3.5
                    dftamper.loc[rows_to_tamper, 'true_label'] = 'T'
                    dftamper.loc[rows_to_tamper, 'true_label_bma'] = 'T'
            

            elif type == "K":     
                if not dftamper.empty:
                    num_rows_to_tamper = int(0.5 * len(dftamper))
                    rows_to_tamper = np.random.choice(
                    dftamper.index,
                    num_rows_to_tamper,
                    replace=False )
                    highest_3 = sorted(sig, reverse=True)[:3]
                    result = [1 if value in highest_3 else 0 for value in sig]
                    for line in rows_to_tamper:
                        dftamper.loc[line, 'speed'] = max((dftamper.loc[line, 'speed'] * (1 - result[0]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'latency'] = max((dftamper.loc[line, 'latency'] * (1 - result[1]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'bandwidth'] = max((dftamper.loc[line, 'bandwidth'] * (1 - result[2]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'coverage'] = max((dftamper.loc[line, 'coverage'] * (1 - result[3]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'reliability'] = max((dftamper.loc[line, 'reliability'] * (1 - result[4]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'security'] = max((dftamper.loc[line, 'security'] * (1 - result[5]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'true_label'] = 'T'
                        dftamper.loc[line, 'true_label_bma'] = 'T'
            
            elif type == "S":
                if not dftamper.empty:
                    dftamper = self.trust_astimation(dftamper.copy(),[0.3,0.1,0.2,0.1,0.1,0.2])
                    p = 0.3  # probability of bad-mouthing (stealth level)
                    grouped = dftamper.groupby('providerid')
                    for providerid, group_df in grouped:
                        if len(group_df) < 2:
                            continue
                        ts_median = group_df['TS'].median()
                        attr_means = group_df[['speed', 'latency', 'bandwidth','coverage', 'reliability', 'security']].mean()
                        for idx in group_df.index:
                            if (group_df.loc[idx, 'TS'] > ts_median and np.random.rand() < p):      
                                for attr in attr_means.index:
                                    dftamper.loc[idx, attr] = round((dftamper.loc[idx, attr] + attr_means[attr]) / 2, 2)
                                    dftamper.loc[idx, 'true_label'] = 'T'
                                    dftamper.loc[idx, 'true_label_bma'] = 'T'
                    dftamper = dftamper.drop(columns=['TS'])

            tampered_data = pd.concat([tampered_data, dftamper], ignore_index=True) 
        untampered_data = result_df[~result_df['gen_microcell'].isin(sampled_microcells)]
        return pd.concat([untampered_data, tampered_data], ignore_index=True).reset_index(drop=True)

    def trust_astimation(self,datain,weight_matrix):
            datain['TS'] = (
            datain['speed'] * weight_matrix[0] +
            datain['latency'] * weight_matrix[1] +
            datain['bandwidth'] * weight_matrix[2] +
            datain['coverage'] * weight_matrix[3] +
            datain['reliability'] * weight_matrix[4] +
            datain['security'] * weight_matrix[5]) 
            return datain
 
        
   