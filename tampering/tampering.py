
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
                        round(dftamper.loc[line, 'speed'] * (1 + result[0]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'latency'] = min(
                        round(dftamper.loc[line, 'latency'] * (1 + result[1]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'bandwidth'] = min(
                        round(dftamper.loc[line, 'bandwidth'] * (1 + result[2]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'coverage'] = min(
                        round(dftamper.loc[line, 'coverage'] * (1 + result[3]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'reliability'] = min(
                        round(dftamper.loc[line, 'reliability'] * (1 + result[4]*(each_attribute / 100))), val
                    )
                    dftamper.loc[line, 'security'] = min(
                        round(dftamper.loc[line, 'security'] * (1 + result[5]*(each_attribute / 100))), val
                    )

                    dftamper.loc[line, 'true_label'] = 'T'
                    dftamper.loc[line, 'true_label_spa'] = 'T'

        elif type == "S":
            if not dftamper.empty:

                p = 0.3  # probability of tampering (controls stealth level)

                # dftamper['TS'] = sum(
                #         weights[col] * dftamper[col] for col in weights
                #     )
                
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



        return dftamper


    def bma_tampering(self, data, sp_percent, type, each_attribute=40, val=2, sig=[0.3,0.1,0.2,0.1,0.1,0.2]):
        result_df = data.copy()
        unique_microcells = data['gen_microcell'].unique()
        sp_amount = round(len(unique_microcells) * (sp_percent / 100))
        # print(sp_amount)
        

        sampled_microcells = set(random.sample(list(unique_microcells), sp_amount))
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
                        dftamper.loc[line, 'speed'] = min(round(dftamper.loc[line, 'speed'] * (1 - result[0]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'latency'] = min(round(dftamper.loc[line, 'latency'] * (1 - result[1]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'bandwidth'] = min(round(dftamper.loc[line, 'bandwidth'] * (1 - result[2]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'coverage'] = min(round(dftamper.loc[line, 'coverage'] * (1 - result[3]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'reliability'] = min(round(dftamper.loc[line, 'reliability'] * (1 - result[4]*(each_attribute / 100))), val)
                        dftamper.loc[line, 'security'] = min(round(dftamper.loc[line, 'security'] * (1 - result[5]*(each_attribute / 100))), val)
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
                    # print(ts_median)
                            # provider-level mean behavior (used for soft degradation)
                        attr_means = group_df[['speed', 'latency', 'bandwidth','coverage', 'reliability', 'security']].mean()
                        for idx in group_df.index:
                            if (group_df.loc[idx, 'TS'] > ts_median and np.random.rand() < p):      
                                for attr in attr_means.index:
                                    dftamper.loc[idx, attr] = round((dftamper.loc[idx, attr] + attr_means[attr]) / 2, 2)
                                    dftamper.loc[idx, 'true_label'] = 'T'
                                    dftamper.loc[idx, 'true_label_bma'] = 'T'







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
 
        
   
        
        # attributes = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']
 
        # for microcell in sampled_microcells:
        #     mask = result_df['gen_microcell'] == microcell
        #     dftamper = result_df.loc[mask].copy().reset_index(drop=True)

        #     if type == "N":
        #         num_rows_to_tamper = int(0.5 * len(dftamper))
        #         if num_rows_to_tamper > 0:
        #             rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
        #             dftamper.loc[rows_to_tamper, attributes] = 3.5
        #             dftamper.loc[rows_to_tamper, 'true_label'] = 'T'
        #             dftamper.loc[rows_to_tamper, 'true_label_bma'] = 'T'

            

        #     result_df.loc[mask, dftamper.columns] = dftamper

        # # for microcell in sampled_microcells:
        # #     mask = result_df['gen_microcell'] == microcell
        # #     dftamper = result_df.loc[mask].copy().reset_index(drop=True)
        # #     if dftamper.empty:
        # #         continue

        # #     if type == "N":
        # #         num_rows_to_tamper = int(0.5 * len(dftamper))
        # #         if num_rows_to_tamper > 0:
        # #             rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
        # #             dftamper.loc[rows_to_tamper, attributes] = 3.5
        # #             dftamper.loc[rows_to_tamper, 'true_label'] = 'T'
        # #             dftamper.loc[rows_to_tamper, 'true_label_bma'] = 'T'

        # #     elif type == "K":
        # #         highest_3 = sorted(sig, reverse=True)[:3]
        # #         result = [1 if value in highest_3 else 0 for value in sig]
        # #         for i, attr in enumerate(attributes):
        # #             dftamper[attr] = (dftamper[attr] * (1 - result[i] * (each_attribute / 100))).round().clip(lower=0)
        # #         dftamper['true_label'] = 'T'
        # #         dftamper['true_label_bma'] = 'T'

        # #     elif type == "S":
        # #         grouped = dftamper.groupby('providerid')
        # #         for _, group_df in grouped:
        # #             if len(group_df) >= 2:
        # #                 target_index = group_df[attributes].sum(axis=1).idxmax()
        # #                 dftamper.loc[target_index, 'speed'] = group_df['speed'].mean()
        # #                 dftamper.loc[target_index, 'latency'] = group_df['latency'].mean()
        # #                 dftamper.loc[target_index, 'bandwidth'] = group_df['bandwidth'].mean()
        # #                 dftamper.loc[target_index, 'coverage'] = group_df['coverage'].mean()
        # #                 dftamper.loc[target_index, 'reliability'] = group_df['reliability'].mean()
        # #                 dftamper.loc[target_index, 'security'] = group_df['security'].mean()
        # #                 dftamper.loc[target_index, 'true_label'] = 'T'
        # #                 dftamper.loc[target_index, 'true_label_bma'] = 'T'

        # #     result_df.loc[mask, dftamper.columns] = dftamper

        # return result_df