
import numpy as np
import random


class Tampering:
    def __init__(self):
        pass
    
    def spa_tampering(self, data, sp_percent, type, each_attribute=10, val=5, sig=[1/6,1/6,1/6,1/6,1/6,1/6]):
        """
        SPA (Self-Promoting Attack) tampering - increases attribute values.
        
        Args:
            data: DataFrame with a 'microcell' column
            sp_percent: Percentage of microcells to tamper
            type: Tampering type - "N" (Naive), "K" (Knowledgeable), "S" (Sophisticated)
            each_attribute: Percentage change for each attribute (default: 10)
            val: Maximum value cap (default: 5)
            sig: Significance weights for attributes (default: equal weights)
            
        Returns:
            Tampered DataFrame
        """
        return self._tamper_dataframe(data, sp_percent, type, each_attribute, val, sig, attack_type='spa')
    
    def bma_tampering(self, data, sp_percent, type, each_attribute=10, val=5, sig=[1/6,1/6,1/6,1/6,1/6,1/6]):
        """
        BMA (Bad Mouthing Attack) tampering - reduces attribute values instead of increasing.
        
        Args:
            data: DataFrame with a 'microcell' column
            sp_percent: Percentage of microcells to tamper
            type: Tampering type - "N" (Naive), "K" (Knowledgeable), "S" (Sophisticated)
            each_attribute: Percentage change for each attribute (default: 10)
            val: Maximum/minimum value cap (default: 5)
            sig: Significance weights for attributes (default: equal weights)
            
        Returns:
            Tampered DataFrame
        """
        return self._tamper_dataframe(data, sp_percent, type, each_attribute, val, sig, attack_type='bma')
    
    def _tamper_dataframe(self, df, sp_percent, type, each_attribute, val, sig, attack_type='spa'):
        """
        Tamper a DataFrame by sampling microcells and modifying their records.
        """
        if df is None or df.empty:
            return df.copy() if df is not None else df
        if 'microcell' not in df.columns:
            raise ValueError("Input DataFrame must include a 'microcell' column.")

        result_df = df.copy()
        unique_microcells = result_df['microcell'].unique()
        sp_amount = round(len(unique_microcells) * (sp_percent / 100))
        if sp_amount <= 0:
            return result_df

        sampled_microcells = set(random.sample(list(unique_microcells), sp_amount))
        for microcell in sampled_microcells:
            mask = result_df['microcell'] == microcell
            subset = result_df.loc[mask].copy()
            tampered_subset = self._tamper_subset(subset, type, each_attribute, val, sig, attack_type)
            result_df.loc[mask, tampered_subset.columns] = tampered_subset

        return result_df

    def _tamper_subset(self, dftamper, type, each_attribute, val, sig, attack_type='spa'):
        """
        Tamper a subset DataFrame (single microcell).
        """
        if dftamper.empty:
            return dftamper

        attributes = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

        if type == "N":
            num_rows_to_tamper = int(0.5 * len(dftamper))
            if num_rows_to_tamper > 0:
                rows_to_tamper = np.random.choice(dftamper.index, num_rows_to_tamper, replace=False)
                tamper_value = 4.8 if attack_type == 'spa' else 3.5
                dftamper.loc[rows_to_tamper, attributes] = tamper_value
                dftamper.loc[rows_to_tamper, 'true_label'] = 'T'

        elif type == "K":
            highest_3 = sorted(sig, reverse=True)[:3]
            result = [1 if value in highest_3 else 0 for value in sig]
            for i, attr in enumerate(attributes):
                if attack_type == 'spa':
                    dftamper[attr] = (dftamper[attr] * (1 + result[i] * (each_attribute / 100))).round().clip(upper=val)
                else:
                    dftamper[attr] = (dftamper[attr] * (1 - result[i] * (each_attribute / 100))).round().clip(lower=0)
            dftamper['true_label'] = 'T'

        elif type == "S":
            grouped = dftamper.groupby('providerid')
            for _, group_df in grouped:
                if len(group_df) >= 2:
                    if attack_type == 'spa':
                        target_index = group_df[attributes].sum(axis=1).idxmin()
                    else:
                        target_index = group_df[attributes].sum(axis=1).idxmax()
                    dftamper.loc[target_index, 'speed'] = group_df['speed'].mean()
                    dftamper.loc[target_index, 'latency'] = group_df['latency'].mean()
                    dftamper.loc[target_index, 'bandwidth'] = group_df['bandwidth'].mean()
                    dftamper.loc[target_index, 'coverage'] = group_df['coverage'].mean()
                    dftamper.loc[target_index, 'reliability'] = group_df['reliability'].mean()
                    dftamper.loc[target_index, 'security'] = group_df['security'].mean()
                    dftamper.loc[target_index, 'true_label'] = 'T'

        return dftamper
    
    # def tamper_data(self, data, sp_percent, type, each_attribute=10, val=5, sig=[1/6,1/6,1/6,1/6,1/6,1/6]):
    #     """
    #     Alias for spa_tampering() for backward compatibility.
    #     """
    #     return self.spa_tampering(data, sp_percent, type, each_attribute, val, sig)

        