import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from components.general_operations import GeneralOp

class BaselineLOF:
    def __init__(self):
        pass
        
    def baseline_detection(self, tampered_data):
        print("Detection LOF")
        general = GeneralOp()
        microcell_data = {}
        for key2, df2 in tampered_data.items():
            # print("LOF Baseline Microcell:"+key2)
            temp_provider_dfs = {}
            unique_keys = df2.providerid.unique()
            for provider in unique_keys:
                temp_provider_dfs["{}".format(provider)] = df2[df2.providerid == provider]

            temp_provider_result_dfs = {}
            for key_provider, df_provider in temp_provider_dfs.items():
                df_provider = df_provider.reset_index(drop=True)
                df_provider = df_provider.copy() 
                # features = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']
                # X = df_provider[features]

                X = df_provider[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                        
                
    
                # Custom check for identical records
                if len(X.drop_duplicates()) == 1:
                    df_provider['is_outlier'] = -1  # Assuming -1 represents tampered data
                else:

                    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
                    labels = lof.fit_predict(X_scaled)
                    df_provider['Cluster'] = labels
                    # df_provider['Predicted_Label'] = df_to_clust_lofm['Cluster'].map({1: 'C', -1: 'T'})
                    
                    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
                    outliers = lof.fit_predict(X)
                    df_provider['is_outlier'] = outliers
                
                df_provider['label'] = df_provider['is_outlier'].map({1: 'C', -1: 'T'})
                temp_provider_result_dfs[key_provider] = df_provider
                
            combined_microcell_df = general.dictionary_to_merged_df(temp_provider_result_dfs)
            microcell_data[key2] = combined_microcell_df

        return general.dictionary_to_merged_df(microcell_data)


