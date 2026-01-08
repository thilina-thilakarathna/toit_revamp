
import pandas as pd
import time
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest

from components.general_operations import GeneralOp

class BaselineIFUM:
    def __init__(self):
        pass
        
    def baseline_detection(self, tampered_data):
        print("Detection IFUM")
        general = GeneralOp()
        microcell_data = {}
        time_val=[]
        for key2, df2 in tampered_data.items():
            start_time = time.time()
            # print("Baseline Microcell:"+key2)
            temp_provider_dfs = {}
            unique_keys = df2.providerid.unique()
            for provider in unique_keys:
                temp_provider_dfs["{}".format(provider)] = df2[df2.providerid == provider]

            temp_provider_result_dfs = {}
            for key_provider, df_provider in temp_provider_dfs.items():
                df_provider = df_provider.reset_index(drop=True)
                df_provider = df_provider.copy() 
                X = df_provider[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                        # Calculate metrics
                        # accuracy_ifum = accuracy_score(df_to_clust_ifum['true_label'], df_to_clust_ifum['Predicted_Label'])
                        # precision_ifum = precision_score(df_to_clust_ifum['true_label'], df_to_clust_ifum['Predicted_Label'], pos_label='T', zero_division=0)
                        # recall_ifum = recall_score(df_to_clust_ifum['true_label'], df_to_clust_ifum['Predicted_Label'], pos_label='T', zero_division=0)

                        # accuracy_list_ifum.append(accuracy_ifum)
                        # precision_list_ifum.append(precision_ifum)
                        # recall_list_ifum.append(recall_ifum)
                        # all_df_to_clust_ifum.append(df_to_clust_ifum)
                


                # features = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']
                # X = df_provider[features]
                
                # Custom check for identical records
                if len(X.drop_duplicates()) == 1:
                    # print("All records are identical in provider:", key_provider)
                    # You can choose to flag this as tampered or take other actions
                    df_provider['is_outlier'] = -1  # Assuming -1 represents tampered data
                else:
                    # isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                    # isolation_forest.fit(X)
                    # outliers = isolation_forest.predict(X)
                    # df_provider['is_outlier'] = outliers
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    labels = iso_forest.fit_predict(X_scaled)
                    df_provider['is_outlier'] = labels
                    # df_provider['is_outlier'] = df_provider['Cluster'].map({1: 'C', -1: 'T'})
                
                df_provider['label'] = df_provider['is_outlier'].map({1: 'C', -1: 'T'})
                temp_provider_result_dfs[key_provider] = df_provider
                
            combined_microcell_df = general.dictionary_to_merged_df(temp_provider_result_dfs)
            microcell_data[key2] = combined_microcell_df
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_val.append([df2.shape[0],elapsed_time])
            # print(df2.shape[0],elapsed_time)
        return general.dictionary_to_merged_df(microcell_data),time_val
    




