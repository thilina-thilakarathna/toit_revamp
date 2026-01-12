
import time
import pandas as pd
from sklearn.cluster import DBSCAN



class Detection:
    def __init__(self):
        pass

    def detect_tampered_records(self,correct_data,tampered_data):
        general = GeneralOp()

        results_review ={}
        for key2, df2 in tampered_data.items():
            # start_time = time.time()
            df2=general.add_a_column_with_a_value(df2,'label','NA')
            # print("Microcell:"+key2)
            temp_provider_dfs={}
            unique_keys = df2.providerid.unique()
            for provider in unique_keys:
                temp_provider_dfs["{}".format(provider)] = df2[df2.providerid==provider]

            temp_provider_result_dfs={}
            for key_provider,df_provider in temp_provider_dfs.items():
                df2_received = df_provider[df_provider['origin'] == 'R']
                df2_generated = df_provider[df_provider['origin'] == 'G']
                df2_received=df2_received.reset_index(drop=True)
                if not df2_received.empty:
                    for i in range(len(df2_received)):
                        origin_record = correct_data[correct_data['serviceid']==df2_received.iloc[i]['serviceid']]  
                        if((df2_received.iloc[i]['speed']==origin_record.iloc[0]['speed']) and (df2_received.iloc[i]['latency']==origin_record.iloc[0]['latency'])and (df2_received.iloc[i]['bandwidth']==origin_record.iloc[0]['bandwidth'])and (df2_received.iloc[i]['coverage']==origin_record.iloc[0]['coverage'])and (df2_received.iloc[i]['reliability']==origin_record.iloc[0]['reliability'])and (df2_received.iloc[i]['security']==origin_record.iloc[0]['security'])):
                            df2_received.loc[i, 'label'] = 'C'
                        else:
                            df2_received.loc[i, 'label'] = 'T'
                df2_generated=df2_generated.reset_index(drop=True)
                if not df2_generated.empty:
                    for j in range(len(df2_generated)):
                        provider_other_df1 = correct_data[correct_data['providerid']==df2_generated.iloc[j]['providerid']]
                        provider_other_df = provider_other_df1[provider_other_df1['microcell']!=key2]
                        provider_other_df=provider_other_df.reset_index(drop=True)

                        if(provider_other_df.shape[0]>1):
                            df_to_clust = provider_other_df[[ 'speed','latency','bandwidth','coverage','reliability','security']]
                            dbscan = DBSCAN(eps=0.5, min_samples=5)
                            clusters = dbscan.fit_predict(df_to_clust)
                            outliers = provider_other_df.iloc[clusters == -1]
                            check_list = outliers['serviceid'].unique()
                            if df2_generated.iloc[j]['serviceid'] in check_list:
                                df2_generated.loc[j, 'label'] = 'S'
                            else:
                                df2_generated.loc[j, 'label'] = 'NS'
                        else:
                            df2_generated.loc[j, 'label'] = 'NS'

                t_count = (df2_received['label'] == 'T').sum()
                c_count = (df2_received['label'] == 'C').sum()
                if c_count + t_count != 0:
                    if((t_count/(c_count+t_count))<0.8):
                        df2_generated['label'].replace('S', 'C', inplace=True)
                        df2_generated['label'].replace('NS', 'C', inplace=True)
                    elif((t_count/(c_count+t_count))>0.8):
                        df2_generated['label'].replace('S', 'T', inplace=True)
                        df2_generated['label'].replace('NS', 'T', inplace=True)
                    
                    else:
                        df2_generated['label'].replace('S', 'T', inplace=True)
                        df2_generated['label'].replace('NS', 'C', inplace=True)
                else:
            
                    df2_generated['label'].replace('S', 'T', inplace=True)
                    df2_generated['label'].replace('NS', 'C', inplace=True)


                concatenated_df_provider = pd.concat([df2_generated, df2_received])
                temp_provider_result_dfs[key_provider] = concatenated_df_provider
            
            combined_microcell_df = general.dictionary_to_merged_df(temp_provider_result_dfs)
            results_review[key2] = combined_microcell_df
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(df2.shape[0],elapsed_time)
        return general.dictionary_to_merged_df(results_review) 
    
    def detect_tampered_records_tith_time(self,correct_data,tampered_data):
        general = GeneralOp()
        time_val = []

        results_review ={}
        for key2, df2 in tampered_data.items():
            start_time = time.time()
            df2=general.add_a_column_with_a_value(df2,'label','NA')
            # print("Microcell:"+key2)
            temp_provider_dfs={}
            unique_keys = df2.providerid.unique()
            for provider in unique_keys:
                temp_provider_dfs["{}".format(provider)] = df2[df2.providerid==provider]

            temp_provider_result_dfs={}
            for key_provider,df_provider in temp_provider_dfs.items():
                df2_received = df_provider[df_provider['origin'] == 'R']
                df2_generated = df_provider[df_provider['origin'] == 'G']
                df2_received=df2_received.reset_index(drop=True)
                if not df2_received.empty:
                    for i in range(len(df2_received)):
                        origin_record = correct_data[correct_data['serviceid']==df2_received.iloc[i]['serviceid']]  
                        if((df2_received.iloc[i]['speed']==origin_record.iloc[0]['speed']) and (df2_received.iloc[i]['latency']==origin_record.iloc[0]['latency'])and (df2_received.iloc[i]['bandwidth']==origin_record.iloc[0]['bandwidth'])and (df2_received.iloc[i]['coverage']==origin_record.iloc[0]['coverage'])and (df2_received.iloc[i]['reliability']==origin_record.iloc[0]['reliability'])and (df2_received.iloc[i]['security']==origin_record.iloc[0]['security'])):
                            df2_received.loc[i, 'label'] = 'C'
                        else:
                            df2_received.loc[i, 'label'] = 'T'
                df2_generated=df2_generated.reset_index(drop=True)
                if not df2_generated.empty:
                    for j in range(len(df2_generated)):
                        provider_other_df1 = correct_data[correct_data['providerid']==df2_generated.iloc[j]['providerid']]
                        provider_other_df = provider_other_df1[provider_other_df1['microcell']!=key2]
                        provider_other_df=provider_other_df.reset_index(drop=True)

                        if(provider_other_df.shape[0]>1):
                            df_to_clust = provider_other_df[[ 'speed','latency','bandwidth','coverage','reliability','security']]
                            dbscan = DBSCAN(eps=0.5, min_samples=5)
                            clusters = dbscan.fit_predict(df_to_clust)
                            outliers = provider_other_df.iloc[clusters == -1]
                            check_list = outliers['serviceid'].unique()
                            if df2_generated.iloc[j]['serviceid'] in check_list:
                                df2_generated.loc[j, 'label'] = 'S'
                            else:
                                df2_generated.loc[j, 'label'] = 'NS'
                        else:
                            df2_generated.loc[j, 'label'] = 'NS'

                t_count = (df2_received['label'] == 'T').sum()
                c_count = (df2_received['label'] == 'C').sum()
                if c_count + t_count != 0:
                    if((t_count/(c_count+t_count))<0.8):
                        df2_generated['label'].replace('S', 'C', inplace=True)
                        df2_generated['label'].replace('NS', 'C', inplace=True)
                    elif((t_count/(c_count+t_count))>0.8):
                        df2_generated['label'].replace('S', 'T', inplace=True)
                        df2_generated['label'].replace('NS', 'T', inplace=True)
                    
                    else:
                        df2_generated['label'].replace('S', 'T', inplace=True)
                        df2_generated['label'].replace('NS', 'C', inplace=True)
                else:
            
                    df2_generated['label'].replace('S', 'T', inplace=True)
                    df2_generated['label'].replace('NS', 'C', inplace=True)


                concatenated_df_provider = pd.concat([df2_generated, df2_received])
                temp_provider_result_dfs[key_provider] = concatenated_df_provider
            
            combined_microcell_df = general.dictionary_to_merged_df(temp_provider_result_dfs)
            results_review[key2] = combined_microcell_df
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_val.append([df2.shape[0],elapsed_time])
            # print(df2.shape[0],elapsed_time)
        return results_review, time_val
    




    def accuracy(self,y_true, y_pred):
        correct_predictions = 0
        total_predictions = len(y_true)
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct_predictions += 1
        return correct_predictions / total_predictions

    def precision(self,y_true, y_pred, positive_class):
        true_positives = sum((true == positive_class) and (pred == positive_class) for true, pred in zip(y_true, y_pred))
        predicted_positives = sum(pred == positive_class for pred in y_pred)
        return true_positives / predicted_positives if predicted_positives != 0 else 0

    def recall(self,y_true, y_pred, positive_class):
        true_positives = sum((true == positive_class) and (pred == positive_class) for true, pred in zip(y_true, y_pred))
        actual_positives = sum(true == positive_class for true in y_true)
        return true_positives / actual_positives if actual_positives != 0 else 0
    

