

from datetime import datetime
import pandas as pd


class EvaluationData:
    def __init__(self):
        self.data = None

    def get_data(self):
        if self.data is None:
            self._load_data()
        print("Preparing evaluation data environment...")
        dfin = self.data[['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell']].copy()
        dfin['origin']='G'
        dfin['true_label']='C'
        data_list = self._dataframe_devide_to_microcell_dictionary(dfin)
        print("Replicating data ...")
        data_rep = self._replicate_totally(data_list,dfin)
        merged_df = self._dictionary_to_merged_df(data_rep)
        print("Data environment is ready.")
        merged_df.to_csv('evaluations/evaluation_data/evaluation_data.csv',index=False)
        return merged_df
    
    def _dataframe_devide_to_microcell_dictionary(self,df):
        temp_dictionary={}
        unique_keys = df.microcell.unique()
        for microcell in unique_keys:
            temp_dictionary["{}".format(microcell)] = df[df.microcell==microcell]
        return temp_dictionary
    
    def _dictionary_to_merged_df(self,dic):
        temp = pd.concat(dic.values(), ignore_index=True)
        temp.reset_index(drop=True, inplace=True)
        return temp
    
    # def _replicate_totally(self, data_list, dfin):
    #     data = {}

    #     # Ensure timestamp is datetime (do this once)
    #     if not pd.api.types.is_datetime64_any_dtype(dfin['timestamp']):
    #         dfin = dfin.copy()
    #         dfin['timestamp'] = pd.to_datetime(dfin['timestamp'])

    #     # Pre-group by provider for fast lookup
    #     provider_groups = dfin.groupby('providerid')

    #     for key, df1 in data_list.items():
    #         df1 = df1.copy().reset_index(drop=True)

    #         replicated_records = []

    #         for _, row in df1.iterrows():
    #             provider_id = row['providerid']
    #             current_time = row['timestamp']

    #             if provider_id not in provider_groups.groups:
    #                 continue

    #             provider_df = provider_groups.get_group(provider_id)

    #             # Vectorized filtering
    #             mask = (
    #                 (provider_df['microcell'] != key) &
    #                 (provider_df['timestamp'] < current_time)
    #             )

    #             replicated_records.append(provider_df.loc[mask])

    #         if replicated_records:
    #             df_remote = pd.concat(replicated_records, ignore_index=True)
    #             df_remote = df_remote.drop_duplicates(subset='serviceid')

    #             df_remote.loc[:, 'origin'] = 'R'
    #             df_remote.loc[:, 'currect_microcell'] = key
    #         else:
    #             df_remote = pd.DataFrame(columns=df1.columns)

    #         # Combine local + replicated
    #         result = pd.concat([df1, df_remote], ignore_index=True)
    #         data[key] = result

    #     return data

    
    def _replicate_totally(self,data_list,dfin):
        data={}
        column_names = ['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell']

        for key, value in data_list.items():
            df1 = value
            df1=df1.reset_index(drop=True)

            df_temporaty = pd.DataFrame(columns=column_names)
            list_test =[]
            for i in range (0,df1.shape[0]):
                dflag = dfin[dfin.providerid==df1['providerid'][i]]
                dflag = dflag.reset_index(drop=True)
                for j in range (0,dflag.shape[0]):
                    if(str(dflag['microcell'][j])!=key):
                        if(datetime.strptime(dflag['timestamp'][j], '%Y-%m-%d')<datetime.strptime(df1['timestamp'][i], '%Y-%m-%d')):
                            list_test.append(dflag.iloc[j])
            
            df_temporaty = pd.DataFrame(list_test)
            df_temporaty['origin'] = 'R'
            df_temporaty['currect_microcell'] = key
            df_temporaty=df_temporaty.reset_index(drop=True)
            df_no_duplicates = df_temporaty.drop_duplicates(subset='serviceid')
            result = pd.concat([df1, df_no_duplicates], axis=0)
            result = result.reset_index(drop=True)
            data[key] = result
        return data
    

    def _print_statistics(self,df):
        print("Number of samples : "+str(len(df['serviceid'].unique())))
        print("Number of uniques providers : "+str(len(df['providerid'].unique())))
        print("Number of microcells:"+str(len(df['microcell'].unique())))

    
    def _load_data(self):
        print("Loading Airbnb data for evaluation...")
        custompath = 'data_source/Sydney'
        # custompath2 = 'New/Melbourne'
        # custompath3 = 'New/MidNorthCoast'
        # custompath4 = 'New/NothernRivers'

        data_frame_in1 = pd.read_csv(custompath+'/listings1.csv')
        data_frame_in2 = pd.read_csv(custompath+'/listings2.csv')
        data_frame_in3 = pd.read_csv(custompath+'/listings3.csv')
        data_frame_in4 = pd.read_csv(custompath+'/listings4.csv')

        data_frame_1 = pd.concat([data_frame_in1, data_frame_in2, data_frame_in3, data_frame_in4], ignore_index=True)
        data_frame_in = data_frame_1

        dfin = data_frame_in[['id','host_id','last_review','neighbourhood_cleansed','latitude','longitude','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']]

        column_mapping = {
            'id': 'serviceid',
            'host_id': 'providerid',
            'last_review': 'timestamp',
            # 'host_acceptance_rate': 'responsiveness',
            # 'host_name': 'Host Name',
            'neighbourhood_cleansed': 'microcell',
            # 'host_acceptance_rate': 'Host Response Rate',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'review_scores_rating': 'speed',
            'review_scores_accuracy': 'throughput',
            'review_scores_cleanliness': 'bandwidth',
            'review_scores_checkin': 'coverage',
            'review_scores_communication': 'reliability',
            'review_scores_location': 'security',
            'review_scores_value': 'latency'
        }


        dfin= dfin.rename(columns=column_mapping)
        dfin=dfin.dropna()

        dfin['serviceid'] = range(100000, 100000 + len(dfin))
        unique_microcells = dfin['microcell'].unique()
        microcell_mapping = {name: f"M{102 + i}" for i, name in enumerate(unique_microcells)}
        dfin['microcell'] = dfin['microcell'].map(microcell_mapping)
        unique_providers = dfin['providerid'].unique()
        provider_mapping = {name: f"P{1000000 + j}" for j, name in enumerate(unique_providers)}
        dfin['providerid'] = dfin['providerid'].map(provider_mapping)

        # self._print_statistics(dfin)

        samples_per_provider = dfin.groupby('providerid').size().reset_index(name='sample_count')
        microcells_per_provider = dfin.groupby('providerid')['microcell'].nunique().reset_index(name='microcell_count')

        counts_per_provider = pd.merge(samples_per_provider, microcells_per_provider, on='providerid')

        
        filtered_counts_per_provider = counts_per_provider[counts_per_provider['microcell_count'] >= 10]
      

        selected_provider_ids = filtered_counts_per_provider['providerid']

        filtered_df = dfin[dfin['providerid'].isin(selected_provider_ids)]

        # self._print_statistics(filtered_df)

        result = filtered_df.groupby('microcell').first()[['latitude', 'longitude']]

        # Merge the selected latitude and longitude pairs back into the original dataframe
        merged_df = filtered_df.merge(result, on='microcell', suffixes=('', '_selected'))

        # Replace the latitude and longitude with the selected values
        merged_df['latitude'] = merged_df['latitude_selected']
        merged_df['longitude'] = merged_df['longitude_selected']

        # Drop the extra columns used for merging
        final_df = merged_df.drop(columns=['latitude_selected', 'longitude_selected'])

        self._print_statistics(final_df)
        final_df['currect_microcell']=final_df['microcell']

      
        self.data = final_df
        print("Data loading completed.")