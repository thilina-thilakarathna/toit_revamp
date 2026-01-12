#Run the experiemt for trust assessment. we will get a data set and go thjrough one by one fopr this.

# 1 creating the data set using airbnb data

import pandas as pd

from evaluations.evaluation_data.evaluation_data import EvaluationData
from tampering.tampering import Tampering


class Evaluations:
    def __init__(self):
        self.data=None
        try:
            self.data = pd.read_csv('evaluations/evaluation_data/evaluation_data.csv')
            print("Loaded existing evaluation data from CSV.")
        except:
            print("No existing evaluation data found. It will be created.")
            self.data=None
        self.evaluation_data = EvaluationData()
        self.tampering = Tampering()

        pass

    def setup_experments(self):
        if self.data is None: 
            self.data = self.evaluation_data.get_data()
        print("Experment setup is complete.")
        
    def experiment_1(self):
        for tampering_type in ["N","K","S"]:
            for tampering_percentage in range(10,100,10):
                print("Running experiment 1 with tampering type:", tampering_type, "and tampering percentage:", tampering_percentage)
                tampered_data = self.tampering.tamper_data(self._dataframe_devide_to_microcell_dictionary(self.data), tampering_percentage, tampering_type)
                # print(tampered_data)
                print(self._dictionary_to_merged_df(tampered_data)['true_label'].value_counts())




                
    














    def _dictionary_to_merged_df(self,dic):
        temp = pd.concat(dic.values(), ignore_index=True)
        temp.reset_index(drop=True, inplace=True)
        return temp

    def _dataframe_devide_to_microcell_dictionary(self,df):
        temp_dictionary={}
        unique_keys = df.microcell.unique()
        for microcell in unique_keys:
            temp_dictionary["{}".format(microcell)] = df[df.microcell==microcell]
        return temp_dictionary
        # seen_providers = set()
        # #select a randon microcell, random provider
        # for microcell in self.data['microcell'].unique():
        #     df_microcell = self.data[self.data['microcell'] == microcell]
        #     print("Microcell:", microcell)

        #     for provider in df_microcell['providerid'].unique():
        #         if provider in seen_providers:
        #             continue

        #         seen_providers.add(provider)
        #         print(" Provider:", provider)



        # Run experiment 1
       



#
# dfin = general.open_file_csv('data_alg_16000.csv')
# dfin = general.slice_df(dfin,['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell'])