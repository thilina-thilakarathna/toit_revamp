import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class Replication:
    def __init__(self):
        pass

    def replicate_totally(self,data_list,dfin):
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
    
    def visualize_samples_per_microcell(self, data_list):
        g_counts = []
        r_counts = []
        microcells = list(data_list.keys())

        for key, value in data_list.items():
            df1 = value
            df1 = df1.reset_index(drop=True)

            g_count = df1[df1['origin'] == 'G'].shape[0]
            r_count = df1[df1['origin'] == 'R'].shape[0]

            g_counts.append(g_count)
            r_counts.append(r_count)

        # Calculate total counts for each microcell
        total_counts = [g_count + r_count for g_count, r_count in zip(g_counts, r_counts)]

        # Sort microcells based on total counts
        sorted_microcells = [x for _, x in sorted(zip(total_counts, microcells), reverse=True)]

        # Sort corresponding counts
        sorted_g_counts = [g_counts[microcells.index(m)] for m in sorted_microcells]
        sorted_r_counts = [r_counts[microcells.index(m)] for m in sorted_microcells]

        plt.figure(figsize=(10, 6))
        plt.bar(sorted_microcells, sorted_g_counts, color='blue', label='Generated (G)')
        plt.bar(sorted_microcells, sorted_r_counts, color='red', bottom=sorted_g_counts, label='Replicated (R)')
        plt.xlabel('Microcell')
        plt.ylabel('Trust Infromation Record (TIR) Count')
        # plt.title('Number of Generated and Received Trust Infromation Records per Microcell')
        plt.legend()
        plt.xticks(rotation=90)
            
        # plt.yticks(rotation=90)
        plt.tight_layout()
        plt.grid(True)


        plt.show()

