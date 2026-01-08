
from matplotlib import pyplot as plt
import pandas as pd


class GeneralOp:
    def __init__(self):
        pass
    
    def test(self):
        print("test")

    def print_statistics(self,df):
        print("Number of samples : "+str(len(df['serviceid'].unique())))
        print("Number of uniques providers : "+str(len(df['providerid'].unique())))
        print("Number of microcells:"+str(len(df['microcell'].unique())))

    def open_file(self,file_name):
        return pd.read_excel(file_name)
    
    def open_file_csv(self,file_name):
        return pd.read_csv(file_name)
    
    def save_file(self,df,file_name):
        df.to_excel(file_name)
    
    def save_file_csv(self,df,file_name):
        df.to_csv(file_name)

    def slice_df(self,df,fields):
        return df[fields]

    def dataframe_devide_to_microcell_dictionary(self,df):
        temp_dictionary={}
        unique_keys = df.microcell.unique()
        for microcell in unique_keys:
            temp_dictionary["{}".format(microcell)] = df[df.microcell==microcell]
        return temp_dictionary

    def trust_score_calculation(self,datain,weight_matrix):
        """Calculate trust scores for each microcell dictionary entry.

        Added `show_progress` parameter to optionally display a tqdm
        progress bar when iterating over microcells.
        """
        scores = {}
        # backward-compatible signature: allow calling with or without show_progress
        show_progress = False
        # if caller passed a third argument by mistake, ignore; otherwise detect named arg
        # NOTE: callers should pass show_progress=True if they want progress bars
        # Build iterator
        iterator = datain.keys()
        if isinstance(datain, dict):
            iterator = list(datain.keys())

        try:
            # allow callers to pass attribute on this object to enable progress globally
            if hasattr(self, 'show_progress') and self.show_progress:
                raise ImportError  # trigger tqdm import below
        except Exception:
            pass

        # If caller requested a progress bar through a convention (set attribute), try to use tqdm
        if getattr(self, 'show_progress', False):
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc='Scoring microcells')
            except Exception:
                iterator = iterator

        for k in iterator:
            wifi_df = datain[k]
            wifi_df = wifi_df.reset_index(drop=True)
            list1 = []
            for i in range(0, wifi_df.shape[0]):
                list1.append(((wifi_df['speed'][i] * weight_matrix[0]) + (wifi_df['latency'][i] * weight_matrix[1]) + (wifi_df['bandwidth'][i] * weight_matrix[2]) + (wifi_df['coverage'][i] * weight_matrix[3]) + (wifi_df['reliability'][i] * weight_matrix[4]) + (wifi_df['security'][i] * weight_matrix[5])) / 10)
            wifi_df['TS'] = list1
            scores[k] = wifi_df
        return scores
    
    def add_a_column_with_a_value(self,df,name,val):
        df[name] = val
        return df
    
    def dictionary_to_merged_df(self,dic):
        temp = pd.concat(dic.values(), ignore_index=True)
        temp.reset_index(drop=True, inplace=True)
        return temp
    
    def visualize_data_one_value(self,df_dic):
        vis_list = []
        for k in df_dic.keys():
            vis_list.append([k,df_dic[k]['TS'].mean()])
        cities = [item[0] for item in vis_list]
        values = [item[1] for item in vis_list]
        plt.figure(figsize=(10, 6))
        plt.scatter(cities, values)
        plt.title('Trust scores for Microcells')
        plt.xlabel('Microcells')
        plt.ylabel('Trust scores')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def visualize_data_two_values(self,df_dic,df_tam_dic):
        vis_list_add = []
        for k in df_dic.keys():
            vis_list_add.append([k,df_dic[k]['TS'].mean(),df_tam_dic[k]['TS'].mean()])

        # Extracting city names and values from the data list
        cities = [item[0] for item in vis_list_add]
        values1 = [item[1] for item in vis_list_add]
        values2 = [item[2] for item in vis_list_add]
        print("plotting")
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(cities, values1,marker='o',color='blue')
        plt.scatter(cities, values2,marker='x',color='red')
        plt.title('Trust scores for Microcells')
        plt.xlabel('Microcells')
        plt.ylabel('Trust scores')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_data_four_values(self,df_dic,df_tam_1,df_tam_2,df_tam_3):
        vis_list_add = []
        for k in df_dic.keys():
            vis_list_add.append([k, df_dic[k]['TS'].mean() , df_tam_1[k]['TS'].mean(),df_tam_2[k]['TS'].mean(),df_tam_3[k]['TS'].mean()])

        # Extracting city names and values from the data list
        cities = [item[0] for item in vis_list_add]
        values1 = [item[1] for item in vis_list_add]
        values2 = [item[2] for item in vis_list_add]
        values3 = [item[3] for item in vis_list_add]
        values4 = [item[4] for item in vis_list_add]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(cities, values1,marker='o',color='blue',label='Original')
        plt.scatter(cities, values2,marker='x',color='red',label='Naive')
        plt.scatter(cities, values3,marker='^',color='orange',label='Knowledgable')
        plt.scatter(cities, values4,marker='s',color='black',label='Sophisticated')
        plt.title('Trust scores for Microcells')
        plt.xlabel('Microcells')
        plt.ylabel('Trust scores')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_data_four_tampredrecords(self,df_dic,df_tam_1,df_tam_2,df_tam_3):
        vis_list_add = []
        for k in df_dic.keys():
            vis_list_add.append([k, df_dic[k]['TS'].mean() , sum(df['true_label'].eq('T').sum() for df in df_tam_1.values()),sum(df['true_label'].eq('T').sum() for df in df_tam_3.values()),sum(df['true_label'].eq('T').sum() for df in df_tam_3.values())])
        # Extracting city names and values from the data list
        cities = [item[0] for item in vis_list_add]
        values1 = [item[1] for item in vis_list_add]
        values2 = [item[2] for item in vis_list_add]
        values3 = [item[3] for item in vis_list_add]
        values4 = [item[4] for item in vis_list_add]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(cities, values1,marker='o',color='blue',label='Original')
        plt.scatter(cities, values2,marker='x',color='red',label='Naive')
        plt.scatter(cities, values3,marker='^',color='orange',label='Knowledgable')
        plt.scatter(cities, values4,marker='s',color='black',label='Sophisticated')
        plt.title('Trust scores for Microcells')
        plt.xlabel('Microcells')
        plt.ylabel('Trust scores')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_data_three_values(self,df_dic,df_tam_1,df_tam_2):
        vis_list_add = []
        for k in df_dic.keys():
            vis_list_add.append([k, df_dic[k]['TS'].mean() , df_tam_1[k]['TS'].mean(),df_tam_2[k]['TS'].mean()])

        # Extracting city names and values from the data list
        cities = [item[0] for item in vis_list_add]
        values1 = [item[1] for item in vis_list_add]
        values2 = [item[2] for item in vis_list_add]
        values3 = [item[3] for item in vis_list_add]


        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(cities, values1,marker='o',color='blue',label='Values 1')
        plt.scatter(cities, values2,marker='x',color='red',label='Values 2')
        plt.scatter(cities, values3,marker='^',color='orange',label='Values 3')

        plt.title('Trust scores for Microcells')
        plt.xlabel('Microcells')
        plt.ylabel('Trust scores')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def calculate_trust(self, data):
        df = pd.DataFrame(data)

        # Define the weights for each metric, assuming equal importance for this example
        weights = {
            'speed': 3,
            'latency': 2,
            'bandwidth': 1,
            'coverage': 2,
            'reliability': 1,
            'security': 1
        }

        # Normalize weights so they sum to 1
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight

        # Calculate the trust score
        df['trust_score'] = (
            df['speed'] * weights['speed'] +
            df['latency'] * weights['latency'] +
            df['bandwidth'] * weights['bandwidth'] +
            df['coverage'] * weights['coverage'] +
            df['reliability'] * weights['reliability'] +
            df['security'] * weights['security']
        ) / len(weights)
        print(df)
        # df['trust_score'] = (
        #     df['speed']  +
        #     df['latency']  +
        #     df['bandwidth'] +
        #     df['coverage'] +
        #     df['reliability']  +
        #     df['security'] 
        # ) / len(weights)
        # print(df)
        # Show results
        # print(df[['serviceid', 'trust_score']])
        return df['trust_score'].mean()


