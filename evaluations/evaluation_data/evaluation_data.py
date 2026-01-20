

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

class EvaluationData:
    def __init__(self):
        self.data = None

    def get_data(self):
        """
        Get evaluation data. If CSV exists, load and return it. Otherwise, create new data.
        
        Returns:
            DataFrame with evaluation data including origin and true_label columns
        """
        # Try to load existing CSV
        try:
            data = pd.read_csv('evaluations/evaluation_data/evaluation_data.csv')
            print("Loaded existing evaluation data from CSV.")
            return data
        except FileNotFoundError:
            print("No existing evaluation data found. It will be created.")
        except Exception as e:
            print(f"Error loading evaluation data: {e}")
        
        # CSV not available - run full data preparation
        print("Preparing evaluation data environment...")
        
        # Load raw data
        if self.data is None:
            self._load_data()
        
        # Prepare data with origin and true_label
        dfin = self.data[['serviceid','providerid','gen_microcell','latitude','longitude','timestamp',
                          'speed','latency','bandwidth','coverage','reliability','security']].copy()
        dfin['origin'] = 'G'
        dfin['true_label'] = 'C'
        dfin['true_label_spa'] = 'C'
        dfin['true_label_bma'] = 'C'
        
        # print("Replicating data ...")
        # merged_df = self._replicate_partially(dfin)
        # self.plot_record_counts_per_microcell(merged_df)
        # print("Data environment is ready.")
        
        # Save to CSV for future use
        dfin.to_csv('evaluations/evaluation_data/evaluation_data.csv', index=False)
        return dfin
    
    def plot_record_counts_per_microcell(self, df):
        """
        Plot bar chart showing generated and replicated record counts per microcell.
        
        Args:
            df: DataFrame with 'microcell' and 'origin' columns
        """
        # Count records by microcell and origin
        counts = df.groupby(['gen_microcell', 'origin']).size().unstack(fill_value=0)
        
        # Ensure both 'G' and 'R' columns exist
        if 'G' not in counts.columns:
            counts['G'] = 0
        if 'R' not in counts.columns:
            counts['R'] = 0
        
        # Reorder columns for consistency
        counts = counts[['G', 'R']]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        counts.plot(kind='bar', stacked=False, ax=ax, 
                   color=['#2ecc71', '#e74c3c'],
                   label=['Generated', 'Replicated'])
        
        ax.set_xlabel('Microcell', fontsize=12)
        ax.set_ylabel('Record Count', fontsize=12)
        ax.set_title('Record Counts per Microcell (Generated vs Replicated)', fontsize=14, fontweight='bold')
        ax.legend(['Generated', 'Replicated'], fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
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

    
   

    def _print_statistics(self,df):
        print("Number of samples : "+str(len(df['serviceid'].unique())))
        print("Number of uniques providers : "+str(len(df['providerid'].unique())))
        print("Number of microcells:"+str(len(df['gen_microcell'].unique())))

    
    def _load_data(self):
        print("Loading Airbnb data for evaluation...")
        
        # Load data from multiple locations
        data_frames = []
        locations = [
            'data_source/Sydney',
            # 'data_source/Melbourne',
            # 'data_source/MidNorthCoast',
            # 'data_source/NothernRivers'
        ]
        
        for location in locations:
            print(f"  Loading from {location}...")
            try:
                for i in range(1, 5):
                    try:
                        df = pd.read_csv(f'{location}/listings{i}.csv')
                        data_frames.append(df)
                    except FileNotFoundError:
                        continue
            except Exception as e:
                print(f"  Warning: Could not load from {location}: {e}")
        
        data_frame_1 = pd.concat(data_frames, ignore_index=True)
        data_frame_in = data_frame_1

        dfin = data_frame_in[['id','host_id','last_review','neighbourhood_cleansed','latitude','longitude','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']]

        column_mapping = {
            'id': 'serviceid',
            'host_id': 'providerid',
            'last_review': 'timestamp',
            # 'host_acceptance_rate': 'responsiveness',
            # 'host_name': 'Host Name',
            'neighbourhood_cleansed': 'gen_microcell',
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
        unique_microcells = dfin['gen_microcell'].unique()
        microcell_mapping = {name: f"M{102 + i}" for i, name in enumerate(unique_microcells)}
        dfin['gen_microcell'] = dfin['gen_microcell'].map(microcell_mapping)
        unique_providers = dfin['providerid'].unique()
        provider_mapping = {name: f"P{1000000 + j}" for j, name in enumerate(unique_providers)}
        dfin['providerid'] = dfin['providerid'].map(provider_mapping)

        # self._print_statistics(dfin)

        samples_per_provider = dfin.groupby('providerid').size().reset_index(name='sample_count')
        microcells_per_provider = dfin.groupby('providerid')['gen_microcell'].nunique().reset_index(name='microcell_count')

        counts_per_provider = pd.merge(samples_per_provider, microcells_per_provider, on='providerid')

        
        filtered_counts_per_provider = counts_per_provider[counts_per_provider['microcell_count'] >= 2]
      

        selected_provider_ids = filtered_counts_per_provider['providerid']

        filtered_df = dfin[dfin['providerid'].isin(selected_provider_ids)]

        # self._print_statistics(filtered_df)

        result = filtered_df.groupby('gen_microcell').first()[['latitude', 'longitude']]

        # Merge the selected latitude and longitude pairs back into the original dataframe
        merged_df = filtered_df.merge(result, on='gen_microcell', suffixes=('', '_selected'))

        # Replace the latitude and longitude with the selected values
        merged_df['latitude'] = merged_df['latitude_selected']
        merged_df['longitude'] = merged_df['longitude_selected']

        # Drop the extra columns used for merging
        final_df = merged_df.drop(columns=['latitude_selected', 'longitude_selected'])

        self._print_statistics(final_df)
        # final_df['currect_microcell']=final_df['microcell']

      
        self.data = final_df
        print("Data loading completed.")