

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

class EvaluationData:
    def __init__(self):
        self.data = None

    def get_data(self):
        if self.data is None:
            self._load_data()
        print("Preparing evaluation data environment...")
        dfin = self.data[['serviceid','providerid','microcell','latitude','longitude','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell']].copy()
        dfin['origin']='G'
        dfin['true_label']='C'
        print("Replicating data ...")
        merged_df = self._replicate_partially(dfin)
        self.plot_record_counts_per_microcell(merged_df)
        print("Data environment is ready.")
        merged_df.to_csv('evaluations/evaluation_data/evaluation_data.csv',index=False)
        return merged_df
    
    def plot_record_counts_per_microcell(self, df):
        """
        Plot bar chart showing generated and replicated record counts per microcell.
        
        Args:
            df: DataFrame with 'microcell' and 'origin' columns
        """
        # Count records by microcell and origin
        counts = df.groupby(['microcell', 'origin']).size().unstack(fill_value=0)
        
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

    
    def _replicate_totally(self, dfin):
        """
        Replicate remote data records for each microcell.
        
        For each microcell, find all providers in that microcell,
        then replicate all records for those providers from other microcells.
        
        Args:
            dfin: DataFrame with all data (already with 'origin' and 'true_label' columns)
            
        Returns:
            DataFrame with original + replicated records, all in one dataframe
        """
        # Ensure timestamp is datetime
        dfin = dfin.copy()
        if not pd.api.types.is_datetime64_any_dtype(dfin['timestamp']):
            dfin['timestamp'] = pd.to_datetime(dfin['timestamp'])
        
        all_results = []
        
        # Process each microcell
        for microcell in dfin['microcell'].unique():
            # Get local data for this microcell
            local_df = dfin[dfin['microcell'] == microcell].copy().reset_index(drop=True)
            
            # Get all providers in this microcell
            providers_in_microcell = local_df['providerid'].unique()
            
            # Find all records for these providers from other microcells
            remote_mask = (
                (dfin['providerid'].isin(providers_in_microcell)) &
                (dfin['microcell'] != microcell)
            )
            
            df_remote = dfin.loc[remote_mask].copy()
            
            # Add origin and currect_microcell for remote records
            if not df_remote.empty:
                df_remote = df_remote.drop_duplicates(subset='serviceid')
                df_remote['origin'] = 'R'
                df_remote['currect_microcell'] = microcell
                
                # Combine local + replicated for this microcell
                microcell_result = pd.concat([local_df, df_remote], ignore_index=True)
            else:
                microcell_result = local_df
            
            all_results.append(microcell_result)
        
        # Merge all microcells into single dataframe
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    
    def _replicate_partially(self, dfin, k_nearest=5):
        """
        Replicate remote data records from K nearest microcells that have relevant provider data.
        
        For each microcell:
        1. Find all providers in that microcell
        2. Find other microcells that have data for those same providers
        3. From those candidate microcells, select the K nearest by geographic distance
        4. Replicate records from only those K nearest microcells
        
        Args:
            dfin: DataFrame with all data (already with 'origin' and 'true_label' columns)
            k_nearest: Number of nearest microcells with relevant data to consider for replication (default: 2)
            
        Returns:
            DataFrame with original + replicated records from nearby microcells with relevant provider data
        """
        # Ensure timestamp is datetime
        dfin = dfin.copy()
        if not pd.api.types.is_datetime64_any_dtype(dfin['timestamp']):
            dfin['timestamp'] = pd.to_datetime(dfin['timestamp'])
        
        # Get unique microcells with their coordinates
        microcell_coords = dfin.groupby('microcell')[['latitude', 'longitude']].first().reset_index()
        
        all_results = []
        
        # Process each microcell
        for microcell in dfin['microcell'].unique():
            # Get local data for this microcell
            local_df = dfin[dfin['microcell'] == microcell].copy().reset_index(drop=True)
            
            # Get all providers in this microcell
            providers_in_microcell = local_df['providerid'].unique()
            
            # Get coordinates of current microcell
            current_coords = microcell_coords[microcell_coords['microcell'] == microcell]
            if current_coords.empty:
                all_results.append(local_df)
                continue
            
            lat1 = current_coords['latitude'].values[0]
            lon1 = current_coords['longitude'].values[0]
            
            # Find microcells that have data for the same providers
            candidate_microcells = []
            for _, row in microcell_coords.iterrows():
                if row['microcell'] == microcell:
                    continue
                
                # Check if this microcell has data for any of the providers
                has_relevant_data = (
                    (dfin['microcell'] == row['microcell']) &
                    (dfin['providerid'].isin(providers_in_microcell))
                ).any()
                
                if has_relevant_data:
                    dist = self._haversine_distance(lat1, lon1, row['latitude'], row['longitude'])
                    candidate_microcells.append((row['microcell'], dist))
            
            # Select K nearest from candidates
            if candidate_microcells:
                candidate_microcells.sort(key=lambda x: x[1])
                nearby_microcells = [m for m, _ in candidate_microcells[:k_nearest]]
                print(f"  Microcell: {microcell} -> Selected for replication: {nearby_microcells}")
            else:
                nearby_microcells = []
                print(f"  Microcell: {microcell} -> No relevant microcells found")
            
            # Find records for these providers from nearby microcells only
            if nearby_microcells:
                remote_mask = (
                    (dfin['providerid'].isin(providers_in_microcell)) &
                    (dfin['microcell'].isin(nearby_microcells))
                )
                
                df_remote = dfin.loc[remote_mask].copy()
            else:
                df_remote = pd.DataFrame()
            
            # Add origin and currect_microcell for remote records
            if not df_remote.empty:
                df_remote = df_remote.drop_duplicates(subset='serviceid')
                df_remote['origin'] = 'R'
                df_remote['currect_microcell'] = microcell
                
                # Combine local + replicated for this microcell
                microcell_result = pd.concat([local_df, df_remote], ignore_index=True)
            else:
                microcell_result = local_df
            
            all_results.append(microcell_result)
        
        # Merge all microcells into single dataframe
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees).
        Returns distance in kilometers.
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
    
    def _find_k_nearest_microcells(self, microcell_coords, lat1, lon1, current_microcell, k_nearest=10):
        """
        Find K nearest microcells based on geographic distance.
        
        Args:
            microcell_coords: DataFrame with microcell coordinates
            lat1, lon1: Current microcell coordinates
            current_microcell: Current microcell name (to exclude from results)
            k_nearest: Number of nearest neighbors to return
            
        Returns:
            List of K nearest microcell names sorted by distance
        """
        distances = []
        
        for _, row in microcell_coords.iterrows():
            if row['microcell'] == current_microcell:
                continue
            
            dist = self._haversine_distance(lat1, lon1, row['latitude'], row['longitude'])
            distances.append((row['microcell'], dist))
        
        if not distances:
            return []
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return K nearest neighbors
        nearby = [m for m, _ in distances[:k_nearest]]
        
        return nearby

    def _print_statistics(self,df):
        print("Number of samples : "+str(len(df['serviceid'].unique())))
        print("Number of uniques providers : "+str(len(df['providerid'].unique())))
        print("Number of microcells:"+str(len(df['microcell'].unique())))

    
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