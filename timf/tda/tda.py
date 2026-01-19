import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from timf.trust_assessment.trust_assessment import TrustAssessment
import time


class Detection:
    def __init__(self, data_service):
        self.data_service = data_service
        self.trust_assessor = TrustAssessment([0.3, 0.1, 0.2, 0.1, 0.1, 0.2])

        self.attrs = [
            'speed',
            'latency',
            'bandwidth',
            'coverage',
            'reliability',
            'security'
        ]

    def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
        start_time = time.time()

        df = local_data.reset_index(drop=True).copy()
        df['label'] = 'U'   # U: unknown, C: clean, T: tampered

        # ==========================================================
        # Phase 1: Source consistency (ONLY origin == 'R')
        # ==========================================================
        if remote_data is not None and not remote_data.empty:

            remote_subset = df[df['origin'] == 'R']

            for i in remote_subset.index:
                sid = df.loc[i, 'serviceid']
                origin_row = remote_data[remote_data['serviceid'] == sid]

                if not origin_row.empty:
                    diff = np.mean([
                        abs(df.loc[i, a] - origin_row.iloc[0][a])
                        for a in self.attrs
                    ])

                    if diff < 0.0005:
                        df.loc[i, 'label'] = 'P-C'   # provisionally clean
                    else:
                        df.loc[i, 'label'] = 'T'

        # ==========================================================
        # Phase 2: Microcell-level clustering (source-consistent)
        # ==========================================================
        pc_df = df[df['label'] == 'P-C'].copy()
        clean_reference = None

        if len(pc_df) >= 5:
            pc_norm = self.normalize(pc_df, self.attrs)

            dbscan = DBSCAN(eps=0.3, min_samples=4)
            clusters = dbscan.fit_predict(pc_norm[self.attrs])
            pc_df['cluster'] = clusters

            # Largest cluster = honest majority
            largest_cluster = pc_df['cluster'].value_counts().idxmax()
            clean_pc = pc_df[pc_df['cluster'] == largest_cluster]

            # Mark minority clusters as tampered
            df.loc[
                pc_df[pc_df['cluster'] != largest_cluster].index,
                'label'
            ] = 'T'

            df.loc[clean_pc.index, 'label'] = 'C'

            # Reference centroid (optional, not strictly needed now)
            clean_reference = clean_pc[self.attrs].mean()

        # ==========================================================
        # Phase 3: Provider-level consistency (ONLY origin == 'G')
        # ==========================================================
        generated_df = df[
            (df['label'] == 'U') &
            (df['origin'] == 'G')
        ].copy()

        if not generated_df.empty:
            for provider in generated_df['providerid'].unique():

                gen_subset = generated_df[
                    generated_df['providerid'] == provider
                ]

                # Provider historical clean data from OTHER microcells
                provider_hist = df[
                    (df['providerid'] == provider) &
                    (df['gen_microcell'] != microcell_id) &
                    (df['label'] == 'C')
                ]

                # Not enough provider history → conservative clean
                if len(provider_hist) < 5:
                    df.loc[gen_subset.index, 'label'] = 'C'
                    continue

                hist_norm = self.normalize(provider_hist, self.attrs)

                dbscan = DBSCAN(eps=0.4, min_samples=5)
                clusters = dbscan.fit_predict(hist_norm[self.attrs])

                provider_hist = provider_hist.copy()
                provider_hist['cluster'] = clusters

                # Dominant provider behavior
                largest_cluster = provider_hist['cluster'].value_counts().idxmax()
                dominant_hist = provider_hist[
                    provider_hist['cluster'] == largest_cluster
                ]

                centroid = dominant_hist[self.attrs].mean()

                for i in gen_subset.index:
                    dist = self.distance_to_centroid(
                        df.loc[i], centroid, self.attrs
                    )
                    df.loc[i, 'label'] = 'T' if dist > 0.5 else 'C'

        elapsed_time = time.time() - start_time

        return df.copy(), {
            'records': df.shape[0],
            'time': elapsed_time
        }

    # ==========================================================
    # Utility methods
    # ==========================================================
    def distance_to_centroid(self, row, centroid, cols):
        return np.mean([abs(row[c] - centroid[c]) for c in cols])

    def normalize(self, df, cols):
        df = df.copy()
        for c in cols:
            min_v = df[c].min()
            max_v = df[c].max()
            df[c] = (df[c] - min_v) / (max_v - min_v + 1e-9)
        return df




# from locale import normalize
# import numpy as np
# import pandas as pd
# from sklearn.cluster import DBSCAN
# from timf.trust_assessment.trust_assessment import TrustAssessment
# import time

# class Detection:
#     def __init__(self, data_service):
#         self.data_service = data_service
#         self.trust_assessor = TrustAssessment([0.3,0.1,0.2,0.1,0.1,0.2])

#     def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
#         start_time = time.time()

#         df = local_data.reset_index(drop=True).copy()
#         df['label'] = 'U'

#         attrs = ['speed','latency','bandwidth','coverage','reliability','security']

#         # ---------- Phase 1: Source consistency ----------
#         if remote_data is not None and not remote_data.empty:
#             for i in range(len(df)):
#                 sid = df.loc[i, 'serviceid']
#                 origin = remote_data[remote_data['serviceid'] == sid]

#                 if not origin.empty:
#                     diff = np.mean([abs(df.loc[i, a] - origin.iloc[0][a]) for a in attrs])

#                     if diff < 0.0005:
#                         df.loc[i, 'label'] = 'P-C'
#                     else:
#                         df.loc[i, 'label'] = 'T'

#         # ---------- Phase 2: Cluster source-consistent records ----------
#         pc_df = df[df['label'] == 'P-C'].copy()

#         clean_reference = None

#         if len(pc_df) >= 5:
#             pc_norm = self.normalize(pc_df, attrs)

#             dbscan = DBSCAN(eps=0.3, min_samples=4)
#             clusters = dbscan.fit_predict(pc_norm[attrs])

#             pc_df['cluster'] = clusters

#             # Largest cluster = honest majority
#             largest_cluster = pc_df['cluster'].value_counts().idxmax()
#             clean_pc = pc_df[pc_df['cluster'] == largest_cluster]

#             # Mark others as tampered (BMA)
#             df.loc[pc_df[pc_df['cluster'] != largest_cluster].index, 'label'] = 'T'
#             df.loc[clean_pc.index, 'label'] = 'C'

#             # Build pseudo-reference
#             clean_reference = clean_pc[attrs].mean()

#         # ---------- Phase 3: Bootstrap detection for local-only ----------
#         local_only = df[df['label'] == 'U']

#         if clean_reference is not None:
#             for i in local_only.index:
#                 dist = self.distance_to_centroid(df.loc[i], clean_reference, attrs)
#                 df.loc[i, 'label'] = 'C' if dist < 0.5 else 'T'
#         else:
#             # Conservative fallback
#             df.loc[df['label'] == 'U', 'label'] = 'C'

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         # time_val.append([df2.shape[0],elapsed_time])

#         return df.copy(),{df.shape[0],elapsed_time}



#     def distance_to_centroid(self,row, centroid, cols):
#         return np.mean([abs(row[c] - centroid[c]) for c in cols])


    
#     def normalize(self,df, cols):
#         df = df.copy()
#         for c in cols:
#             min_v = df[c].min()
#             max_v = df[c].max()
#             df[c] = (df[c] - min_v) / (max_v - min_v + 1e-9)
#         return df


       
 
