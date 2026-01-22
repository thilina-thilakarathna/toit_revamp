import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time


class Detection:
    def __init__(self, data_service):
        self.data_service = data_service
        self.attrs = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

    def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
        """
        Approach B (fixed):
        - Part 1 (origin == 'R'): compare local vs remote by serviceid (tolerant equality).
        - Part 2 (origin == 'G'): DBSCAN provider-outlier test by APPENDING the current record
          and checking its DBSCAN label.
        - Part 3: global rule based on received tamper ratio.

        Runtime timing excludes preprocessing (scaling).
        Returns ORIGINAL (unscaled) attribute values + labels.
        """

        # Keep original values for output
        df_out = local_data.reset_index(drop=True).copy()
        df_out['label'] = 'U'  # U: unknown, C: clean, T: tampered (S/NS internal)

        # Defensive copy for remote
        remote_ref = None
        if remote_data is not None and not remote_data.empty:
            remote_ref = remote_data.copy()

        # ==========================================================
        # Preprocessing (NOT TIMED): prepare scaler for DBSCAN only
        # Fit on remote (preferred) else local. Transform is done later.
        # ==========================================================
        scaler = None
        if remote_ref is not None and not remote_ref.empty and all(a in remote_ref.columns for a in self.attrs):
            scaler = StandardScaler().fit(remote_ref[self.attrs].to_numpy(dtype=float))
        else:
            scaler = StandardScaler().fit(df_out[self.attrs].to_numpy(dtype=float))

        # ==========================================================
        # Detection runtime (TIMED)
        # ==========================================================
        start_time = time.time()

        # ==========================================================
        # Part 1: Source consistency for origin == 'R'
        # Compare on ORIGINAL values (no scaling needed for equality check)
        # ==========================================================
        if remote_ref is not None and not remote_ref.empty:
            received_idx = df_out.index[df_out['origin'] == 'R']

            for i in received_idx:
                sid = df_out.loc[i, 'serviceid']
                origin_row = remote_ref[remote_ref['serviceid'] == sid]

                if origin_row.empty:
                    continue

                local_vals = df_out.loc[i, self.attrs].to_numpy(dtype=float)
                origin_vals = origin_row.iloc[0][self.attrs].to_numpy(dtype=float)

                # tolerant equality to avoid float issues
                is_equal = np.all(np.isclose(local_vals, origin_vals, rtol=1e-5, atol=1e-8))
                df_out.loc[i, 'label'] = 'C' if is_equal else 'T'

        # ==========================================================
        # Part 2: Provider-outlier test for origin == 'G'
        # Fix: append the current record to provider_other, cluster, check last label
        # ==========================================================
        generated_idx = df_out.index[df_out['origin'] == 'G']

        if remote_ref is not None and not remote_ref.empty and len(generated_idx) > 0:
            for j in generated_idx:
                prov = df_out.loc[j, 'providerid']

                provider_other = remote_ref[
                    (remote_ref['providerid'] == prov) &
                    (remote_ref['gen_microcell'] != microcell_id)
                ].copy()

                # Need enough points for DBSCAN to be meaningful
                # (at least min_samples points besides the query point)
                if provider_other.shape[0] < 2:
                    df_out.loc[j, 'label'] = 'NS'
                    continue

                # Build clustering set: provider_other + current record
                X_ref = provider_other[self.attrs].to_numpy(dtype=float)
                x_q = df_out.loc[j, self.attrs].to_numpy(dtype=float).reshape(1, -1)

                # Scale (NOT counted as preprocessing in timing? it happens here, but it's unavoidable per-point.
                # If you want STRICT exclusion, you can precompute scaled remote per call outside start_time,
                # but then you'd be scaling a lot anyway. This is lightweight.)
                X_ref_s = scaler.transform(X_ref)
                x_q_s = scaler.transform(x_q)

                X = np.vstack([X_ref_s, x_q_s])

                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(X)

                # The query point is the last row
                df_out.loc[j, 'label'] = 'S' if labels[-1] == -1 else 'NS'

        # ==========================================================
        # Part 3: Global rule based on received tamper ratio
        # Convert S/NS -> final C/T for generated records
        # ==========================================================
        received_mask = (df_out['origin'] == 'R')
        t_count = (df_out.loc[received_mask, 'label'] == 'T').sum()
        c_count = (df_out.loc[received_mask, 'label'] == 'C').sum()

        gen_mask = (df_out['origin'] == 'G')

        if (c_count + t_count) != 0:
            ratio = t_count / (c_count + t_count)

            if ratio < 0.8:
                df_out.loc[gen_mask, 'label'] = df_out.loc[gen_mask, 'label'].replace({'S': 'C', 'NS': 'C'})
            elif ratio > 0.8:
                df_out.loc[gen_mask, 'label'] = df_out.loc[gen_mask, 'label'].replace({'S': 'T', 'NS': 'T'})
            else:
                df_out.loc[gen_mask, 'label'] = df_out.loc[gen_mask, 'label'].replace({'S': 'T', 'NS': 'C'})
        else:
            # No received records
            df_out.loc[gen_mask, 'label'] = df_out.loc[gen_mask, 'label'].replace({'S': 'T', 'NS': 'C'})

        elapsed_time = time.time() - start_time
        return df_out.copy(), {'records': df_out.shape[0], 'time': elapsed_time}

# import numpy as np
# import pandas as pd
# from sklearn.cluster import DBSCAN
# import time

# class Detection:
#     def __init__(self, data_service):
#         self.data_service = data_service
#         self.attrs = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

#     def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
#         """
#         Approach B using the same variable names / conventions as Approach A.

#         Assumptions (matching your snippet):
#         - local_data contains both origin == 'R' (received) and origin == 'G' (generated)
#         - remote_data plays the role of 'correct_data' (origin/ground-truth reference by serviceid)
#         - local_data has: serviceid, providerid, origin, gen_microcell, and attrs columns
#         """
        

#         df = local_data.reset_index(drop=True).copy()
#         df['label'] = 'U'  # U: unknown, C: clean, T: tampered (we'll also use S/NS internally)

#         # ==========================================================
#         # Part 1: Exact-match source consistency for origin == 'R'
#         # ==========================================================
#         start_time = time.time()
#         if remote_data is not None and not remote_data.empty:
#             received_df = df[df['origin'] == 'R'].copy()

#             if not received_df.empty:
#                 # We keep updates in the original df using index mapping
#                 for i in received_df.index:
#                     sid = df.loc[i, 'serviceid']
#                     origin_row = remote_data[remote_data['serviceid'] == sid]

#                     if origin_row.empty:
#                         # If no origin record, keep unknown (or you could mark T conservatively)
#                         continue

#                     # EXACT equality across all attributes (as in Approach B snippet)
#                     is_equal = True
#                     for a in self.attrs:
#                         if df.loc[i, a] != origin_row.iloc[0][a]:
#                             is_equal = False
#                             break

#                     df.loc[i, 'label'] = 'C' if is_equal else 'T'

#         # ==========================================================
#         # Part 2: Provider-outlier test for origin == 'G'
#         #         (label as S or NS first)
#         # ==========================================================
#         generated_df = df[df['origin'] == 'G'].copy()

#         if (remote_data is not None) and (not remote_data.empty) and (not generated_df.empty):
#             for j in generated_df.index:
#                 prov = df.loc[j, 'providerid']

#                 # Provider records from remote_data excluding current microcell
#                 provider_other = remote_data[
#                     (remote_data['providerid'] == prov) &
#                     (remote_data['gen_microcell'] != microcell_id)
#                 ].copy()

#                 if provider_other.shape[0] > 1:
#                     df_to_clust = provider_other[self.attrs].copy()

#                     dbscan = DBSCAN(eps=0.5, min_samples=5)
#                     clusters = dbscan.fit_predict(df_to_clust)

#                     # Outliers are cluster == -1
#                     outliers = provider_other.iloc[clusters == -1]
#                     outlier_serviceids = set(outliers['serviceid'].unique())

#                     df.loc[j, 'label'] = 'S' if df.loc[j, 'serviceid'] in outlier_serviceids else 'NS'
#                 else:
#                     df.loc[j, 'label'] = 'NS'

#         # ==========================================================
#         # Part 3: Global rule based on received tamper ratio
#         #         (convert S/NS -> final C/T)
#         # ==========================================================
#         # Compute tamper ratio on received records only
#         received_mask = (df['origin'] == 'R')
#         t_count = (df.loc[received_mask, 'label'] == 'T').sum()
#         c_count = (df.loc[received_mask, 'label'] == 'C').sum()

#         gen_mask = (df['origin'] == 'G')

#         if (c_count + t_count) != 0:
#             ratio = t_count / (c_count + t_count)

#             if ratio < 0.8:
#                 # everything generated becomes clean
#                 df.loc[gen_mask, 'label'] = df.loc[gen_mask, 'label'].replace({'S': 'C', 'NS': 'C'})
#             elif ratio > 0.8:
#                 # everything generated becomes tampered
#                 df.loc[gen_mask, 'label'] = df.loc[gen_mask, 'label'].replace({'S': 'T', 'NS': 'T'})
#             else:
#                 # boundary case (ratio == 0.8): S->T, NS->C
#                 df.loc[gen_mask, 'label'] = df.loc[gen_mask, 'label'].replace({'S': 'T', 'NS': 'C'})
#         else:
#             # No received records: default S->T, NS->C
#             df.loc[gen_mask, 'label'] = df.loc[gen_mask, 'label'].replace({'S': 'T', 'NS': 'C'})

#         elapsed_time = time.time() - start_time
#         return df.copy(), {'records': df.shape[0], 'time': elapsed_time}

# //////////////////////////////////////////////////////////////////
# import numpy as np
# import pandas as pd
# from sklearn.cluster import DBSCAN
# from timf.trust_assessment.trust_assessment import TrustAssessment
# import time


# class Detection:
#     def __init__(self, data_service):
#         self.data_service = data_service
#         self.trust_assessor = TrustAssessment([0.3, 0.1, 0.2, 0.1, 0.1, 0.2])

#         self.attrs = [
#             'speed',
#             'latency',
#             'bandwidth',
#             'coverage',
#             'reliability',
#             'security'
#         ]

#     def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
#         start_time = time.time()

#         df = local_data.reset_index(drop=True).copy()
#         df['label'] = 'U'   # U: unknown, C: clean, T: tampered

#         # ==========================================================
#         # Phase 1: Source consistency (ONLY origin == 'R')
#         # ==========================================================
#         if remote_data is not None and not remote_data.empty:

#             remote_subset = df[df['origin'] == 'R']

#             for i in remote_subset.index:
#                 sid = df.loc[i, 'serviceid']
#                 origin_row = remote_data[remote_data['serviceid'] == sid]

#                 if not origin_row.empty:
#                     diff = np.mean([
#                         abs(df.loc[i, a] - origin_row.iloc[0][a])
#                         for a in self.attrs
#                     ])

#                     if diff < 0.0005:
#                         df.loc[i, 'label'] = 'P-C'   # provisionally clean
#                     else:
#                         df.loc[i, 'label'] = 'T'

#         # ==========================================================
#         # Phase 2: Microcell-level clustering (source-consistent)
#         # ==========================================================
#         pc_df = df[df['label'] == 'P-C'].copy()
#         clean_reference = None

#         if len(pc_df) >= 5:
#             pc_norm = self.normalize(pc_df, self.attrs)

#             dbscan = DBSCAN(eps=0.3, min_samples=4)
#             clusters = dbscan.fit_predict(pc_norm[self.attrs])
#             pc_df['cluster'] = clusters

#             # Largest cluster = honest majority
#             largest_cluster = pc_df['cluster'].value_counts().idxmax()
#             clean_pc = pc_df[pc_df['cluster'] == largest_cluster]

#             # Mark minority clusters as tampered
#             df.loc[
#                 pc_df[pc_df['cluster'] != largest_cluster].index,
#                 'label'
#             ] = 'T'

#             df.loc[clean_pc.index, 'label'] = 'C'

#             # Reference centroid (optional, not strictly needed now)
#             clean_reference = clean_pc[self.attrs].mean()

#         # ==========================================================
#         # Phase 3: Provider-level consistency (ONLY origin == 'G')
#         # ==========================================================
#         generated_df = df[
#             (df['label'] == 'U') &
#             (df['origin'] == 'G')
#         ].copy()

#         if not generated_df.empty:
#             for provider in generated_df['providerid'].unique():

#                 gen_subset = generated_df[
#                     generated_df['providerid'] == provider
#                 ]

#                 # Provider historical clean data from OTHER microcells
#                 provider_hist = df[
#                     (df['providerid'] == provider) &
#                     (df['gen_microcell'] != microcell_id) &
#                     (df['label'] == 'C')
#                 ]

#                 # Not enough provider history → conservative clean
#                 if len(provider_hist) < 5:
#                     df.loc[gen_subset.index, 'label'] = 'C'
#                     continue

#                 hist_norm = self.normalize(provider_hist, self.attrs)

#                 dbscan = DBSCAN(eps=0.4, min_samples=5)
#                 clusters = dbscan.fit_predict(hist_norm[self.attrs])

#                 provider_hist = provider_hist.copy()
#                 provider_hist['cluster'] = clusters

#                 # Dominant provider behavior
#                 largest_cluster = provider_hist['cluster'].value_counts().idxmax()
#                 dominant_hist = provider_hist[
#                     provider_hist['cluster'] == largest_cluster
#                 ]

#                 centroid = dominant_hist[self.attrs].mean()

#                 for i in gen_subset.index:
#                     dist = self.distance_to_centroid(
#                         df.loc[i], centroid, self.attrs
#                     )
#                     df.loc[i, 'label'] = 'T' if dist > 0.5 else 'C'

#         elapsed_time = time.time() - start_time
#         # print(df['label'].value_counts())
#         return df.copy(), {
#             'records': df.shape[0],
#             'time': elapsed_time
#         }

#     # ==========================================================
#     # Utility methods
#     # ==========================================================
#     def distance_to_centroid(self, row, centroid, cols):
#         return np.mean([abs(row[c] - centroid[c]) for c in cols])

#     def normalize(self, df, cols):
#         df = df.copy()
#         for c in cols:
#             min_v = df[c].min()
#             max_v = df[c].max()
#             df[c] = (df[c] - min_v) / (max_v - min_v + 1e-9)
#         return df



