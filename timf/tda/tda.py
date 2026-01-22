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
        # print(local_data.shape[0])
        """
        Updated to support AUC scoring:

        - Adds two continuous scores:
            * score_R: Part 1 (origin == 'R') deviation-from-origin score (higher => more likely tampered)
            * score_G: Part 2 (origin == 'G') isolation score using k-NN distance in scaled space
                       (higher => more likely tampered)
        - Keeps your original labeling logic intact.
        - Returns df_out with: label, score_R, score_G, and a unified score column (score)
          where score = score_R for R and score_G for G (useful for a single overall AUC if desired).
        """

        # Keep original values for output
        df_out = local_data.reset_index(drop=True).copy()
        df_out['label'] = 'U'  # U: unknown, C: clean, T: tampered (S/NS internal)

        # ---- NEW: add score columns
        df_out['score_R'] = np.nan
        df_out['score_G'] = np.nan
        df_out['score'] = np.nan  # unified score: score_R for R, score_G for G

        # Defensive copy for remote
        remote_ref = None
        if remote_data is not None and not remote_data.empty:
            remote_ref = remote_data.copy()

        # ==========================================================
        # Preprocessing (NOT TIMED): prepare scaler for DBSCAN only
        # Fit on remote (preferred) else local.
        # ==========================================================
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
        # Score: mean absolute deviation from origin (higher => more likely tampered)
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

                # ---- NEW: score_R as deviation magnitude
                diff = np.abs(local_vals - origin_vals)
                score_r = float(np.mean(diff))  # alternative: float(np.linalg.norm(diff))
                df_out.loc[i, 'score_R'] = score_r
                df_out.loc[i, 'score'] = score_r

                # tolerant equality to avoid float issues
                is_equal = np.all(np.isclose(local_vals, origin_vals, rtol=1e-5, atol=1e-8))
                df_out.loc[i, 'label'] = 'C' if is_equal else 'T'

        # ==========================================================
        # Part 2: Provider-outlier test for origin == 'G'
        # Score: k-NN distance to provider reference points in scaled space (higher => more isolated)
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
                if provider_other.shape[0] < 2:
                    df_out.loc[j, 'label'] = 'NS'
                    # score_G stays NaN (cannot compute meaningful isolation)
                    continue

                # Build clustering set: provider_other + current record
                X_ref = provider_other[self.attrs].to_numpy(dtype=float)
                x_q = df_out.loc[j, self.attrs].to_numpy(dtype=float).reshape(1, -1)

                # Scale
                X_ref_s = scaler.transform(X_ref)
                x_q_s = scaler.transform(x_q)

                # ---- NEW: score_G using k-NN distance (k matches min_samples by default)
                dists = np.linalg.norm(X_ref_s - x_q_s, axis=1)
                k = min(5, dists.shape[0])  # match DBSCAN min_samples=5 when possible
                knn_dist = float(np.partition(dists, k - 1)[k - 1])  # k-th smallest distance
                df_out.loc[j, 'score_G'] = -knn_dist
                df_out.loc[j, 'score'] = -knn_dist

                # DBSCAN decision (unchanged)
                X = np.vstack([X_ref_s, x_q_s])
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels = dbscan.fit_predict(X)

                # The query point is the last row
                df_out.loc[j, 'label'] = 'S' if labels[-1] == -1 else 'NS'

        # ==========================================================
        # Part 3: Global rule based on received tamper ratio
        # Convert S/NS -> final C/T for generated records (unchanged)
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
