from locale import normalize
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from timf.trust_assessment.trust_assessment import TrustAssessment


class Detection:
    def __init__(self, data_service):
        self.data_service = data_service
        self.trust_assessor = TrustAssessment([0.3,0.1,0.2,0.1,0.1,0.2])

    def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):

        df = local_data.reset_index(drop=True).copy()
        df['label'] = 'U'

        attrs = ['speed','latency','bandwidth','coverage','reliability','security']

        # ---------- Phase 1: Source consistency ----------
        if remote_data is not None and not remote_data.empty:
            for i in range(len(df)):
                sid = df.loc[i, 'serviceid']
                origin = remote_data[remote_data['serviceid'] == sid]

                if not origin.empty:
                    diff = np.mean([abs(df.loc[i, a] - origin.iloc[0][a]) for a in attrs])

                    if diff < 0.05:
                        df.loc[i, 'label'] = 'P-C'
                    else:
                        df.loc[i, 'label'] = 'T'

        # ---------- Phase 2: Cluster source-consistent records ----------
        pc_df = df[df['label'] == 'P-C']

        clean_reference = None

        if len(pc_df) >= 5:
            pc_norm = self.normalize(pc_df, attrs)

            dbscan = DBSCAN(eps=0.3, min_samples=4)
            clusters = dbscan.fit_predict(pc_norm[attrs])

            pc_df['cluster'] = clusters

            # Largest cluster = honest majority
            largest_cluster = pc_df['cluster'].value_counts().idxmax()
            clean_pc = pc_df[pc_df['cluster'] == largest_cluster]

            # Mark others as tampered (BMA)
            df.loc[pc_df[pc_df['cluster'] != largest_cluster].index, 'label'] = 'T'
            df.loc[clean_pc.index, 'label'] = 'C'

            # Build pseudo-reference
            clean_reference = clean_pc[attrs].mean()

        # ---------- Phase 3: Bootstrap detection for local-only ----------
        local_only = df[df['label'] == 'U']

        if clean_reference is not None:
            for i in local_only.index:
                dist = self.distance_to_centroid(df.loc[i], clean_reference, attrs)
                df.loc[i, 'label'] = 'C' if dist < 0.1 else 'T'
        else:
            # Conservative fallback
            df.loc[df['label'] == 'U', 'label'] = 'C'

        return df.copy()



    def distance_to_centroid(self,row, centroid, cols):
        return np.mean([abs(row[c] - centroid[c]) for c in cols])



    # def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):

    #     df = local_data.reset_index(drop=True).copy()
    #     df['label'] = 'U'   # Unknown initially

    #     attrs = ['speed','latency','bandwidth','coverage','reliability','security']

    #     # ---------- Stage 1: Origin consistency check ----------
    #     if remote_data is not None and not remote_data.empty:

    #         for i in range(len(df)):
    #             sid = df.loc[i, 'serviceid']
    #             origin = remote_data[remote_data['serviceid'] == sid]

    #             if not origin.empty:
    #                 diff = 0
    #                 for a in attrs:
    #                     diff += abs(df.loc[i, a] - origin.iloc[0][a])

    #                 diff = diff / len(attrs)

    #                 # tolerance-based decision (realistic)
    #                 if diff < 0.05:
    #                     df.loc[i, 'label'] = 'C'
    #                 else:
    #                     df.loc[i, 'label'] = 'T'

    #     # ---------- Stage 2: Behavioral anomaly detection ----------
    #     unlabeled = df[df['label'] == 'U']

    #     if len(unlabeled) >= 5:
    #         df_norm = self.normalize(unlabeled, attrs)

    #         dbscan = DBSCAN(eps=0.35, min_samples=4)
    #         clusters = dbscan.fit_predict(df_norm[attrs])

    #         for idx, cluster in zip(unlabeled.index, clusters):
    #             if cluster == -1:
    #                 df.loc[idx, 'label'] = 'T'
    #             else:
    #                 df.loc[idx, 'label'] = 'C'

    #     # remaining unknowns → conservative decision
    #     df.loc[df['label'] == 'U', 'label'] = 'C'

    #     # ---------- Clean data for trust calculation ----------
    #     clean_df = df[df['label'] == 'C'].copy()
    #     # trust_score = self.trust_assessor.calculate(clean_df) if not clean_df.empty else 0.0

    #     return df.copy()
    
    def normalize(self,df, cols):
        df = df.copy()
        for c in cols:
            min_v = df[c].min()
            max_v = df[c].max()
            df[c] = (df[c] - min_v) / (max_v - min_v + 1e-9)
        return df


       
 
