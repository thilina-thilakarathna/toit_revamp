import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Detection:
    def __init__(self, data_service):
        self.data_service = data_service
        # all attributes are normalized: 0 (bad) .. 5 (good)
        self.attrs = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

    def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
        """
        Detects tampering in two parts:
          - Received (origin == 'R'): compare by serviceid against remote_ref -> label as 'T' or 'BMP'
            then re-label BMP into {C,T} using consensus-based deviation + optional DBSCAN stabilization.
          - Generated (origin == 'G'): label {C,T} using kNN distance to reference distribution built
            from received records labeled as C.

        Returns:
          df_out, stats_dict
        """
        start_time = time.time()

        # ==========================================================
        # Base output
        # ==========================================================
        df_out = local_data.reset_index(drop=True).copy()
        df_out['label'] = 'U'        # U: unknown
        df_out['score_R'] = np.nan
        df_out['score_G'] = np.nan
        df_out['score_B'] = np.nan
        df_out['score'] = np.nan     # unified score
        df_out['pred'] = np.nan      # used internally for BMP consensus step


        # remote_ref = None
        # if remote_data is not None and not remote_data.empty:
        remote_ref = remote_data.copy()

        # ==========================================================
        # Split by origin
        # ==========================================================
        received_df = df_out[df_out['origin'] == 'R'].copy()
        generated_df = df_out[df_out['origin'] == 'G'].copy()



        remote_by_sid = remote_ref.set_index('serviceid', drop=False)
        df_remote_1 = None
    

        for idx, row in received_df.iterrows():
            sid = row.get('serviceid', None)

            if sid not in remote_by_sid.index:
                # print("No data")
                received_df.at[idx, 'label'] = 'BMP'
            else:
                # print("ok")
                origin_row = remote_by_sid.loc[sid]
                

                if isinstance(origin_row, type(remote_by_sid.iloc[:0])):  # DataFrame
                    origin_row = origin_row.iloc[0]

                
                local_vals = row[self.attrs].to_numpy(dtype=float)
                origin_vals = origin_row[self.attrs].to_numpy(dtype=float)
                diff = np.abs(local_vals - origin_vals)

                
                score_r = float(np.mean(diff))
                received_df.at[idx, 'score_R'] = score_r

                is_equal = np.all(np.isclose(local_vals, origin_vals, rtol=1e-5, atol=1e-8))
                received_df.at[idx, 'label'] = 'T' if not is_equal else 'BMP'
                received_df.at[idx, 'label_test'] = 'LT' if not is_equal else 'BMP'

   

        bmp_df = received_df[received_df['label'] == 'BMP'].copy()
 
        sproviders = bmp_df['gen_microcell'].unique()
        anchor_df = self.data_service.get_correct_data(sproviders,provider_id)

        ATTRS = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

        if bmp_df.empty or anchor_df is None or anchor_df.empty or len(anchor_df) < 5:
            # fallback policy
            bmp_df['label'] = 'C'  # or keep 'BMP'/'U'
            # pass
        else:
            Xa = anchor_df[ATTRS].to_numpy(dtype=float)
            Xb = bmp_df[ATTRS].to_numpy(dtype=float)

            # Fit scaler on anchor ONLY
            scaler = StandardScaler().fit(Xa)
            Xa_s = scaler.transform(Xa)
            Xb_s = scaler.transform(Xb)

            # kNN score (mean distance to k nearest anchor points)
            k = min(10, len(anchor_df))  # try 5 or 10
            nn = NearestNeighbors(n_neighbors=k).fit(Xa_s)

            d_b, _ = nn.kneighbors(Xb_s)
            score_b = d_b.mean(axis=1)

            # Threshold from anchor self-scores
            d_a, _ = nn.kneighbors(Xa_s)
            score_a = d_a.mean(axis=1)

            thr = float(np.quantile(score_a, 0.8))  # tune 0.90–0.99

            bmp_df.loc[bmp_df.index, 'score_B'] = score_b

            bmp_df.loc[bmp_df.index, 'score_R'] = bmp_df.loc[bmp_df.index, 'score_R']  # keep your phase-1 score_R
            bmp_df.loc[bmp_df.index, 'score']   = score_b                              # unified score for AUC if needed
            bmp_df.loc[bmp_df.index, 'label']   = np.where(score_b > thr, 'T', 'C')
      
        received_df.loc[bmp_df.index, 'label'] = bmp_df['label']
        received_df.loc[bmp_df.index, 'score_B'] = bmp_df['score_B']

  

        gen_df = generated_df.copy()


        if (gen_df is not None) and (not gen_df.empty) and (anchor_df is not None) and (not anchor_df.empty) and (len(anchor_df) >= 5):

            Xa = anchor_df[ATTRS].to_numpy(dtype=float)
            Xg = gen_df[ATTRS].to_numpy(dtype=float)

            # --- Scale using anchor only
            scaler = StandardScaler().fit(Xa)
            Xa_s = scaler.transform(Xa)
            Xg_s = scaler.transform(Xg)

            # --- kNN distance to anchor (base distance signal)
            k = min(10, len(anchor_df))
            nn = NearestNeighbors(n_neighbors=k).fit(Xa_s)

            d_g, _ = nn.kneighbors(Xg_s)
            knn_g = d_g.mean(axis=1)

            d_a, _ = nn.kneighbors(Xa_s)
            knn_a = d_a.mean(axis=1)

            # robust center/scale on anchor
            mu  = np.median(knn_a)
            mad = np.median(np.abs(knn_a - mu)) + 1e-9

            # NK-mode: FAR (larger knn distance => suspicious)
            z_far_g = (knn_g - mu) / mad
            z_far_a = (knn_a - mu) / mad

            # S-mode: CLOSE (smaller knn distance => suspicious)
            z_close_g = (mu - knn_g) / mad
            z_close_a = (mu - knn_a) / mad

            # S-mode: PERFECT (closer to all-5 vector => suspicious)
            ideal = np.full(len(ATTRS), 5.0, dtype=float)
            dist_ideal_g = np.linalg.norm(Xg - ideal, axis=1)
            dist_ideal_a = np.linalg.norm(Xa - ideal, axis=1)

            mu_i  = np.median(dist_ideal_a)
            mad_i = np.median(np.abs(dist_ideal_a - mu_i)) + 1e-9
            z_perfect_g = (mu_i - dist_ideal_g) / mad_i
            z_perfect_a = (mu_i - dist_ideal_a) / mad_i

            # S-mode: LOW VAR (lower within-record variance => suspicious)
            var_g = np.var(Xg, axis=1)
            var_a = np.var(Xa, axis=1)

            mu_v  = np.median(var_a)
            mad_v = np.median(np.abs(var_a - mu_v)) + 1e-9
            z_lowvar_g = (mu_v - var_g) / mad_v
            z_lowvar_a = (mu_v - var_a) / mad_v

            # --- Build two separate scores
            score_NK_g = z_far_g
            score_NK_a = z_far_a

            score_S_g  = 0.45 * z_close_g + 0.35 * z_perfect_g + 0.20 * z_lowvar_g
            score_S_a  = 0.45 * z_close_a + 0.35 * z_perfect_a + 0.20 * z_lowvar_a

            # --- Separate thresholds from anchor (tune quantiles)
            q_nk = 0.95
            q_s  = 0.95
            thr_NK = float(np.quantile(score_NK_a, q_nk))
            thr_S  = float(np.quantile(score_S_a,  q_s))

            # --- Decision: tampered if either mode is strong
            gen_df["label"] = np.where(
                (score_NK_g > thr_NK) | (score_S_g > thr_S),
                "T", "C"
            )

            # --- Unified continuous score for AUC: normalize by thresholds, then max
            # (ensures "higher => more suspicious" consistently across modes)
            score_NK_norm = score_NK_g / (thr_NK + 1e-9)
            score_S_norm  = score_S_g  / (thr_S  + 1e-9)

            gen_df["score_G"] = 0.35 * score_NK_norm + 0.65 * score_S_norm
            gen_df["score"]   = gen_df["score_G"]
            # gen_df["score"]   = gen_df["score_G"]

        else:
            # Not enough info to score generated
            if gen_df is not None and not gen_df.empty:
                gen_df.loc[:, 'label'] = 'NS'  # or 'U'
                gen_df.loc[:, 'score_G'] = np.nan
                gen_df.loc[:, 'score'] = np.nan
                
        generated_df = gen_df

        df_out.loc[received_df.index, received_df.columns] = received_df
        df_out.loc[generated_df.index, generated_df.columns] = generated_df

        elapsed_time = time.time() - start_time
        return df_out.copy(), {'records': df_out.shape[0], 'time': elapsed_time}


    def _to_bin(self,series, pos='T'):
        s = pd.Series(series).astype(str)
        return (s == pos).astype(int).to_numpy()

    def _safe_auc(self,y_true_bin, scores):
        scores = np.asarray(scores, dtype=float)
        m = np.isfinite(scores)
        y = y_true_bin[m]
        s = scores[m]
        if len(y) == 0 or len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, s))

    def print_cmp(self,title, df, pred_col='label', true_col='true_label', score_col=None, pos='T'):
        if df is None or df.empty:
            print(f"\n[{title}] empty")
            return
        if pred_col not in df.columns or true_col not in df.columns:
            print(f"\n[{title}] missing columns: pred={pred_col} or true={true_col}")
            return

        y_true = self._to_bin(df[true_col], pos=pos)
        y_pred = self._to_bin(df[pred_col], pos=pos)

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)

        auc = np.nan
        if score_col is not None and score_col in df.columns:
            auc = self._safe_auc(y_true, df[score_col].to_numpy())

        print(f"\n[{title}] n={len(df)}  acc={acc:.4f}  prec={pre:.4f}  rec={rec:.4f}  f1={f1:.4f}  auc={auc if np.isfinite(auc) else 'nan'}")