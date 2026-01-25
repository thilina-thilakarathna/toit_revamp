import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ATTRS = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']
D_COLS = [f"D_{a}" for a in ATTRS]


class Sophistication:
    def __init__(self, random_state=42, n_clusters=3, n_init=30):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.n_init = n_init

    # =========================================================
    # Main: detect sophistication using majority cluster centroid
    # =========================================================
    def detect_sophistication(self, remote_data, tampered_data, weight,
                             mic_id=None,
                             return_debug=False,
                             plot_majority=False):
        """
        Uses ONLY confirmed local tampering records:
            (origin == 'R') AND (label_test == 'LT')

        Computes Impact/Significance/Trend per record, clusters in 3D (k=3),
        labels the microcell by the majority cluster centroid.

        Parameters
        ----------
        remote_data : DataFrame
            Remote reference pool containing origin records (serviceid + ATTRS)
        tampered_data : DataFrame
            Output containing at least:
            origin, label_test, providerid, serviceid, currect_microcell, ATTRS
        weight : list[float]
            Trust weights for ATTRS (len=6)
        mic_id : str or None
            If provided, restrict to that microcell; otherwise use all rows.
        return_debug : bool
            If True, return (label, debug_dict)
        plot_majority : bool
            If True, show 3D plot highlighting the majority cluster.

        Returns
        -------
        label : str or None
            'N' / 'K' / 'S' or None if insufficient data.
        debug (optional) : dict
            Includes feats with clusters, majority cluster id, centroid, etc.
        """

        df_sct = tampered_data.copy()

        # confirmed local tampering (LT) among received
        df_received = df_sct[(df_sct['origin'] == 'R') & (df_sct['label_test'] == 'LT')].copy()
        if df_received.empty:
            return (None, {}) if return_debug else None

        if mic_id is not None:
            df_received = df_received[df_received['currect_microcell'] == mic_id].copy()
            if df_received.empty:
                return (None, {}) if return_debug else None

        # Add D_* from origin (remote reference)
        df_received = self._process_received_data(df_received, remote_data)

        # Coerce D_* to numeric + drop invalid comparisons
        for c in D_COLS:
            df_received[c] = pd.to_numeric(df_received[c], errors='coerce')
        df_received = df_received.dropna(subset=D_COLS).copy()

        # Optional: filter zeros
        df_received = df_received[
            (df_received['speed'] != 0) &
            (df_received['latency'] != 0) &
            (df_received['bandwidth'] != 0) &
            (df_received['coverage'] != 0) &
            (df_received['reliability'] != 0) &
            (df_received['security'] != 0)
        ].copy()

        if df_received.empty:
            return (None, {}) if return_debug else None

        df_received = df_received.reset_index(drop=True)

        # compute features
        impact = np.asarray(self._calculate_impact(df_received, weight), dtype=float)
        sig    = np.asarray(self._calculate_significance(df_received, weight), dtype=float)
        trend  = np.asarray(self._calculate_trend(df_received, remote_data), dtype=float)

        feats = pd.DataFrame({"Impact": impact, "Significance": sig, "Trend": trend})
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna()

        if feats.empty:
            return (None, {}) if return_debug else None

        # If too few points, fallback to mean centroid
        if len(feats) < max(5, self.n_clusters):
            centroid = feats[["Impact", "Significance", "Trend"]].mean()
            label = self.classify_sophistication(
                float(centroid["Impact"]),
                float(centroid["Significance"]),
                float(centroid["Trend"])
            )

            mic = df_received["currect_microcell"].unique()
            print(mic, "few_points centroid=", centroid.to_dict(), "->", label)

            debug = {
                "feats": feats,
                "cluster_labels": None,
                "maj_cluster": None,
                "centroid": centroid,
                "label": label,
                "note": "fallback_mean_centroid"
            }
            return (label, debug) if return_debug else label

        # =========================================================
        # Clustering
        # =========================================================
        X = feats[["Impact", "Significance", "Trend"]].to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(X)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state
        )
        cl = kmeans.fit_predict(Xs)
        feats["cluster"] = cl

        # majority cluster id
        maj_cluster = int(feats["cluster"].value_counts().idxmax())
        feats["is_majority"] = (feats["cluster"] == maj_cluster)

        # centroid in ORIGINAL feature scale
        centroid = feats[feats["cluster"] == maj_cluster][["Impact", "Significance", "Trend"]].mean()

        # classify using centroid
        label = self.classify_sophistication(
            float(centroid["Impact"]),
            float(centroid["Significance"]),
            float(centroid["Trend"])
        )

        mic = df_received["currect_microcell"].unique()
        print(mic, "maj_cluster=", maj_cluster, "centroid=", centroid.to_dict(), "->", label)

        # optional: plot majority highlight
        if plot_majority:
            self.plot_majority_cluster_3d(feats, mic_id=mic_id or str(mic), maj_cluster=maj_cluster)

        debug = {
            "feats": feats,
            "cluster_labels": cl,
            "maj_cluster": maj_cluster,
            "centroid": centroid,
            "label": label,
            "kmeans": kmeans
        }

        return (label, debug) if return_debug else label

    # =========================================================
    # Decision rule (tuned)
    # =========================================================
    def classify_sophistication(self, mean_impact, mean_sig, mean_trend,
                                t_N_impact=0.12,
                                t_S_impact=0.04,
                                t_S_trend=0.95):
        # Naive: big deviations from origin
        if mean_impact >= t_N_impact:
            return "N"

        # Sophisticated: small deviation AND very camouflaged
        if mean_impact <= t_S_impact and mean_trend >= t_S_trend:
            return "S"

        # Knowledgeable: moderate deviation / partial camouflage
        return "K"

    # =========================================================
    # D_* construction (origin values)
    # =========================================================
    def _process_received_data(self, df_received, correct_data):
        for i, row in df_received.iterrows():
            origin_record = correct_data[correct_data['serviceid'] == row['serviceid']]
            if origin_record.empty:
                df_received.loc[i, D_COLS] = [np.nan] * 6
            else:
                df_received.loc[i, D_COLS] = origin_record[ATTRS].iloc[0].values
        return df_received

    # =========================================================
    # Impact (absolute deviation magnitude)
    # =========================================================
    def _calculate_impact(self, df_row, weight):
        d = np.column_stack([
            (df_row['speed']       - df_row['D_speed']).to_numpy(dtype=float),
            (df_row['latency']     - df_row['D_latency']).to_numpy(dtype=float),
            (df_row['bandwidth']   - df_row['D_bandwidth']).to_numpy(dtype=float),
            (df_row['coverage']    - df_row['D_coverage']).to_numpy(dtype=float),
            (df_row['reliability'] - df_row['D_reliability']).to_numpy(dtype=float),
            (df_row['security']    - df_row['D_security']).to_numpy(dtype=float),
        ])

        w = np.asarray(weight, dtype=float)
        w = w / (w.sum() + 1e-12)

        impact = (np.abs(d) @ w) / 5.0
        return np.clip(impact, 0, None)

    # =========================================================
    # Significance (targeting concentration; binary changed mask)
    # =========================================================
    def _calculate_significance(self, df_row, weight1, eps=1e-9):
        w = np.asarray(weight1, dtype=float)
        w = w / (w.sum() + eps)

        sig_list = []
        for i in range(len(df_row)):
            d = np.array([
                df_row.iloc[i]['speed']       - df_row.iloc[i]['D_speed'],
                df_row.iloc[i]['latency']     - df_row.iloc[i]['D_latency'],
                df_row.iloc[i]['bandwidth']   - df_row.iloc[i]['D_bandwidth'],
                df_row.iloc[i]['coverage']    - df_row.iloc[i]['D_coverage'],
                df_row.iloc[i]['reliability'] - df_row.iloc[i]['D_reliability'],
                df_row.iloc[i]['security']    - df_row.iloc[i]['D_security'],
            ], dtype=float)

            changed = (np.abs(d) > eps).astype(float)
            if changed.sum() == 0:
                sig_list.append(0.0)
                continue

            mass = w * changed
            s = float(mass.max() / (mass.sum() + eps))
            sig_list.append(s)

        return sig_list

    # =========================================================
    # Trend (closeness to provider mean; high means blends well)
    # =========================================================
    def _calculate_trend(self, df_row, dfcorrect):
        trend_list = []
        for i in range(len(df_row)):
            prov = df_row.iloc[i]['providerid']
            sid  = df_row.iloc[i]['serviceid']

            temp_df = dfcorrect[(dfcorrect['providerid'] == prov) & (dfcorrect['serviceid'] != sid)]
            if len(temp_df) == 0:
                trend_list.append(np.nan)
                continue

            means = temp_df[ATTRS].mean()

            diffs = np.array([
                df_row.iloc[i]['speed']       - means['speed'],
                df_row.iloc[i]['latency']     - means['latency'],
                df_row.iloc[i]['bandwidth']   - means['bandwidth'],
                df_row.iloc[i]['coverage']    - means['coverage'],
                df_row.iloc[i]['reliability'] - means['reliability'],
                df_row.iloc[i]['security']    - means['security'],
            ], dtype=float)

            closeness = 1.0 - (np.abs(diffs) / 5.0)
            closeness = np.clip(closeness, 0.0, 1.0)
            trend_list.append(float(np.mean(closeness)))

        return trend_list

    # =========================================================
    # Plot majority cluster highlighted (3D)
    # =========================================================
    def plot_majority_cluster_3d(self, feats, mic_id, maj_cluster):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        others = feats[feats["cluster"] != maj_cluster]
        main   = feats[feats["cluster"] == maj_cluster]

        # other clusters (faded)
        if not others.empty:
            ax.scatter(others["Impact"], others["Significance"], others["Trend"],
                       alpha=0.2, s=15, label="Other clusters")

        # majority cluster (highlighted)
        ax.scatter(main["Impact"], main["Significance"], main["Trend"],
                   s=40, marker="o", edgecolor="black", label="Majority cluster")

        # majority centroid
        centroid = main[["Impact", "Significance", "Trend"]].mean()
        ax.scatter(centroid["Impact"], centroid["Significance"], centroid["Trend"],
                   s=200, marker="X", color="red", label="Majority centroid")

        ax.set_xlabel("Impact")
        ax.set_ylabel("Significance")
        ax.set_zlabel("Trend")
        ax.set_title(f"Microcell {mic_id}: Majority Cluster Highlighted")
        ax.legend()
        plt.show()


# =========================================================
# Standalone helpers for your experiment visualization
# =========================================================
def compute_features_for_microcell(sophistication_obj, remote_data, tampered_data, mic_id, weight):
    df_sct = tampered_data.copy()

    df_received = df_sct[(df_sct['origin'] == 'R') & (df_sct['label_test'] == 'LT')].copy()
    if df_received.empty:
        return pd.DataFrame(columns=["Impact","Significance","Trend"])

    df_received = sophistication_obj._process_received_data(df_received, remote_data)

    for c in D_COLS:
        df_received[c] = pd.to_numeric(df_received[c], errors='coerce')
    df_received = df_received.dropna(subset=D_COLS).copy()

    df_received = df_received[df_received['currect_microcell'] == mic_id].copy()
    if df_received.empty:
        return pd.DataFrame(columns=["Impact","Significance","Trend"])

    df_received = df_received[
        (df_received['speed'] != 0) &
        (df_received['latency'] != 0) &
        (df_received['bandwidth'] != 0) &
        (df_received['coverage'] != 0) &
        (df_received['reliability'] != 0) &
        (df_received['security'] != 0)
    ].copy()
    if df_received.empty:
        return pd.DataFrame(columns=["Impact","Significance","Trend"])

    impact = np.asarray(sophistication_obj._calculate_impact(df_received, weight), dtype=float)
    sig    = np.asarray(sophistication_obj._calculate_significance(df_received, weight), dtype=float)
    trend  = np.asarray(sophistication_obj._calculate_trend(df_received, remote_data), dtype=float)

    feats = pd.DataFrame({"Impact": impact, "Significance": sig, "Trend": trend})
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def plot_microcell_3d_subplots(features_by_type, mic_id):
    fig = plt.figure(figsize=(18, 5))

    tampering_order = ["N", "K", "S"]
    titles = {"N": "Naive", "K": "Knowledgeable", "S": "Sophisticated"}

    all_feats = pd.concat(
        [df for df in features_by_type.values() if df is not None and not df.empty],
        ignore_index=True
    )
    if all_feats.empty:
        print("No data to plot.")
        return

    xlim = (all_feats["Impact"].min(), all_feats["Impact"].max())
    ylim = (all_feats["Significance"].min(), all_feats["Significance"].max())
    zlim = (all_feats["Trend"].min(), all_feats["Trend"].max())

    for i, ttype in enumerate(tampering_order, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        feats = features_by_type.get(ttype, None)

        if feats is None or feats.empty:
            ax.set_title(f"{titles[ttype]} (no data)")
            continue

        ax.scatter(feats["Impact"], feats["Significance"], feats["Trend"], s=20, alpha=0.7)

        ax.set_title(f"{titles[ttype]} Tampering")
        ax.set_xlabel("Impact")
        ax.set_ylabel("Significance")
        ax.set_zlabel("Trend")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    fig.suptitle(f"Microcell {mic_id}: Impact–Significance–Trend (LT Received)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_microcell_cluster_centers(features_by_type, mic_id, k=3, random_state=42):
    fig = plt.figure(figsize=(18, 5))
    tampering_order = ["N", "K", "S"]
    titles = {"N": "Naive", "K": "Knowledgeable", "S": "Sophisticated"}

    centers_by_type = {}
    all_centers = []

    for ttype in tampering_order:
        feats = features_by_type.get(ttype, None)
        if feats is None or feats.empty or len(feats) < k:
            centers = pd.DataFrame(columns=["Impact","Significance","Trend"])
        else:
            X = feats[["Impact","Significance","Trend"]].to_numpy(dtype=float)
            Xs = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=k, n_init=30, random_state=random_state)
            labels = km.fit_predict(Xs)

            tmp = feats.copy()
            tmp["cluster"] = labels
            centers = tmp.groupby("cluster")[["Impact","Significance","Trend"]].mean().reset_index(drop=True)

        centers_by_type[ttype] = centers
        if not centers.empty:
            all_centers.append(centers)

    if not all_centers:
        print("No cluster centers to plot.")
        return

    all_centers = pd.concat(all_centers, ignore_index=True)
    xlim = (all_centers["Impact"].min(), all_centers["Impact"].max())
    ylim = (all_centers["Significance"].min(), all_centers["Significance"].max())
    zlim = (all_centers["Trend"].min(), all_centers["Trend"].max())

    for i, ttype in enumerate(tampering_order, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        centers = centers_by_type[ttype]

        if centers.empty:
            ax.set_title(f"{titles[ttype]} (no data)")
            continue

        ax.scatter(centers["Impact"], centers["Significance"], centers["Trend"],
                   s=120, marker="X", edgecolor="black")

        for idx, row in centers.iterrows():
            ax.text(row["Impact"], row["Significance"], row["Trend"], f"C{idx}", fontsize=9)

        ax.set_title(f"{titles[ttype]} – Cluster Centers")
        ax.set_xlabel("Impact")
        ax.set_ylabel("Significance")
        ax.set_zlabel("Trend")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    fig.suptitle(f"Microcell {mic_id}: Cluster Centers in Impact–Significance–Trend", fontsize=14)
    plt.tight_layout()
    plt.show()


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# ATTRS = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']
# D_COLS = [f"D_{a}" for a in ATTRS]


# class Sophistication:
#     def __init__(self, random_state=42):
#         self.random_state = random_state

#     # -----------------------------
#     # Main
#     # -----------------------------
#     def detect_sophistication(self, remote_data, tampered_data, weight):
#         """
#         Uses ONLY confirmed local tampering records: (origin=='R' AND label_test=='LT')
#         Computes Impact/Significance/Trend per record, clusters in 3D (k=3),
#         and labels the microcell by the majority cluster centroid.
#         """

#         df_sct = tampered_data.copy()

#         # Guaranteed tampered local copies
#         df_received = df_sct[(df_sct['origin'] == 'R') & (df_sct['label_test'] == 'LT')].copy()
#         if df_received.empty:
#             return None

#         # Add D_* from origin (remote reference)
#         df_received = self._process_received_data(df_received, remote_data)

#         # Coerce D_* to numeric + drop invalid comparisons
#         for c in D_COLS:
#             df_received[c] = pd.to_numeric(df_received[c], errors='coerce')
#         df_received = df_received.dropna(subset=D_COLS).copy()

#         # Optional: filter zeros (keep if you want)
#         df_received = df_received[
#             (df_received['speed'] != 0) &
#             (df_received['latency'] != 0) &
#             (df_received['bandwidth'] != 0) &
#             (df_received['coverage'] != 0) &
#             (df_received['reliability'] != 0) &
#             (df_received['security'] != 0)
#         ].copy()

#         if df_received.empty:
#             return None

#         df_received = df_received.reset_index(drop=True)

#         impact = np.asarray(self._calculate_impact(df_received, weight), dtype=float)
#         sig    = np.asarray(self._calculate_significance(df_received, weight), dtype=float)
#         trend  = np.asarray(self._calculate_trend(df_received, remote_data), dtype=float)

#         feats = pd.DataFrame({"Impact": impact, "Significance": sig, "Trend": trend})
#         feats = feats.replace([np.inf, -np.inf], np.nan).dropna()

#         if len(feats) < 5:
#             # Too few points -> fallback to centroid by mean
#             centroid = feats[["Impact","Significance","Trend"]].mean()
#             label = self.classify_sophistication(float(centroid["Impact"]),
#                                                  float(centroid["Significance"]),
#                                                  float(centroid["Trend"]))
#             mic = df_received["currect_microcell"].unique()
#             print(mic, "few_points centroid=", centroid.to_dict(), "->", label)
#             return label

#         # -----------------------------
#         # Clustering in 3D
#         # -----------------------------
#         X = feats[["Impact","Significance","Trend"]].to_numpy(dtype=float)
#         Xs = StandardScaler().fit_transform(X)

#         kmeans = KMeans(n_clusters=3, n_init=30, random_state=self.random_state)
#         cl = kmeans.fit_predict(Xs)
#         feats["cluster"] = cl

#         maj_cluster = int(feats["cluster"].value_counts().idxmax())
#         centroid = feats[feats["cluster"] == maj_cluster][["Impact","Significance","Trend"]].mean()

#         label = self.classify_sophistication(float(centroid["Impact"]),
#                                              float(centroid["Significance"]),
#                                              float(centroid["Trend"]))

#         mic = df_received["currect_microcell"].unique()
#         print(mic, "maj_cluster=", maj_cluster, "centroid=", centroid.to_dict(), "->", label)
#         return label

#     # -----------------------------
#     # Decision rule (tuned to new scales)
#     # -----------------------------
#     def classify_sophistication(self, mean_impact, mean_sig, mean_trend,
#                                 t_N_impact=0.12,
#                                 t_S_impact=0.04,
#                                 t_S_trend=0.95):
#         # Naive: big deviations from origin
#         if mean_impact >= t_N_impact:
#             return "N"

#         # Sophisticated: small deviation AND very camouflaged
#         if mean_impact <= t_S_impact and mean_trend >= t_S_trend:
#             return "S"

#         # Knowledgeable: moderate deviation / partial camouflage
#         return "K"


#     # -----------------------------
#     # D_* construction (origin values)
#     # -----------------------------
#     def _process_received_data(self, df_received, correct_data):
#         # correct_data = remote reference pool
#         for i, row in df_received.iterrows():
#             origin_record = correct_data[correct_data['serviceid'] == row['serviceid']]
#             if origin_record.empty:
#                 df_received.loc[i, D_COLS] = [np.nan] * 6
#             else:
#                 df_received.loc[i, D_COLS] = origin_record[ATTRS].iloc[0].values
#         return df_received

#     # -----------------------------
#     # Impact: magnitude of deviation from origin, weighted
#     # -----------------------------
#     def _calculate_impact(self, df_row, weight):
#         d = np.column_stack([
#             (df_row['speed']       - df_row['D_speed']).to_numpy(dtype=float),
#             (df_row['latency']     - df_row['D_latency']).to_numpy(dtype=float),
#             (df_row['bandwidth']   - df_row['D_bandwidth']).to_numpy(dtype=float),
#             (df_row['coverage']    - df_row['D_coverage']).to_numpy(dtype=float),
#             (df_row['reliability'] - df_row['D_reliability']).to_numpy(dtype=float),
#             (df_row['security']    - df_row['D_security']).to_numpy(dtype=float),
#         ])
#         w = np.asarray(weight, dtype=float)
#         w = w / (w.sum() + 1e-12)

#         # Use absolute deviation for impact magnitude (more stable)
#         impact = (np.abs(d) @ w) / 5.0
#         return np.clip(impact, 0, None)

#     # -----------------------------
#     # Significance: targeting/concentration of changed-weight mass
#     #   ~1 if only 1 important attribute changed
#     #   ~1/6 if all attributes changed
#     # -----------------------------
#     def _calculate_significance(self, df_row, weight1, eps=1e-9):
#         w = np.asarray(weight1, dtype=float)
#         w = w / (w.sum() + eps)

#         sig_list = []
#         for i in range(len(df_row)):
#             d = np.array([
#                 df_row.iloc[i]['speed']       - df_row.iloc[i]['D_speed'],
#                 df_row.iloc[i]['latency']     - df_row.iloc[i]['D_latency'],
#                 df_row.iloc[i]['bandwidth']   - df_row.iloc[i]['D_bandwidth'],
#                 df_row.iloc[i]['coverage']    - df_row.iloc[i]['D_coverage'],
#                 df_row.iloc[i]['reliability'] - df_row.iloc[i]['D_reliability'],
#                 df_row.iloc[i]['security']    - df_row.iloc[i]['D_security'],
#             ], dtype=float)

#             changed = (np.abs(d) > eps).astype(float)
#             if changed.sum() == 0:
#                 sig_list.append(0.0)
#                 continue

#             mass = w * changed
#             # concentration in [~1/6, 1]
#             s = float(mass.max() / (mass.sum() + eps))
#             sig_list.append(s)

#         return sig_list

#     # -----------------------------
#     # Trend: camouflage = closeness to provider mean (excluding the same serviceid)
#     #   Returns in [0,1], high means "blends well"
#     # -----------------------------
#     def _calculate_trend(self, df_row, dfcorrect):
#         trend_list = []
#         for i in range(len(df_row)):
#             prov = df_row.iloc[i]['providerid']
#             sid  = df_row.iloc[i]['serviceid']

#             temp_df = dfcorrect[(dfcorrect['providerid'] == prov) & (dfcorrect['serviceid'] != sid)]
#             if len(temp_df) == 0:
#                 trend_list.append(np.nan)
#                 continue

#             means = temp_df[ATTRS].mean()

#             diffs = np.array([
#                 df_row.iloc[i]['speed']       - means['speed'],
#                 df_row.iloc[i]['latency']     - means['latency'],
#                 df_row.iloc[i]['bandwidth']   - means['bandwidth'],
#                 df_row.iloc[i]['coverage']    - means['coverage'],
#                 df_row.iloc[i]['reliability'] - means['reliability'],
#                 df_row.iloc[i]['security']    - means['security'],
#             ], dtype=float)

#             # Convert closeness to [0,1]
#             # close => ~1 ; far => ~0
#             closeness = 1.0 - (np.abs(diffs) / 5.0)
#             closeness = np.clip(closeness, 0.0, 1.0)

#             trend_list.append(float(np.mean(closeness)))

#         return trend_list
    



from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ATTRS = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']
D_COLS = [f"D_{a}" for a in ATTRS]


class Sophistication:
    def __init__(self):
        pass

    def detect_sophistication(self, remote_data, tampered_data, weight):
        if 'BMP' in tampered_data['label'].unique():
            print("@@@@@@@@@")

        
 
        # print(tampered_data.columns)
        df_sct = tampered_data.copy()


        df_received = df_sct[df_sct['label_test'] == 'LT']


   

        # df_sct = df_sct[df_sct['label'] == 'T'].copy()
        # df2 = df_sct
        # df2_received = df2[(df2['label_test'] == 'LT') & (df2['origin'] == 'R')].copy()
        if not df_received.empty:
            df2_received = self._process_received_data(df_received, remote_data)

        else:
            print("empty")
        combined_microcell_df = df2_received

        if combined_microcell_df.empty:
            return

        #     # filter out NC comparisons
        combined_microcell_df = combined_microcell_df[combined_microcell_df['D_speed'] != 'NC'].copy()

        #     # optional: filter zeros
        combined_microcell_df = combined_microcell_df[
                (combined_microcell_df['speed'] != 0) &
                (combined_microcell_df['latency'] != 0) &
                (combined_microcell_df['bandwidth'] != 0) &
                (combined_microcell_df['coverage'] != 0) &
                (combined_microcell_df['reliability'] != 0) &
                (combined_microcell_df['security'] != 0)
            ].copy()

        combined_microcell_df = combined_microcell_df.reset_index(drop=True)
        df = combined_microcell_df.copy().reset_index(drop=True)

        impact = self._calculate_impact(df, weight)
        sig = self._calculate_significance(df, weight)
        trend = self._calculate_trend(df, remote_data)

        

        new_df = pd.DataFrame({'Impact': impact, 'Significance': sig, 'Trend': trend})
        mean_impact = new_df['Impact'].mean()
        mean_sig    = new_df['Significance'].mean()
        mean_trend  = new_df['Trend'].mean()
        label = self.classify_sophistication(mean_impact, mean_sig, mean_trend)

        print(df_received['currect_microcell'].unique(),mean_impact,mean_sig,mean_trend,label)
        

        return label
    

    def classify_sophistication(self, mean_impact, mean_sig, mean_trend,
                                t_trend_high=0.95,
                                t_trend_low=0.2,
                                t_sig=0.05,
                                t_impact=0.05):

        # Naive
        if mean_trend < t_trend_low and mean_sig < t_sig:
            return "N"

        # Sophisticated
        if mean_trend >= t_trend_high and mean_impact <= t_impact:
            return "S"

        # Knowledgeable
        return "K"


    def _process_received_data(self, df_received, correct_data):
        # df_received = self._ensure_D_cols(df_received)

        for i, row in df_received.iterrows():
            origin_record = correct_data[correct_data['serviceid'] == row['serviceid']]
            if origin_record.empty:
                df_received.loc[i, D_COLS] = ['NC'] * 6
            else:
                df_received.loc[i, D_COLS] = origin_record[ATTRS].iloc[0].values

        return df_received

    def _process_generated_data(self, df_generated, correct_data):
        # df_generated = self._ensure_D_cols(df_generated)

        for i, row in df_generated.iterrows():
            provider_other_df = correct_data[
                (correct_data['providerid'] == row['providerid'])
            ]

            if len(provider_other_df) > 1:
                mean_values = provider_other_df[ATTRS].mean()
                df_generated.loc[i, D_COLS] = mean_values.values
            else:
                df_generated.loc[i, D_COLS] = ['NC'] * 6

        return df_generated

    def _calculate_impact(self, df_row, weight):
        # ensure numeric for subtraction (in case strings slipped through)
        d_speed = df_row['speed'] - df_row['D_speed']
        d_latency = df_row['latency'] - df_row['D_latency']
        d_bandwidth = df_row['bandwidth'] - df_row['D_bandwidth']
        d_coverage = df_row['coverage'] - df_row['D_coverage']
        d_reliability = df_row['reliability'] - df_row['D_reliability']
        d_security = df_row['security'] - df_row['D_security']

        impact = (
            d_speed * weight[0] +
            d_latency * weight[1] +
            d_bandwidth * weight[2] +
            d_coverage * weight[3] +
            d_reliability * weight[4] +
            d_security * weight[5]
        ).div(5)

        impact = impact.clip(lower=0)
        return impact

    def _calculate_significance(self, df_row, weight1):
        power = 1
        powered_matrix = [x ** power for x in weight1]
        total_sum = sum(powered_matrix)
        weight = [x / total_sum for x in powered_matrix]

        sig_list = []
        for i in range(len(df_row)):
            # if all attributes equal (weird sentinel), significance = 0
            if (df_row.iloc[i]['speed'] == df_row.iloc[i]['latency'] == df_row.iloc[i]['bandwidth'] ==
                df_row.iloc[i]['coverage'] == df_row.iloc[i]['reliability'] == df_row.iloc[i]['security']):
                sig_list.append(0)
                continue

            d = [
                df_row.iloc[i]['speed'] - df_row.iloc[i]['D_speed'],
                df_row.iloc[i]['latency'] - df_row.iloc[i]['D_latency'],
                df_row.iloc[i]['bandwidth'] - df_row.iloc[i]['D_bandwidth'],
                df_row.iloc[i]['coverage'] - df_row.iloc[i]['D_coverage'],
                df_row.iloc[i]['reliability'] - df_row.iloc[i]['D_reliability'],
                df_row.iloc[i]['security'] - df_row.iloc[i]['D_security']
            ]

            num_changed = sum(abs(x) > 0 for x in d)
            if num_changed == 0:
                sig_list.append(0)
                continue

            # counts only positive-direction changes (your original behavior)
            significance = (
                (1 if d[0] > 0 else 0) * weight[0] +
                (1 if d[1] > 0 else 0) * weight[1] +
                (1 if d[2] > 0 else 0) * weight[2] +
                (1 if d[3] > 0 else 0) * weight[3] +
                (1 if d[4] > 0 else 0) * weight[4] +
                (1 if d[5] > 0 else 0) * weight[5]
            )

            significance = max(significance, 0)
            sig_list.append(significance / num_changed)

        return sig_list

    def _calculate_trend(self, df_row, dfcorrect):
        trend_list = []

        for i in range(len(df_row)):
            if (df_row.iloc[i]['speed'] == df_row.iloc[i]['latency'] == df_row.iloc[i]['bandwidth'] ==
                df_row.iloc[i]['coverage'] == df_row.iloc[i]['reliability'] == df_row.iloc[i]['security']):
                trend_list.append(0)
                continue

            temp_df = dfcorrect[dfcorrect['providerid'] == df_row.iloc[i]['providerid']]
            temp_df = temp_df[temp_df['serviceid'] != df_row.iloc[i]['serviceid']]

            if temp_df.shape[0] > 0:
                means = temp_df[ATTRS].mean()

                s_trend = df_row.iloc[i]['speed'] - means['speed']
                l_trend = df_row.iloc[i]['latency'] - means['latency']
                b_trend = df_row.iloc[i]['bandwidth'] - means['bandwidth']
                c_trend = df_row.iloc[i]['coverage'] - means['coverage']
                r_trend = df_row.iloc[i]['reliability'] - means['reliability']
                sec_trend = df_row.iloc[i]['security'] - means['security']

                total_trend = (
                    self._normalize_trend(s_trend) +
                    self._normalize_trend(l_trend) +
                    self._normalize_trend(b_trend) +
                    self._normalize_trend(c_trend) +
                    self._normalize_trend(r_trend) +
                    self._normalize_trend(sec_trend)
                ) / 6

                total_trend = max(total_trend, 0)
            else:
                total_trend = 'NC'

            trend_list.append(total_trend)

        return trend_list

    def _normalize_trend(self, trend_value):
        min_value = 0
        max_value = 5
        score = 1 - (trend_value - min_value) / (max_value - min_value)
        return max(0, min(score, 1))
