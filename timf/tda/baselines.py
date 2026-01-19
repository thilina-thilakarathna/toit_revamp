import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from timf.trust_assessment.trust_assessment import TrustAssessment


class BaselineTda:
    def __init__(self):
        pass

    def baseline_detection(self, df_provider):
        """
        Applies three baseline tampering detection methods:
        1. Isolation Forest
        2. Local Outlier Factor
        3. One-Class SVM
        Returns:
            Dictionary of DataFrames with method-specific labels ('C' normal, 'T' tampered)
        """
        attrs = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']


        df_provider = df_provider.reset_index(drop=True).copy()
        X = df_provider[attrs]
        X += np.random.normal(0, 1e-6, X.shape)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = {}

        # ---------- Isolation Forest ----------
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso_labels = iso.fit_predict(X_scaled)
        df_iso = df_provider.copy()
        df_iso['label'] = ['C' if l != -1 else 'T' for l in iso_labels]
        results['IsolationForest'] = df_iso

        # ---------- Local Outlier Factor ----------
        lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
        lof_labels = lof.fit_predict(X_scaled)
        df_lof = df_provider.copy()
        df_lof['label'] = ['C' if l != -1 else 'T' for l in lof_labels]
        results['LOF'] = df_lof

        # ---------- One-Class SVM ----------
        ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        ocsvm_labels = ocsvm.fit_predict(X_scaled)
        df_ocsvm = df_provider.copy()
        df_ocsvm['label'] = ['C' if l != -1 else 'T' for l in ocsvm_labels]
        results['OCSVM'] = df_ocsvm

        return results

