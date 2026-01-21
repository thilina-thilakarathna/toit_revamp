import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class BaselineTda:
    def __init__(self):
        self.attrs = ['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']

    # -----------------------
    # Shared preprocessing
    # -----------------------
    def _prepare_X(self, df_provider):
        df_provider = df_provider.reset_index(drop=True).copy()

        X = df_provider[self.attrs].to_numpy(dtype=float)

        # tiny noise to avoid duplicate-value warnings (esp LOF)
        X = X + np.random.normal(0, 1e-6, X.shape)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return df_provider, X_scaled

    # =========================================================
    # Public API: 3 separate methods (each runs end-to-end)
    # =========================================================
    def run_isolation_forest(self, df_provider, contamination=0.1, random_state=42):
        df_provider, X_scaled = self._prepare_X(df_provider)

        start = time.time()
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        labels = iso.fit_predict(X_scaled)
        elapsed = time.time() - start

        df_out = df_provider.copy()
        df_out['label'] = np.where(labels == -1, 'T', 'C')
        return df_out, {'records': df_out.shape[0], 'time': elapsed}

    def run_lof(self, df_provider, n_neighbors=10, contamination=0.1):
        df_provider, X_scaled = self._prepare_X(df_provider)

        start = time.time()
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        labels = lof.fit_predict(X_scaled)
        elapsed = time.time() - start

        df_out = df_provider.copy()
        df_out['label'] = np.where(labels == -1, 'T', 'C')
        return df_out, {'records': df_out.shape[0], 'time': elapsed}

    def run_ocsvm(self, df_provider, nu=0.1, kernel='rbf', gamma='scale'):
        df_provider, X_scaled = self._prepare_X(df_provider)

        start = time.time()
        ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        labels = ocsvm.fit_predict(X_scaled)
        elapsed = time.time() - start

        df_out = df_provider.copy()
        df_out['label'] = np.where(labels == -1, 'T', 'C')
        return df_out, {'records': df_out.shape[0], 'time': elapsed}
