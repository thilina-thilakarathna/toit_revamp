
from typing import Optional, Dict, Any

from components.general_operations import GeneralOp
from components.replication import Replication
from components.tampering import Tampering
from components.detection import Detection
from data.cleaner import DataCleaner


class Experiment1:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.general = GeneralOp()
        self.replication = Replication()
        self.tampering = Tampering()
        self.detection = Detection()
        self.cleaner = DataCleaner()

        
    def run(self):

        return self.run(data_df=None, tamper_percent=10, tamper_type="N2")

    def run(self,
            data_df=None,
            tamper_percent: int = 10,
            tamper_type: str = "N2",
            weights: Optional[list] = None,
            rtis_only: bool = False) -> Dict[str, Any]:
        """Run the TDA experiment.

        Parameters
        - data_df: optional pre-loaded DataFrame. If None the `DataCleaner`
          will read the default CSVs.
        - tamper_percent: percentage of microcells to tamper
        - tamper_type: tampering scenario string (e.g. 'N2','K3','S2')
        - weights: weight vector for trust score calculation

        Returns a dict with keys: `metrics`, `detected_df`, `cleaned_df` and
        intermediate dicts used (replicated and scored dictionaries).
        """
        weights = weights or [0.3, 0.1, 0.2, 0.1, 0.1, 0.2]

        # 1) load / clean
        if data_df is None:
            cleaned = self.cleaner.get_cleaned_data(rtis_only=rtis_only)
        else:
            cleaned = data_df

        # 2) slice to the expected columns (same as notebook)
        fields = ['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell']
        dfin = self.general.slice_df(cleaned, [f for f in fields if f in cleaned.columns])

        # 3) mark origin and true label for original data
        dfin = self.general.add_a_column_with_a_value(dfin, 'origin', 'G')
        dfin = self.general.add_a_column_with_a_value(dfin, 'true_label', 'C')

        # 4) split by microcell and replicate
        data_list = self.general.dataframe_devide_to_microcell_dictionary(dfin)
        data_rep = self.replication.replicate_totally(data_list, dfin)

        # 5) compute trust scores on replicated data (show progress if tqdm available)
        try:
            self.general.show_progress = True
        except Exception:
            pass
        data_with_scores = self.general.trust_score_calculation(data_rep, weights)

        # 6) apply tampering and recompute trust scores for tampered data
        tampered = self.tampering.tamper_data1(data_with_scores, tamper_percent, tamper_type, sig=weights)
        tampered_with_scores = self.general.trust_score_calculation(tampered, weights)

        # 7) run detection (TDA)
        detected = self.detection.detect_tampered_records(dfin, tampered_with_scores)

        # 8) compute metrics
        metrics = {}
        try:
            y_true = detected['true_label']
            y_pred = detected['label']
            metrics['accuracy'] = self.detection.accuracy(y_true, y_pred)
            metrics['precision'] = self.detection.precision(y_true, y_pred, 'T')
            metrics['recall'] = self.detection.recall(y_true, y_pred, 'T')
        except Exception:
            metrics['accuracy'] = metrics['precision'] = metrics['recall'] = None

        return {
            'experiment': 'Experiment1',
            'method': 'TDA',
            'metrics': metrics,
            'detected_df': detected,
            'cleaned_df': dfin,
            'replicated_dict': data_rep,
            'scored_dict': data_with_scores,
            'tampered_scored_dict': tampered_with_scores,
        }
