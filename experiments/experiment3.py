"""Experiment 3: Effectiveness sweep across tampering percentages.

Implements the evaluation loop from `detection_new.ipynb` /
`sophistacation.ipynb` that applies tampering at different strengths and
collects accuracy/precision/recall for TDA and baseline detectors.
"""
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from components.general_operations import GeneralOp
from components.replication import Replication
from components.tampering import Tampering
from components.detection import Detection
from components.baseline_ifum import BaselineIFUM
from components.baseline_lof import BaselineLOF
from components.baseline_ocsvm import BaselineOCSVM
from data.cleaner import DataCleaner


class Experiment3:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.general = GeneralOp()
        self.replication = Replication()
        self.tampering = Tampering()
        self.detection = Detection()
        self.ifum = BaselineIFUM()
        self.lof = BaselineLOF()
        self.ocsvm = BaselineOCSVM()
        self.cleaner = DataCleaner()

    def run(self,
            data_df: Optional[pd.DataFrame] = None,
            tampering_levels: Optional[List[str]] = None,
            sp_percentages: Optional[List[int]] = None,
            repeats: int = 5,
            weights: Optional[List[float]] = None) -> pd.DataFrame:
        """Run effectiveness sweep.

        Returns a DataFrame where each row corresponds to a tampering level
        / strength and columns contain averaged metrics for each detector.
        """
        weights = weights or [0.3,0.1,0.2,0.1,0.1,0.2]
        tampering_levels = tampering_levels or ['N2','K3','S2']
        sp_percentages = sp_percentages or [10,20,30,40]

        if data_df is None:
            cleaned = self.cleaner.get_cleaned_data()
        else:
            cleaned = data_df

        fields = ['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell']
        dfin = self.general.slice_df(cleaned, [f for f in fields if f in cleaned.columns])
        dfin = self.general.add_a_column_with_a_value(dfin, 'origin', 'G')
        dfin = self.general.add_a_column_with_a_value(dfin, 'true_label', 'C')

        data_list = self.general.dataframe_devide_to_microcell_dictionary(dfin)
        data_rep = self.replication.replicate_totally(data_list, dfin)
        data_with_scores = self.general.trust_score_calculation(data_rep, weights)

        results = []

        try:
            from tqdm import tqdm
            progress_levels = lambda it: tqdm(it, desc='Tamper levels')
            progress_sp = lambda it: tqdm(it, desc='SP percentages')
            progress_repeats = lambda it: tqdm(it, desc='Repeats')
        except Exception:
            progress_levels = lambda it: it
            progress_sp = lambda it: it
            progress_repeats = lambda it: it

        for level in progress_levels(tampering_levels):
            for sp in progress_sp(sp_percentages):
                list_scores = []
                for _ in progress_repeats(range(repeats)):
                    # apply tampering and recompute scores
                    tampered = self.tampering.tamper_data1(data_with_scores, sp, level, sig=weights)
                    tampred_with_scores = self.general.trust_score_calculation(tampered, weights)

                    # run TDA detection and baselines
                    detected_tda = self.detection.detect_tampered_records(dfin, tampred_with_scores)
                    detected_ifum, _ = self.ifum.baseline_detection(tampred_with_scores)
                    detected_lof = self.lof.baseline_detection(tampred_with_scores)
                    detected_ocsvm = self.ocsvm.baseline_detection(tampred_with_scores)

                    # compute metrics
                    acc_tda = self.detection.accuracy(detected_tda['true_label'], detected_tda['label'])
                    pre_tda = self.detection.precision(detected_tda['true_label'], detected_tda['label'], 'T')
                    rec_tda = self.detection.recall(detected_tda['true_label'], detected_tda['label'], 'T')

                    acc_ifum = self.detection.accuracy(detected_ifum['true_label'], detected_ifum['label'])
                    pre_ifum = self.detection.precision(detected_ifum['true_label'], detected_ifum['label'], 'T')
                    rec_ifum = self.detection.recall(detected_ifum['true_label'], detected_ifum['label'], 'T')

                    acc_lof = self.detection.accuracy(detected_lof['true_label'], detected_lof['label'])
                    pre_lof = self.detection.precision(detected_lof['true_label'], detected_lof['label'], 'T')
                    rec_lof = self.detection.recall(detected_lof['true_label'], detected_lof['label'], 'T')

                    acc_ocsvm = self.detection.accuracy(detected_ocsvm['true_label'], detected_ocsvm['label'])
                    pre_ocsvm = self.detection.precision(detected_ocsvm['true_label'], detected_ocsvm['label'], 'T')
                    rec_ocsvm = self.detection.recall(detected_ocsvm['true_label'], detected_ocsvm['label'], 'T')

                    list_scores.append([
                        acc_tda, pre_tda, rec_tda,
                        acc_ifum, pre_ifum, rec_ifum,
                        acc_lof, pre_lof, rec_lof,
                        acc_ocsvm, pre_ocsvm, rec_ocsvm
                    ])

                # average across repeats
                arr = np.array(list_scores)
                means = np.nanmean(arr, axis=0)
                results.append({
                    'tamper_level': level,
                    'sp_percent': sp,
                    'Acc_tda': means[0], 'Pre_tda': means[1], 'Rec_tda': means[2],
                    'Acc_ifum': means[3], 'Pre_ifum': means[4], 'Rec_ifum': means[5],
                    'Acc_lof': means[6], 'Pre_lof': means[7], 'Rec_lof': means[8],
                    'Acc_ocsvm': means[9], 'Pre_ocsvm': means[10], 'Rec_ocsvm': means[11],
                })

        return pd.DataFrame(results)
"""Experiment 3: TSLD (Effectiveness: Moderate)"""


class Experiment3:
    """Simple skeleton for Experiment 3 (TSLD).

    Methods
    - run(data): run the experiment on `data` and return a result dict.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def run(self, data=None):
        """Run Experiment 3.

        Placeholder implementation.
        """
        print("Experiment3 (TSLD) running...")
        return {"experiment": "Experiment3", "method": "TSLD", "status": "done"}
