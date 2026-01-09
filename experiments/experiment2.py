"""Experiment 2: Scalability / timing comparison (TDA vs IFUM).

Implements the timing experiment from `sophistacation.ipynb`: for a chosen
microcell it measures time taken by the TDA detection (using
`Detection.detect_tampered_records_tith_time`) and the IFUM baseline
(`BaselineIFUM.baseline_detection`) while increasing number of records.
"""
from typing import Optional, Dict, Any, List
import pandas as pd

from components.general_operations import GeneralOp
from components.replication import Replication
from components.tampering import Tampering
from components.detection import Detection
from components.baseline_ifum import BaselineIFUM
from data.cleaner import DataCleaner


class Experiment2:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.general = GeneralOp()
        self.replication = Replication()
        self.tampering = Tampering()
        self.detection = Detection()
        self.baseline = BaselineIFUM()
        self.cleaner = DataCleaner()

    def run(self):
        print("Experiment2 (TDAQ) running...")

    def report(self):
        print("Experiment 2 (TDAQ) completed. Timing results are available in the output dictionary.")

    # def run(self,
    #         data_df: Optional[pd.DataFrame] = None,
    #         tamper_type: str = 'K3',
    #         tamper_percent: int = 100,
    #         repeats: int = 3,
    #         step: int = 250,
    #        microcell_key: Optional[str] = None) -> Dict[str, Any]:
    #     print("Experiment2 (TDAQ) running...")
        
    # def report(self, output: Dict[str, Any]) -> None:
    #     print("Experiment 2 (TDAQ) completed. Timing results are available in the output dictionary.")

#         """Run timing experiment.

#         Returns a dict with averaged timing series for TDA and IFUM and raw
#         concatenated DataFrames used to compute averages.
#         """
#         # load/clean
#         if data_df is None:
#             cleaned = self.cleaner.get_cleaned_data()
#         else:
#             cleaned = data_df

#         fields = ['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell']
#         dfin = self.general.slice_df(cleaned, [f for f in fields if f in cleaned.columns])
#         dfin = self.general.add_a_column_with_a_value(dfin, 'origin', 'G')
#         dfin = self.general.add_a_column_with_a_value(dfin, 'true_label', 'C')

#         data_list = self.general.dataframe_devide_to_microcell_dictionary(dfin)
#         data_rep = self.replication.replicate_totally(data_list, dfin)
#         data_with_scores = self.general.trust_score_calculation(data_rep, [0.3,0.1,0.2,0.1,0.1,0.2])

#         # create an initial tampered sample and pick a microcell if not provided
#         tampered_sample = self.tampering.tamper_data1(data_with_scores, tamper_percent, tamper_type, sig=[0.3,0.1,0.2,0.1,0.1,0.2])
#         if microcell_key is None:
#             microcell_key = next(iter(tampered_sample.keys()))

#         data_use = tampered_sample.get(microcell_key)
#         if data_use is None or data_use.empty:
#             raise RuntimeError(f"No data available for microcell '{microcell_key}'")

#         dfs_time1: List[pd.DataFrame] = []
#         dfs_time2: List[pd.DataFrame] = []

#         # repeats to smooth noise; show progress if tqdm is available
#         try:
#             from tqdm import tqdm
#             progress_outer = lambda it: tqdm(it, desc='Repeats')
#             progress_inner = lambda it: tqdm(it, desc='Record counts')
#         except Exception:
#             progress_outer = lambda it: it
#             progress_inner = lambda it: it

#         for _ in progress_outer(range(repeats)):
#             time1_list = []
#             time2_list = []
#             indices = list(range(step, len(data_use) + 1, step))
#             for i in progress_inner(indices):
#                 selected_records = data_use[:i]
#                 input_data = {microcell_key: selected_records}

#                 # TDA timing
#                 _, time1 = self.detection.detect_tampered_records_tith_time(dfin, input_data)
#                 # IFUM timing
#                 _, time2 = self.baseline.baseline_detection(input_data)

#                 # each returned time list contains pairs [num_records, elapsed_time]
#                 time1_list.append(time1[0])
#                 time2_list.append(time2[0])

#             df1 = pd.DataFrame(time1_list, columns=['Value', 'Time'])
#             df2 = pd.DataFrame(time2_list, columns=['Value', 'Time'])
#             dfs_time1.append(df1)
#             dfs_time2.append(df2)

#         df_time1_concat = pd.concat(dfs_time1)
#         df_time2_concat = pd.concat(dfs_time2)

#         avg_time_1 = df_time1_concat.groupby('Value')['Time'].mean()
#         avg_time_2 = df_time2_concat.groupby('Value')['Time'].mean()

#         return {
#             'microcell': microcell_key,
#             'avg_time_tda': avg_time_1,
#             'avg_time_ifum': avg_time_2,
#             'raw_concat_tda': df_time1_concat,
#             'raw_concat_ifum': df_time2_concat,
#         }
# """Experiment 2: TDAQ (Scalability: High)"""


# class Experiment2:
#     """Simple skeleton for Experiment 2 (TDAQ).

#     Methods
#     - run(data): run the experiment on `data` and return a result dict.
#     """

#     def __init__(self, config=None):
#         self.config = config or {}

#     def run(self, data=None):
#         """Run Experiment 2.

#         Placeholder implementation.
#         """
#         print("Experiment2 (TDAQ) running...")
#         return {"experiment": "Experiment2", "method": "TDAQ", "status": "done"}
