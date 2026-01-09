
from typing import Optional, Dict, Any
# from components.general_operations import GeneralOp
from components import detection
from components.general_operations import GeneralOp
from components.replication import Replication
from components.tampering import Tampering
from components.detection import Detection

from components.baseline_ifum import BaselineIFUM, Detection


from components.baseline_lof import BaselineLOF, Detection

from components.baseline_ocsvm import BaselineOCSVM, Detection

from data.cleaner import DataCleaner
import gc


class Experiment1:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.replication = Replication()
        self.tampering = Tampering()
        self.detection = Detection()
        self.cleaner = DataCleaner()
        self.general=GeneralOp()

        self.ifum_baseline = BaselineIFUM()
        self.lof_baseline = BaselineLOF()
        self.ocsvm_baseline = BaselineOCSVM()

        self.data_with_scores = None


    def setup(self):
        # dfin = self.general.open_file_csv('data_alg_16000.csv')
        if self.data_with_scores is None:
            dfin = self.cleaner.get_cleaned_data()

            dfin = self.general.slice_df(dfin,['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell'])
            dfin=self.general.add_a_column_with_a_value(dfin,'origin','G')
            dfin=self.general.add_a_column_with_a_value(dfin,'true_label','C')
            data_list = self.general.dataframe_devide_to_microcell_dictionary(dfin)
            data_rep=self.replication.replicate_totally(data_list,dfin)
            # self.replication.visualize_samples_per_microcell(data_rep)
            # merged_df = self.general.dictionary_to_merged_df(data_rep)
            # self.general.save_file(merged_df,'replicated_source.xlsx')
            self.data_with_scores = data_rep
            print(data_rep)
            # data_with_scores = self.general.trust_score_calculation(data_rep,[0.3,0.1,0.2,0.1,0.1,0.2])
            # self.general.visualize_data_one_value(data_with_scores)
        else:
            print("Data with scores already set.")
        # print(len(dfin['microcell'].unique()))

    def run(self):
        print("Experiment1 (TDA) running...")

        # experiment plan: run N K S
        # we nned to run 

        tampering_level = ['N', 'K', 'S']
        for tampering_type in tampering_level:
            print("Tampering type: " + tampering_type)

        for tamper_percentage in range(10,100, 10):
            
            print(str(tamper_percentage))

            tampered_data_temp = self.tampering.tamper_data1(self.data_with_scores, tamper_percentage, tampering_level, sig=[0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
            in1 = self.general.trust_score_calculation(tampered_data_temp)
            # 1. Store your detectors/results in a dictionary
            detectors = {
                'tda': self.detection.detect_tampered_records(dfin, in1),
                'ifum': self.ifum_baseline.baseline_detection(in1),
                'lof': self.lof_baseline.baseline_detection(in1),
                'ocsvm': self.ocsvm_baseline.baseline_detection(in1)
            }

            # 2. Use a list comprehension to calculate all scores dynamically
            current_scores = []
            for key in ['tda', 'ifum', 'lof', 'ocsvm']:
                res = detectors[key]
                true = res['true_label']
                pred = res['label']
                
                # Extend the list with the three metrics for this model
                current_scores.extend([
                    detection.accuracy(true, pred),
                    detection.precision(true, pred, 'T'),
                    detection.recall(true, pred, 'T')
                ])

            list_scores.append(current_scores)


            # print(tampered_data_temp)
                # if hasattr(tampered_data_temp, 'value_counts'):
                #     print(tampered_data_temp['true_label'].value_counts())
                
           
                # # --- MEMORY CLEANUP ---
                # del tampered_data_temp
                # gc.collect()

     




        









# for j in range(1, 5):
#     list_scores = []
#     for i in range(1): 
#         print(str(j) + "  " + str(i))
        
#         # Tamper data
#         # tampered_data_temp = tampering.tamper_data1(data_with_scores, (sp_percentage * j) / 2, tampering_level, sig=[0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
        
#         # Calculate trust scores
#         # tampered_data = general.trust_score_calculation(tampered_data_temp, [0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
#         in1 = general.trust_score_calculation(tampering.tamper_data1(data_with_scores, (sp_percentage * j) / 2, tampering_level, sig=[0.3, 0.1, 0.2, 0.1, 0.1, 0.2]), [0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
#         in2= general.trust_score_calculation(tampering.tamper_data1(data_with_scores, (sp_percentage * j) / 2, tampering_level, sig=[0.3, 0.1, 0.2, 0.1, 0.1, 0.2]), [0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
#         in3= general.trust_score_calculation(tampering.tamper_data1(data_with_scores, (sp_percentage * j) / 2, tampering_level, sig=[0.3, 0.1, 0.2, 0.1, 0.1, 0.2]), [0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
#         in4 = general.trust_score_calculation(tampering.tamper_data1(data_with_scores, (sp_percentage * j) / 2, tampering_level, sig=[0.3, 0.1, 0.2, 0.1, 0.1, 0.2]), [0.3, 0.1, 0.2, 0.1, 0.1, 0.2])
#         # Detect tampered records using the original baseline
#         data_detected_tda = detection.detect_tampered_records(dfin, in1)
#         data_detected_ifum = ifum_baseline.baseline_detection(in2)
#         data_detected_lof = lof_baseline.baseline_detection(in3)
#         data_detected_ocsvm = ocsvm_baseline.baseline_detection(in4)
        
#         # Calculate scores for both baselines
#         list_scores.append([
#             detection.accuracy(data_detected_tda['true_label'], data_detected_tda['label']),
#             detection.precision(data_detected_tda['true_label'], data_detected_tda['label'], 'T'),
#             detection.recall(data_detected_tda['true_label'], data_detected_tda['label'], 'T'),
#             detection.accuracy(data_detected_ifum['true_label'], data_detected_ifum['label']),
#             detection.precision(data_detected_ifum['true_label'], data_detected_ifum['label'], 'T'),
#             detection.recall(data_detected_ifum['true_label'], data_detected_ifum['label'], 'T'),
#             detection.accuracy(data_detected_lof['true_label'], data_detected_lof['label']),
#             detection.precision(data_detected_lof['true_label'], data_detected_lof['label'], 'T'),
#             detection.recall(data_detected_lof['true_label'], data_detected_lof['label'], 'T'),
#             detection.accuracy(data_detected_ocsvm['true_label'], data_detected_ocsvm['label']),
#             detection.precision(data_detected_ocsvm['true_label'], data_detected_ocsvm['label'], 'T'),
#             detection.recall(data_detected_ocsvm['true_label'], data_detected_ocsvm['label'], 'T')
#         ])
    
#     # Convert list to numpy array and calculate the average for each column
#     data_array = np.array(list_scores)
#     column_averages = np.mean(data_array, axis=0)
#     dic_score_list[j] = column_averages








    def report(self):
        print("Experiment 1 (TDA) completed. Effectiveness results are available in the output DataFrame.")

   
 
