"""
Evaluation framework for Trust Information Management Framework (TIMF).

This module provides controlled experiments to evaluate TIMF under various
tampering scenarios in a crowdsourced IoT environment.
"""

import pandas as pd
from evaluations.evaluation_data.evaluation_data import EvaluationData
from tampering.tampering import Tampering
from timf.timf import TIMF
from data_service.data_service import DataService


class Evaluations:
    def __init__(self):
        self.evaluation_data = EvaluationData()
        self.tampering = Tampering()
        
        # Initialize DataService and TIMF properly
        self.data_service = DataService()
        self.timf = TIMF(self.data_service)

    def setup_experments(self):
      
        self.data = self.evaluation_data.get_data()
        print("Experment setup is complete.")
        
    def experiment_1(self, tampering_percentages=None, tampering_types=None):
        """
        Run experiment 1: Evaluate TIMF under controlled tampering scenarios.
        
        This experiment:
        - Applies synthetic tampering to data
        - Varies tampering percentage and type
        - Runs trust assessment for each provider in each microcell
        - Records results
        
        Args:
            tampering_percentages: List of percentages to test (default: 10 to 90 in steps of 10)
            tampering_types: List of tampering types ["N", "K", "S"] (default: all)
        
        Returns:
            results: Dictionary with experiment results
        """
        if tampering_percentages is None:
            tampering_percentages = list(range(10, 100, 10))
        if tampering_types is None:
            tampering_types = ["N", "K", "S"]  # Naive, Knowledgeable, Sophisticated
        
        results = []
        
        
        untampered_data = self.data.copy()
        
        for tampering_type in tampering_types:
            for tampering_percentage in tampering_percentages:
                print(f"Experiment: Tampering Type={tampering_type}, Percentage={tampering_percentage}%")
        
             

                # For each provider in each microcell, run trust assessment
                for microcell in self.data['microcell'].unique():
                    df_microcell = self.data[self.data['microcell'] == microcell]
                    # print(f"\nMicrocell: {microcell}")

                    # --- Per-microcell tampering setup ---
                    local_key = str(microcell)

              
                    spa_tampered_dict = self.tampering.spa_tampering(
                        self.data[self.data['microcell'] == microcell],
                        sp_percent=100,              # always tamper the local microcell
                        type=tampering_type
                    )

                    bma_tampered_df = self.tampering.bma_tampering(
                        self.data[self.data['microcell'] != microcell],
                        tampering_percentage,
                        tampering_type
                    )
                   
               
                    # Set tampered and untampered data directly in data service
                    self.data_service.set_tampered_data(spa_tampered_dict,bma_tampered_df)
                

                    # --- Trust assessment for all providers in this local microcell ---
                    for provider in df_microcell['providerid'].unique():
                       
                        trust_score, df = self.timf.trust_assessment(provider, microcell)
                        
                        # Calculate accuracy, precision, and recall by comparing true_label and label
                        
                        metrics = self._calculate_metrics(df['true_label'], df['label'])
                            
                        result = {
                                'tampering_type': tampering_type,
                                'tampering_percentage': tampering_percentage,
                                'microcell': microcell,
                                'provider_id': provider,
                                'trust_score': trust_score,
                                'accuracy': metrics['accuracy'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall']
                            }
                        results.append(result)
                            
                        # print(f"  Provider: {provider} -> Trust score: {trust_score:.4f}, "
                        #           f"Accuracy: {metrics['accuracy']:.4f}, "
                        #           f"Precision: {metrics['precision']:.4f}, "
                        #           f"Recall: {metrics['recall']:.4f}")
                        
        print(results)
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        return results_df
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate accuracy, precision, and recall metrics.
        
        Args:
            y_true: Series with true labels ('C' for Correct, 'T' for Tampered)
            y_pred: Series with predicted labels ('C' for Correct, 'T' for Tampered)
            
        Returns:
            Dictionary with accuracy, precision, and recall
        """
        # Convert to binary: 'T' = 1 (positive), 'C' = 0 (negative)
        y_true_binary = (y_true == 'T').astype(int)
        y_pred_binary = (y_pred == 'T').astype(int)
        
        # Calculate True Positives, False Positives, True Negatives, False Negatives
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        # Calculate metrics
        total = len(y_true)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }




                
    










    def _dictionary_to_merged_df(self,dic):
        temp = pd.concat(dic.values(), ignore_index=True)
        temp.reset_index(drop=True, inplace=True)
        return temp

    def _dataframe_divide_to_microcell_dictionary(self, df):
        """
        Divide a DataFrame into a dictionary keyed by microcell.
        
        Args:
            df: DataFrame with 'microcell' column
            
        Returns:
            Dictionary with microcell as key, DataFrame as value
        """
        temp_dictionary = {}
        unique_keys = df.microcell.unique()
        for microcell in unique_keys:
            temp_dictionary[str(microcell)] = df[df.microcell == microcell].copy()
        return temp_dictionary
      



        # Run experiment 1
       



#
# dfin = general.open_file_csv('data_alg_16000.csv')
# dfin = general.slice_df(dfin,['serviceid','providerid','microcell','timestamp','speed','latency','bandwidth','coverage','reliability','security','currect_microcell'])