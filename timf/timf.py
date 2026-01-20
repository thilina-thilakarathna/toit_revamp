
import pandas as pd
from timf.tda.baselines import BaselineTda
from timf.tda.tda import Detection
from timf.trust_assessment.trust_assessment import TrustAssessment


class TIMF:    
    def __init__(self, data_service, weight_matrix=None, tda_config=None):
        self.data_service = data_service
        
        # Initialize TDA (Tampering Detection Approach)
        self.tda = Detection(self.data_service)
        self.tda_baseline = BaselineTda()
        self.trust_assessor = TrustAssessment([0.3,0.1,0.2,0.1,0.1,0.2])

    def trust_assessment(self, provider_id, microcell_id):
        # Step 1: Retrieve trust information from distributed environment
        local_data = self.data_service.get_local_data(microcell_id, provider_id)
        remote_data = self.data_service.get_remote_data(microcell_id, provider_id)
        
        df,time_dir = self.tda.detect_tampered_records(local_data,remote_data, provider_id, microcell_id)
        return self.trust_assessor.calculate(df[df['label']=='C']),df
    
    def trust_assessment_baseline(self, provider_id, microcell_id):
        local_data = self.data_service.get_local_data(microcell_id, provider_id)
        remote_data = self.data_service.get_remote_data(microcell_id, provider_id)
        df_dir = self.tda_baseline.baseline_detection(local_data)

        return df_dir
        
       