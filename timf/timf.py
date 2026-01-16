
import pandas as pd
from timf.tda.tda import Detection
from timf.trust_assessment.trust_assessment import TrustAssessment


class TIMF:    
    def __init__(self, data_service, weight_matrix=None, tda_config=None):
        self.data_service = data_service
        
        # Initialize TDA (Tampering Detection Approach)
        self.tda = Detection(self.data_service)
        
        # Initialize Trust Assessment with weight matrix
        if weight_matrix is None:
            weight_matrix = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  # Equal weights
        self.trust_assessor = TrustAssessment(weight_matrix)

    def trust_assessment(self, provider_id, microcell_id):
        # Step 1: Retrieve trust information from distributed environment
        local_data = self.data_service.get_local_data(microcell_id, provider_id)
        remote_data = self.data_service.get_remote_data(microcell_id, provider_id)
        
        trust_score,df = self.tda.detect_tampered_records(local_data, provider_id, microcell_id)
        return trust_score,df
        
       