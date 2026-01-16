import pandas as pd
# from sklearn.cluster import DBSCAN
from timf.trust_assessment.trust_assessment import TrustAssessment


class Detection:
    def __init__(self, data_service):
        self.data_service = data_service
        self.trust_assessor = TrustAssessment([0.3,0.1,0.2,0.1,0.1,0.2])

    def detect_tampered_records(self, local_data, provider_id, microcell_id):
      
        # Label all local data as 'T' (Tampered)
        local_data = local_data.copy()
        local_data['label'] = 'T'

        # print(local_data['origin'].value_counts())

        return self.trust_assessor.calculate(local_data),local_data.copy()
       