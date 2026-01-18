import pandas as pd


class DataService:

    def __init__(self):
        self.local_data = None  # DataFrame with all tampered data
        self.remote_data = None  # DataFrame with all untampered data
    
    def set_local_data(self, data):
        self.local_data = data.copy()
    
    def set_remote_data(self, data):
       self.remote_data = data.copy()
   
    
    def get_local_data(self, microcell, provider_id):
        mask = (self.local_data['currect_microcell'] == str(microcell)) & \
               (self.local_data['providerid'] == provider_id)
        return self.local_data[mask].copy()

    def get_remote_data(self, microcell, provider_id):
        mask = (self.remote_data['gen_microcell'] != str(microcell)) & \
               (self.remote_data['providerid'] == provider_id)
        return self.remote_data[mask].copy()
    
   