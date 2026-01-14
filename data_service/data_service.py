#DataService: Manages tampered data with local/remote separation
# Local data: records from the requested microcell
# Remote data: records from other microcells (simulating data from remote sources)

import pandas as pd


class DataService:
    """
    Manages access to tampered and untampered data with local/remote separation.
    
    Local data represents records from a specific microcell.
    Remote data represents records from other microcells.
    """
    
    def __init__(self):
        self.tampered_data_dict = None  # Dictionary with microcell as key, dataframe as value
        self.untampered_data_dict = None  # Dictionary with microcell as key, dataframe as value
        self.tampered_all_data = None  # Merged tampered dataframe for convenience
        self.untampered_all_data = None  # Merged untampered dataframe for convenience
    
    def set_tampered_data(self, tampered_data_dict):
        """
        Set the tampered data dictionary.
        
        Args:
            tampered_data_dict: Dictionary where keys are microcell names and values are DataFrames
        """
        self.tampered_data_dict = tampered_data_dict
        # Merge all data for easier access
        self.tampered_all_data = pd.concat(tampered_data_dict.values(), ignore_index=True)
        self.tampered_all_data.reset_index(drop=True, inplace=True)
    
    def set_untampered_data(self, untampered_data_dict):
        """
        Set the untampered data dictionary.
        
        Args:
            untampered_data_dict: Dictionary where keys are microcell names and values are DataFrames
        """
        self.untampered_data_dict = untampered_data_dict
        # Merge all data for easier access
        self.untampered_all_data = pd.concat(untampered_data_dict.values(), ignore_index=True)
        self.untampered_all_data.reset_index(drop=True, inplace=True)
    
    def get_local_data(self, microcell):
        """
        Get local data for a specific microcell.
        
        Args:
            microcell: The target microcell identifier
            
        Returns:
            DataFrame containing records from the specified microcell
        """
        if self.tampered_data_dict is None:
            raise ValueError("No tampered data set. Call set_tampered_data() first.")
        
        if str(microcell) not in self.tampered_data_dict:
            return pd.DataFrame()  # Return empty dataframe if microcell not found
        
        return self.tampered_data_dict[str(microcell)].copy()
    
    def get_remote_data(self, microcell):
        """
        Get remote data from all microcells except the specified one.
        
        Args:
            microcell: The local microcell identifier (data to exclude)
            
        Returns:
            DataFrame containing records from all other microcells
        """
        if self.tampered_data_dict is None:
            raise ValueError("No tampered data set. Call set_tampered_data() first.")
        
        remote_dfs = []
        for mc_key, df in self.tampered_data_dict.items():
            if str(mc_key) != str(microcell):
                remote_dfs.append(df)
        
        if not remote_dfs:
            return pd.DataFrame()  # Return empty dataframe if no remote data
        
        remote_data = pd.concat(remote_dfs, ignore_index=True)
        remote_data.reset_index(drop=True, inplace=True)
        return remote_data
    
    def get_all_data(self, data_type='tampered'):
        """
        Get all data (tampered or untampered) merged into a single dataframe.
        
        Args:
            data_type: Either 'tampered' or 'untampered'
        
        Returns:
            DataFrame containing all data of the specified type
        """
        if data_type == 'tampered':
            if self.tampered_all_data is None:
                raise ValueError("No tampered data set. Call set_tampered_data() first.")
            return self.tampered_all_data.copy()
        elif data_type == 'untampered':
            if self.untampered_all_data is None:
                raise ValueError("No untampered data set. Call set_untampered_data() first.")
            return self.untampered_all_data.copy()
        else:
            raise ValueError("data_type must be 'tampered' or 'untampered'")
    
    def get_local_data_by_provider(self, microcell, provider_id):
        """
        Get local data for a specific provider in a microcell.
        
        Args:
            microcell: The target microcell identifier
            provider_id: The provider identifier
            
        Returns:
            DataFrame containing records for the specified provider in the microcell
        """
        local_data = self.get_local_data(microcell)
        if local_data.empty:
            return pd.DataFrame()
        
        return local_data[local_data['providerid'] == provider_id].copy()
    
    def get_remote_data_by_provider(self, microcell, provider_id):
        """
        Get remote data for a specific provider (from other microcells).
        
        Args:
            microcell: The local microcell identifier (data to exclude)
            provider_id: The provider identifier
            
        Returns:
            DataFrame containing records for the specified provider in remote microcells
        """
        remote_data = self.get_remote_data(microcell)
        if remote_data.empty:
            return pd.DataFrame()
        
        return remote_data[remote_data['providerid'] == provider_id].copy()