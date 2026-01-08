"""Data cleaner extracted from `data_cleaning.ipynb`.

Provides `DataCleaner` that reads raw listing CSVs, applies the
transformations in the notebook and returns a cleaned `pandas.DataFrame`.
"""
from typing import List, Optional
import pandas as pd
from components.general_operations import GeneralOp


class DataCleaner:
    """Load and clean listing CSVs into the final dataframe used by experiments.

    Usage:
        cleaner = DataCleaner()
        df = cleaner.get_cleaned_data()

    The class mirrors the main steps in `data_cleaning.ipynb` but is
    defensive about missing files/columns.
    """

    DEFAULT_DIR = "New/Sydney"
    DEFAULT_FILES = ["listings1.csv", "listings2.csv", "listings3.csv", "listings4.csv"]

    def __init__(self, general: Optional[GeneralOp] = None):
        self.general = general or GeneralOp()

    def _read_files(self, file_paths: List[str]) -> pd.DataFrame:
        parts = []
        for p in file_paths:
            try:
                df = self.general.open_file_csv(p)
                parts.append(df)
            except Exception:
                # skip missing or unreadable files
                continue

        if not parts:
            raise FileNotFoundError("No input files could be read from paths: {}".format(file_paths))

        return pd.concat(parts, ignore_index=True)

    def _safe_slice(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        available = [f for f in fields if f in df.columns]
        return df[available]

    def clean(self, data_frame_in: pd.DataFrame, rtis_only: bool = False) -> pd.DataFrame:
        # select relevant columns (fall back to available subset)
        wanted = [
            'id', 'host_id', 'last_review', 'neighbourhood_cleansed', 'latitude', 'longitude',
            'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
            'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
            'review_scores_value'
        ]

        dfin = self._safe_slice(data_frame_in, wanted)

        # mapping as in the notebook
        column_mapping = {
            'id': 'serviceid',
            'host_id': 'providerid',
            'last_review': 'timestamp',
            'neighbourhood_cleansed': 'microcell',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'review_scores_rating': 'speed',
            'review_scores_accuracy': 'throughput',
            'review_scores_cleanliness': 'bandwidth',
            'review_scores_checkin': 'coverage',
            'review_scores_communication': 'reliability',
            'review_scores_location': 'security',
            'review_scores_value': 'latency'
        }

        dfin = dfin.rename(columns={k: v for k, v in column_mapping.items() if k in dfin.columns})

        # drop rows with any NaNs (same as notebook)
        dfin = dfin.dropna()

        # assign serviceid if missing or to ensure sequential unique ids
        dfin['serviceid'] = range(100000, 100000 + len(dfin))

        # normalise microcell/provider ids
        if 'microcell' in dfin.columns:
            unique_microcells = dfin['microcell'].unique()
            microcell_mapping = {name: f"M{102 + i}" for i, name in enumerate(unique_microcells)}
            dfin['microcell'] = dfin['microcell'].map(microcell_mapping)

        if 'providerid' in dfin.columns:
            unique_providers = dfin['providerid'].unique()
            provider_mapping = {name: f"P{1000000 + j}" for j, name in enumerate(unique_providers)}
            dfin['providerid'] = dfin['providerid'].map(provider_mapping)

        # count microcells per provider and filter providers
        if 'providerid' in dfin.columns and 'microcell' in dfin.columns:
            samples_per_provider = dfin.groupby('providerid').size().reset_index(name='sample_count')
            microcells_per_provider = dfin.groupby('providerid')['microcell'].nunique().reset_index(name='microcell_count')
            counts_per_provider = pd.merge(samples_per_provider, microcells_per_provider, on='providerid')

            if rtis_only:
                filtered_counts_per_provider = counts_per_provider[counts_per_provider['microcell_count'] >= 10]
            else:
                filtered_counts_per_provider = counts_per_provider[counts_per_provider['microcell_count'] >= 2]

            selected_provider_ids = filtered_counts_per_provider['providerid']
            filtered_df = dfin[dfin['providerid'].isin(selected_provider_ids)]
        else:
            filtered_df = dfin

        # select representative latitude/longitude per microcell and replace
        if 'microcell' in filtered_df.columns and 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            result = filtered_df.groupby('microcell').first()[['latitude', 'longitude']]
            merged_df = filtered_df.merge(result, on='microcell', suffixes=('', '_selected'))
            merged_df['latitude'] = merged_df['latitude_selected']
            merged_df['longitude'] = merged_df['longitude_selected']
            final_df = merged_df.drop(columns=[c for c in ['latitude_selected', 'longitude_selected'] if c in merged_df.columns])
        else:
            final_df = filtered_df

        final_df['currect_microcell'] = final_df.get('microcell')

        return final_df

    def get_cleaned_data(self, file_paths: Optional[List[str]] = None, rtis_only: bool = False,
                         save_path: Optional[str] = None) -> pd.DataFrame:
        """Read raw files, clean them and optionally save the cleaned CSV.

        If `file_paths` is None the default New/Sydney listings are used.
        """
        if file_paths is None:
            file_paths = [f"{self.DEFAULT_DIR}/{name}" for name in self.DEFAULT_FILES]

        data_frame_in = self._read_files(file_paths)
        cleaned = self.clean(data_frame_in, rtis_only=rtis_only)

        if save_path:
            try:
                self.general.save_file_csv(cleaned, save_path)
            except Exception:
                pass

        return cleaned
