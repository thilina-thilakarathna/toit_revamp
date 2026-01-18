import pandas as pd
from sklearn.cluster import DBSCAN
from timf.trust_assessment.trust_assessment import TrustAssessment


class Detection:
    def __init__(self, data_service):
        self.data_service = data_service
        self.trust_assessor = TrustAssessment([0.3,0.1,0.2,0.1,0.1,0.2])

    def detect_tampered_records(self, local_data, remote_data, provider_id, microcell_id):
        local_data = local_data.reset_index(drop=True)
        local_data['label'] = 'T'
        # df_provider = local_data.loc[local_data['providerid'] == provider_id].copy()
        # if df_provider.empty:
        #     print("************* No local data found for provider: ", provider_id)

        # df2_received = df_provider[df_provider['origin'] == 'R'].copy().reset_index(drop=True)
        # df2_generated = df_provider[df_provider['origin'] == 'G'].copy().reset_index(drop=True)

        # # # correct_local = df_provider[df_provider['origin'] == 'G'].copy()
        # # if remote_data is None or remote_data.empty:
        # #     correct_data = correct_local.copy()
        # # else:
        # #     correct_data = pd.concat([correct_local, remote_data.copy()], ignore_index=True)

        # if not df2_received.empty:
        #     for i in range(len(df2_received)):
        #         origin_record = remote_data[remote_data['serviceid'] == df2_received.iloc[i]['serviceid']]
        #         # if origin_record.empty:
        #         #     origin_record = correct_data[correct_data['serviceid'] == df2_received.iloc[i]['serviceid']]

        #         if not origin_record.empty and (
        #             (df2_received.iloc[i]['speed'] == origin_record.iloc[0]['speed']) and
        #             (df2_received.iloc[i]['latency'] == origin_record.iloc[0]['latency']) and
        #             (df2_received.iloc[i]['bandwidth'] == origin_record.iloc[0]['bandwidth']) and
        #             (df2_received.iloc[i]['coverage'] == origin_record.iloc[0]['coverage']) and
        #             (df2_received.iloc[i]['reliability'] == origin_record.iloc[0]['reliability']) and
        #             (df2_received.iloc[i]['security'] == origin_record.iloc[0]['security'])
        #         ):
        #             df2_received.loc[i, 'label'] = 'C'
        #         else:
        #             df2_received.loc[i, 'label'] = 'T'
        # else:
        #     df2_received['label'] = pd.Series(dtype=str)
        #     print("************* No received data found for provider: ", provider_id)

        # if not df2_generated.empty:
        #     df2_generated['label'] = 'S'
        #     # for j in range(len(df2_generated)):
        #     #     provider_other_df1 = correct_data[correct_data['providerid'] == df2_generated.iloc[j]['providerid']]
        #     #     provider_other_df = provider_other_df1[provider_other_df1['microcell'] != microcell_id]
        #     #     provider_other_df = provider_other_df.reset_index(drop=True)

        #     #     if provider_other_df.shape[0] > 1:
        #     #         df_to_clust = provider_other_df[['speed', 'latency', 'bandwidth', 'coverage', 'reliability', 'security']]
        #     #         dbscan = DBSCAN(eps=0.5, min_samples=5)
        #     #         clusters = dbscan.fit_predict(df_to_clust)
        #     #         outliers = provider_other_df.iloc[clusters == -1]
        #     #         check_list = outliers['serviceid'].unique()
        #     #         if df2_generated.iloc[j]['serviceid'] in check_list:
        #     #             df2_generated.loc[j, 'label'] = 'S'
        #     #         else:
        #     #             df2_generated.loc[j, 'label'] = 'NS'
        #     #     else:
        #     #         df2_generated.loc[j, 'label'] = 'NS'

        # # if 'label' not in df2_received.columns:
        # #     df2_received['label'] = 'T'

        # t_count = (df2_received['label'] == 'T').sum()
        # c_count = (df2_received['label'] == 'C').sum()
        # if c_count + t_count != 0:
        #     ratio = t_count / (c_count + t_count)
        #     if ratio < 0.8:
        #         df2_generated['label'].replace('S', 'C', inplace=True)
        #         df2_generated['label'].replace('NS', 'C', inplace=True)
        #     elif ratio > 0.8:
        #         df2_generated['label'].replace('S', 'T', inplace=True)
        #         df2_generated['label'].replace('NS', 'T', inplace=True)
        #     else:
        #         df2_generated['label'].replace('S', 'T', inplace=True)
        #         df2_generated['label'].replace('NS', 'C', inplace=True)
        # else:
        #     df2_generated['label'].replace('S', 'T', inplace=True)
        #     df2_generated['label'].replace('NS', 'C', inplace=True)

        # concatenated_df_provider = pd.concat([df2_generated, df2_received], ignore_index=True)
        # clean_df = concatenated_df_provider[concatenated_df_provider['label'] != 'T'].copy()
        # trust_score = self.trust_assessor.calculate(clean_df) if not clean_df.empty else 0.0

        # return trust_score, concatenated_df_provider
       
        return local_data.copy()
       
