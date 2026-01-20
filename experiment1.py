import pandas as pd
from evaluations.evaluation_data.evaluation_data import EvaluationData
from tampering.tampering import Tampering
from timf.timf import TIMF
from data_service.data_service import DataService
import numpy as np


evaluation_data = EvaluationData()
tampering = Tampering()
        
data_service = DataService()
timf = TIMF(data_service)

data = evaluation_data.get_data()


tampering_percentages = list(range(10, 100, 10))

tampering_types = ["N", "K", "S"]  # Naive, Knowledgeable, Sophisticated
# tampering_types = ["N"]
        
results = []

def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


results_metrics = []


for tampering_type in tampering_types:
    for tampering_percentage in tampering_percentages:
        print(f"Experiment: Tampering Type={tampering_type}, Percentage={tampering_percentage}%")
        for assessing_mic in data['gen_microcell'].unique():
            df_microcell = data[data['gen_microcell'] == assessing_mic]
            # print(df_microcell)

            bma_tampered_df = tampering.bma_tampering(
                        data[data['gen_microcell'] != assessing_mic].reset_index(drop=True),
                        tampering_percentage,
                        tampering_type
                    )
            
            

            remote_data = bma_tampered_df.copy()

            # Replicate data: for each provider in df_microcell, pull records from 10 nearest microcells
            microcell_coords = data.groupby('gen_microcell')[['latitude', 'longitude']].first().reset_index()
            current_coords = microcell_coords[microcell_coords['gen_microcell'] == assessing_mic]
            if not current_coords.empty:
                lat1 = current_coords['latitude'].values[0]
                lon1 = current_coords['longitude'].values[0]

                df_microcell_part = df_microcell.copy()
                df_microcell_part.loc[:, 'currect_microcell'] = assessing_mic
                
                # df_microcell['currect_microcell'] = assessing_mic
                replicated_parts = [df_microcell_part]
                for provider_id in df_microcell['providerid'].unique():
                    candidate_microcells = []
                    provider_remote = remote_data[remote_data['providerid'] == provider_id]
                    for _, row in microcell_coords.iterrows():
                        if row['gen_microcell'] == assessing_mic:
                            continue
                        if (provider_remote['gen_microcell'] == row['gen_microcell']).any():
                            dist = _haversine_km(lat1, lon1, row['latitude'], row['longitude'])
                            candidate_microcells.append((row['gen_microcell'], dist))

                    if candidate_microcells:
                        candidate_microcells.sort(key=lambda x: x[1])
                        nearby_microcells = [m for m, _ in candidate_microcells[:10]]
                    else:
                        nearby_microcells = []

                    if nearby_microcells:
                        remote_mask = (
                            (remote_data['providerid'] == provider_id) &
                            (remote_data['gen_microcell'].isin(nearby_microcells)) &
                            (remote_data['gen_microcell'] != assessing_mic)
                        )
                        df_remote = remote_data.loc[remote_mask].copy()
                        if not df_remote.empty:
                            df_remote = df_remote.drop_duplicates(subset='serviceid')
                            df_remote['origin'] = 'R'
                            df_remote['currect_microcell'] = assessing_mic
                            replicated_parts.append(df_remote)

                df_microcell_replicated = pd.concat(replicated_parts, ignore_index=True)
            else:
                df_microcell_replicated = df_microcell.copy()

            #df_microcell_replicated contain local generated data, and data gathered from 10 remte microcells per provider now lets assume the lacal super provider is also tamper with the data. 

            spa_tampered_df = tampering.spa_tampering(
                        df_microcell_replicated,
                        type=tampering_type
                    )
            
            # if assessing_mic=='M102':
            #     print(spa_tampered_df['currect_microcell'].unique())

            #     spa_tampered_df.to_csv("test.csv",index=False)
            #     bma_tampered_df.to_csv("test2.csv",index=False)

                

            #     print(df_microcell_replicated['gen_microcell'].unique())
            # else:
            #     continue


            
            data_service.set_local_data(spa_tampered_df.copy())
            data_service.set_remote_data(bma_tampered_df.copy())
            # data_service.set_remote_data(data[data['gen_microcell'] != assessing_mic].copy())
            

            # data service is set with the tamperedcx data which will be received by the timf

            # for each provider lets evaluate the trust score
            for provider in df_microcell['providerid'].unique():
                       
                trust_score, df = timf.trust_assessment(provider, assessing_mic)
                if not df.empty and 'true_label' in df.columns and 'label' in df.columns:
                    y_true = df['true_label']
                    y_pred = df['label']
                        
                        # Binary: T = 1 (tampered), C = 0 (correct)
                    y_true_bin = (y_true == 'T').astype(int)
                    y_pred_bin = (y_pred == 'T').astype(int)
                        
                    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
                    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
                    tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
                    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
                        
                    total = len(y_true_bin)
                    accuracy = (tp + tn) / total if total > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                    try:
                        auc = roc_auc_score(y_true_bin, y_pred_bin)
                    except ValueError:
                        auc = None 
                        

                    results_metrics.append({
                            'tampering_type': tampering_type,
                            'tampering_percentage': tampering_percentage,
                            'microcell': assessing_mic,
                            'provider_id': provider,
                            'method': 'TDA',
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'AUC': auc
                        })
                else:
                    print("issue with df for metrics calculation")

                
                baseline_results = timf.trust_assessment_baseline(provider, assessing_mic)
    

                for method_name, df_baseline in baseline_results.items():
                    if not df_baseline.empty and 'true_label' in df_baseline.columns and 'label' in df_baseline.columns:



                        y_true = (df_baseline['true_label'] == 'T').astype(int)  # 1 = tampered, 0 = correct
                        y_pred = (df_baseline['label'] == 'T').astype(int)

                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)

                        try:
                            auc2 = roc_auc_score(y_true_bin, y_pred_bin)
                        except ValueError:
                            auc2 = None


                        results_metrics.append({
                            'tampering_type': tampering_type,
                            'tampering_percentage': tampering_percentage,
                            'microcell': assessing_mic,
                            'provider_id': provider,
                            'method': method_name,  # specify which baseline
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'AUC':auc2

                        })
                    else:
                        print("issue with df for metrics calculation")


          
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

# Prepare the DataFrame
metrics_df = pd.DataFrame(results_metrics).dropna(
    subset=['accuracy', 'precision', 'recall']
)

# Compute mean metrics
summary = (
    metrics_df
    .groupby(['tampering_type', 'tampering_percentage', 'method'], as_index=False)
    [['accuracy', 'precision', 'recall']]
    .mean()
)

tampering_types = ["N", "K", "S"]
metric_names = ['accuracy', 'precision', 'recall']

# Marker and line-style cycles (color-blind friendly)
markers = cycle(['o', 's', '^', 'D', 'X', 'P', 'v'])
linestyles = cycle(['-', '--', '-.', ':'])

# Loop over each tampering type
for t_type in tampering_types:
    subset_type = summary[summary['tampering_type'] == t_type]
    methods = subset_type['method'].unique()

    # Assign a unique (marker, linestyle) per method
    style_map = {
        method: (next(markers), next(linestyles))
        for method in methods
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle(f'Metrics for Tampering Type {t_type}', fontsize=16)

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        for method in methods:
            subset_method = subset_type[subset_type['method'] == method]
            marker, linestyle = style_map[method]

            ax.plot(
                subset_method['tampering_percentage'],
                subset_method[metric],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                label=method
            )

        ax.set_title(metric.capitalize())
        ax.set_xlabel('Tampering Percentage')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].legend(title='Method', frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
