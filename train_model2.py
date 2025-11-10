from sklearn.metrics import davies_bouldin_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


def elbow_plot(data, max_k=10):
    print("Starting Elbow Method...")
    distortions = []
    K = range(1, max_k+1)
    for k in K:
        km = TimeSeriesKMeans(n_clusters=k, random_state=42)
        km.fit(data)
        distortions.append(km.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig("elbow_plot.png")
    plt.close()

    optimal_k = K[distortions.index(min(distortions))]
    print(f"Elbow Method completed. Optimal K: {optimal_k}")
    return optimal_k


print("Starting script...")

# Load data
csv_path = os.path.join('data', 'DF_TEST_V8.csv')
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Data loaded. Shape: {df.shape}")

# Convert TAHUN and BULAN to datetime  #masukan app.py
df['datetime'] = pd.to_datetime(df['TAHUN'].astype(
    str) + '-' + df['BULAN'].astype(str).str.zfill(2), format='%Y-%m')

# Normalize data
print("Normalizing data...")
scaler = MinMaxScaler()
cols_to_scale = ['normalized_value', 'normalized_PEMAKAIAN_M3']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Pivot the data #masukan app.py
data_for_clustering = df.pivot_table(
    index='datetime', columns='NO_PLG', values=cols_to_scale).fillna(0)
data_for_clustering = data_for_clustering.T

# Reshape the data masukkan app.py
n_customers = data_for_clustering.shape[0] // 2
n_timesteps = data_for_clustering.shape[1]
data_for_clustering_reshaped = data_for_clustering.values.reshape(
    n_customers, n_timesteps, 2)

# Perform Elbow Method
optimal_k = elbow_plot(data_for_clustering_reshaped)
print(f"Optimal K from Elbow Method: {optimal_k}")

# Perform Time Series K-Means clustering with optimal K
print(f"Performing final Time Series K-Means clustering with K={optimal_k}...")
km = TimeSeriesKMeans(n_clusters=optimal_k,
                      metric="euclidean", random_state=42)
y_pred_km = km.fit_predict(data_for_clustering_reshaped)

# Calculate Silhouette score
silhouette_avg = silhouette_score(data_for_clustering_reshaped.reshape(
    n_customers, -1), y_pred_km, metric="euclidean")
print(f"Silhouette score: {silhouette_avg:.4f}")

# Visualize Silhouette score
print("Creating Silhouette score visualization...")
plt.figure(figsize=(10, 5))
silhouette_visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
silhouette_visualizer.fit(
    data_for_clustering_reshaped.reshape(n_customers, -1))
silhouette_visualizer.show(outpath="silhouette_score.png")
print("Silhouette score visualization created successfully.")

# Save model and scaler
print("Saving model and scaler...")
joblib.dump(km, 'kmeans_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")
print("Script completed.")
