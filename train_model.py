from sklearn.metrics import davies_bouldin_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


def elbow_plot(data, features, max_k=10):
    print("Starting Elbow Method...")
    model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(
        model, k=(2, max_k), metric='distortion', timings=False)
    visualizer.fit(data[features])

    print("Creating Elbow plot...")
    elbow_plot_file = "elbow_plot.png"
    visualizer.show(outpath=elbow_plot_file)
    optimal_k = visualizer.elbow_value_
    plt.close()

    print(f"Elbow Method completed. Optimal K: {optimal_k}")
    return optimal_k


print("Starting script...")

# Load data
csv_path = os.path.join('data', 'DF_TEST_V8.csv')
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Data loaded. Shape: {df.shape}")

# Normalize data
print("Normalizing data...")
scaler = MinMaxScaler()
features = ['normalized_value', 'normalized_PEMAKAIAN_M3']
df[features] = scaler.fit_transform(df[features])

# Perform Elbow Method
optimal_k = elbow_plot(df, features)
print(f"Optimal K from Elbow Method: {optimal_k}")

# Perform K-Means clustering with optimal K
print(f"Performing final K-Means clustering with K={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df[features])
df['cluster'] = kmeans.fit_predict(df[features])

# Calculate Silhouette score
silhouette_avg = silhouette_score(df[features], df['cluster'])
print(f"Silhouette score: {silhouette_avg:.4f}")

# Visualize Silhouette score
print("Creating Silhouette score visualization...")
silhouette_visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
silhouette_visualizer.fit(df[features])
silhouette_visualizer.show(outpath="silhouette_score.png")
print("Silhouette score visualization created successfully.")

# Save model and scaler
print("Saving model and scaler...")
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans, 'kmeans_model.pkl')

print("Model and scaler saved successfully.")
print("Script completed.")
