from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import silhouette_score
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash, send_from_directory
import pandas as pd
import numpy as np
import joblib
import io
import os
import base64
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib  # Import matplotlib first
from tabulate import tabulate
from sklearn.metrics import silhouette_score
matplotlib.use('Agg')  # Then set the backend to 'Agg'
app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Load model dan scaler
kmeans = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')


@app.route('/', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin':
            return redirect(url_for('index'))
        elif username == 'user' and password == 'user':
            return redirect(url_for('indexuser'))
        else:
            flash('Invalid username or password')
            return render_template('login.html')
    return render_template('login.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/indexuser')
def indexuser():
    return render_template('indexuser.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login_page'))


def print_cluster_summary(df):
    # Hitung jumlah data anomali dan normal di setiap cluster
    cluster_counts = df.groupby('cluster')['anomaly'].value_counts().unstack()

    # Menyiapkan tabel
    table = []
    for index, row in cluster_counts.iterrows():
        total = row['anomali'] + row['normal']
        table.append([index, row['anomali'], row['normal'], total])

    # Mencetak tabel model output
    print("\nCluster Summary:")
    print(tabulate(table, headers=[
          'Cluster', 'Anomali', 'Normal', 'Total'], tablefmt='grid'))


@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read CSV file
        df = pd.read_csv(file)

        # Normalize data
        features = ['normalized_value', 'normalized_PEMAKAIAN_M3']
        df[features] = scaler.fit_transform(df[features])

        # Convert TAHUN and BULAN to datetime
        df['datetime'] = pd.to_datetime(df['TAHUN'].astype(
            str) + '-' + df['BULAN'].astype(str).str.zfill(2), format='%Y-%m')

        # Pivot the data
        data_for_clustering = df.pivot_table(index='datetime', columns='NO_PLG', values=[
                                             'normalized_PEMAKAIAN_M3', 'normalized_value']).fillna(0)

        # Transpose the data to have time series for each customer (rows represent customers, columns represent time)
        data_for_clustering = data_for_clustering.T

        # Reshape the data
        # Assuming two features: normalized_PEMAKAIAN_M3 and normalized_value
        n_customers = data_for_clustering.shape[0] // 2
        n_timesteps = data_for_clustering.shape[1]
        data_for_clustering_reshaped = data_for_clustering.values.reshape(
            n_customers, n_timesteps, 2)

        # Print cluster summary to terminal
        print(f"Jumlah Cluster (k): {kmeans.n_clusters}")

        # Define Time Series K-Means parameters
        n_clusters = kmeans.n_clusters
        seed = 0

        # Time Series K-Means
        km = TimeSeriesKMeans(
            n_clusters=n_clusters, metric="euclidean", verbose=False, random_state=seed)
        y_pred_km = km.fit_predict(data_for_clustering_reshaped)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data_for_clustering_reshaped.reshape(
            n_customers, -1), y_pred_km, metric="euclidean")

        # Print silhouette score
        print("Time Series K-Means silhouette: {:.4f}".format(silhouette_score(
            data_for_clustering_reshaped.reshape(n_customers, -1), y_pred_km, metric="euclidean")))

        # Plot clusters
        plt.figure(figsize=(18, 9))
        for yi in range(n_clusters):
            plt.subplot(3, n_clusters, yi + 1)
            cluster_data = data_for_clustering_reshaped[y_pred_km == yi]
            for xx in cluster_data:
                # Plotting the first feature (e.g., normalized_PEMAKAIAN_M3)
                plt.plot(xx[:, 0].ravel(), "k-", alpha=.2)
            # Plotting the first feature of the cluster center
            plt.plot(km.cluster_centers_[yi][:, 0].ravel(), "r-")
            plt.xlim(0, n_timesteps)
            plt.ylim(0, 1)
            plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                     transform=plt.gca().transAxes)
            plt.gca().grid(False)
            if yi == 1:
                plt.title("Time Series $k$-means")

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        img_data1 = base64.b64encode(buf.getvalue()).decode()
        plt.savefig('plot_clusters.jpg')
        plt.close()

        # CLUSTERING LABEL
        # Create mapping of NO_PLG to clusters
        no_plg_to_cluster = pd.DataFrame({
            'NO_PLG': data_for_clustering.index.levels[1],
            'Cluster': y_pred_km
        })
        # Merge cluster information back to original dataframe
        df = df.merge(no_plg_to_cluster, on='NO_PLG', how='left')

        # DROP DATA
        df = df[~((df['Cluster'] == 0) &
                  ((df['normalized_value'] > 0.5) |
                   (df['normalized_PEMAKAIAN_M3'] > 0.5)))]

        # Plot data for each cluster
        n_clusters = len(np.unique(y_pred_km))
        plt.figure(figsize=(15, 5 * n_clusters))

        for cluster in range(n_clusters):
            plt.subplot(n_clusters, 1, cluster + 1)
            cluster_data = df[df['Cluster'] == cluster]

            for _, group in cluster_data.groupby('NO_PLG'):
                plt.plot(
                    group['datetime'], group['normalized_PEMAKAIAN_M3'], color='blue', alpha=0.5)
                plt.plot(group['datetime'],
                         group['normalized_value'], color='red', alpha=0.5)

            plt.title(f'Cluster {cluster}')
            plt.xlabel('Time')
            plt.ylabel('Scaled Usage / Pressure')
            plt.legend(['Customer Water Usage', 'Sensor Pressure'])
            plt.grid(False)
            plt.ylim(0, 1)  # Set y-axis limits to range between 0 and 1

        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        img_data2 = base64.b64encode(buf.getvalue()).decode()
        plt.savefig('plot_each_clusters.jpg')
        plt.close()

        # ANOMALI LABEL
        # Label anomalies (assuming cluster 0 is anomalous)
        df['kmeans_time_series_label'] = 'Normal'
        df.loc[df['Cluster'] == 0, 'kmeans_time_series_label'] = 'Anomali'

        # Prepare CSV data
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Count anomalies and normal data
        anomaly_count = df['kmeans_time_series_label'].value_counts().get(
            'Anomali', 0)
        normal_count = df['kmeans_time_series_label'].value_counts().get(
            'Normal', 0)

        dftf_new = df.copy()

        # Calculate descriptive statistics for normalized_PEMAKAIAN_M3
        grouped_pemakaian = dftf_new.groupby('Cluster').agg({
            'normalized_PEMAKAIAN_M3': ['count', 'mean', 'min', 'max'],
        }).reset_index()
        grouped_pemakaian.columns = ['Cluster', 'Count', 'Mean', 'Min', 'Max']
        labels_pemakaian = dftf_new.groupby(
            'Cluster')['kmeans_time_series_label'].first().reset_index()
        grouped_pemakaian = grouped_pemakaian.merge(
            labels_pemakaian, on='Cluster')

        print("Descriptive Statistics for normalized_PEMAKAIAN_M3:")
        print(tabulate(grouped_pemakaian, headers='keys', tablefmt='psql'))

        # Calculate descriptive statistics for normalized_value
        grouped_value = dftf_new.groupby('Cluster').agg({
            'normalized_value': ['count', 'mean', 'min', 'max'],
        }).reset_index()
        grouped_value.columns = ['Cluster', 'Count', 'Mean', 'Min', 'Max']
        labels_value = dftf_new.groupby(
            'Cluster')['kmeans_time_series_label'].first().reset_index()
        grouped_value = grouped_value.merge(labels_value, on='Cluster')

        print("Descriptive Statistics for normalized_value:")
        print(tabulate(grouped_value, headers='keys', tablefmt='psql'))

        return jsonify({
            'csv_data': csv_data,
            'img_data1': img_data1,
            'img_data2': img_data2,
            'anomaly_count': int(anomaly_count),
            'normal_count': int(normal_count),
            'silhouette_score': silhouette_avg
        })


@app.route('/anomaly_detection_user', methods=['POST'])
def anomaly_detection_user():
    # data = request.json
    no_pelanggan = request.json['no_pelanggan']
    device_id = request.json['device_id']

    # Load data
    df = pd.read_csv('data/anomaly_detection.csv')

    # Filter the dataframe for the specific NO_PLG and Device Id
    filtered_df = df[(df['NO_PLG'] == int(no_pelanggan))
                     & (df['Device Id'] == device_id)]

    if filtered_df.empty:
        return jsonify({'message': 'Data tidak ditemukan untuk nomor pelanggan dan device ID yang diberikan.'})
    else:
        if filtered_df['kmeans_time_series_label'].values[0] == 'Normal':
            message = "Tidak ada anomali"
        else:
            message = f"Terdapat {filtered_df['kmeans_time_series_label'].values[0]}. \nDiperlukan pemantauan atau perlu pengecekan terhadap pelanggan."

        # Get latitude and longitude from the filtered dataframe
        latitude = filtered_df['POS_LAT'].values[0]
        longitude = filtered_df['POS_LONG'].values[0]

        return jsonify({
            'message': message,
            'latitude': latitude,
            'longitude': longitude
        })


if __name__ == '__main__':
    app.run(debug=True)
