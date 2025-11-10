from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import silhouette_score
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, flash, send_from_directory
import pandas as pd
import numpy as np
import joblib
import io
import os
import base64
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
        df[features] = scaler.transform(df[features])

        # Predict clusters
        labels = kmeans.predict(df[features])

        # Calculate distances
        distances = [np.linalg.norm(x - kmeans.cluster_centers_[labels[i]])
                     for i, x in enumerate(df[features].values)]

        # Identify anomalies
        anomaly_threshold = np.percentile(distances, 95)
        anomalies = [dist > anomaly_threshold for dist in distances]

        # Add results to dataframe
        df['cluster'] = labels
        df['anomaly'] = ['anomali' if a else 'normal' for a in anomalies]

        # Print cluster summary to terminal
        print(f"Jumlah Cluster (k): {kmeans.n_clusters}")

        # Print cluster summary to terminal
        print_cluster_summary(df)

        # Calculate cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Create 3D visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['yellow', 'green', 'blue', 'purple', 'orange', 'pink']
        for i in range(kmeans.n_clusters):
            cluster_points = df[df['cluster'] == i]
            ax.scatter(cluster_points['normalized_value'], cluster_points['normalized_PEMAKAIAN_M3'],
                       zs=cluster_points.index, c=colors[i], label=f'Cluster {i}')

        # Plot cluster centers (outside the loop)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                   zs=0, c='red', marker='x', s=100, linewidth=2)

        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Normalized PEMAKAIAN_M3')
        ax.set_zlabel('Index')
        ax.set_title('K-Means Clustering')
        ax.legend()

        # Save plot as PNG file
        plt.savefig('clustering.png')

        # Save plot to bytes buffer
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

        # Create 2D visualization
        fig2d = plt.figure(figsize=(10, 8))
        ax2d = fig2d.add_subplot(111)

        colors = ['yellow', 'green', 'blue', 'purple', 'orange', 'pink']
        for i in range(kmeans.n_clusters):
            cluster_points = df[df['cluster'] == i]
            ax2d.scatter(cluster_points['normalized_value'], cluster_points['normalized_PEMAKAIAN_M3'],
                       c=colors[i], label=f'Cluster {i}')

        # Plot cluster centers (outside the loop)
        ax2d.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                   c='red', marker='x', s=100, linewidth=2)

        ax2d.set_xlabel('Normalized Value')
        ax2d.set_ylabel('Normalized PEMAKAIAN_M3')
        ax2d.set_title('K-Means Clustering (2D)')
        ax2d.legend()

        # Save 2D plot as PNG file
        plt.savefig('clustering2d.png')

        # Save plot to bytes buffer
        buf = io.BytesIO()
        FigureCanvas(fig2d).print_png(buf)
        buf.seek(0)
        img_data2d = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig2d)


        # Prepare CSV data
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        anomaly_count = df['anomaly'].value_counts().get('anomali', 0)
        normal_count = df['anomaly'].value_counts().get('normal', 0)

    return jsonify({
        'csv_data': csv_data,
        'img_data': img_data,
        'img_data2d': img_data2d,
        'anomaly_count': int(anomaly_count),
        'normal_count': int(normal_count)
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
        if filtered_df['anomaly'].values[0] == 'normal':
            message = "Tidak ada anomali"
        else:
            message = f"Terdapat {filtered_df['anomaly'].values[0]}. \nDiperlukan pemantauan atau perlu pengecekan terhadap pelanggan."

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
