from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

app = Flask(__name__)

# Load the data
df = pd.read_csv('customerdata.csv')

# Preprocess data (same as previous code)
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
df_scaled = pd.DataFrame(df_scaled, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
df_processed = pd.concat([df_scaled, df['Gender_Male']], axis=1)

# Define optimal number of clusters
optimal_k = 3

# KMeans clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df_processed)

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title(f'KMeans Clustering (k={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

# Save the plot to a BytesIO object
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_png = buffer.getvalue()
buffer.close()

# Convert the image to base64
graph = base64.b64encode(image_png).decode('utf-8')
plt.close()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', graph=graph)

if __name__ == '__main__':
    app.run(debug=True)
