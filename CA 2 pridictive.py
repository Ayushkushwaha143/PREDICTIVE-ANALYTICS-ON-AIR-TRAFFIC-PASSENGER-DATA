# PREDICTIVE ANALYTICS PROJECT (COMPLETE)
# Dataset: Air_Traffic_Passenger_Statistics.csv

import pandas as pd

# Load the dataset
df = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')

# Inspect the first few rows
print(df.head())

# Get information about columns and types
print(df.info())

# Summary statistics
print(df.describe())



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert 'Activity Period Start Date' to datetime
df['Activity Period Start Date'] = pd.to_datetime(df['Activity Period Start Date'])
df['Year'] = df['Activity Period Start Date'].dt.year
df['Month'] = df['Activity Period Start Date'].dt.month

# 1. EDA: Passenger Count over time (Yearly)
yearly_passengers = df.groupby('Year')['Passenger Count'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_passengers, x='Year', y='Passenger Count', marker='o')
plt.title('Total Passenger Count Over Years')
plt.grid(True)
plt.savefig('passenger_over_years.png')

# 2. EDA: Passenger Count by GEO Region
plt.figure(figsize=(12, 6))
region_passengers = df.groupby('GEO Region')['Passenger Count'].sum().sort_values(ascending=False).reset_index()
sns.barplot(data=region_passengers, x='Passenger Count', y='GEO Region', palette='viridis')
plt.title('Total Passenger Count by GEO Region')
plt.savefig('passenger_by_region.png')

# 3. EDA: Price Category Distribution
plt.figure(figsize=(8, 6))
df.groupby('Price Category Code')['Passenger Count'].sum().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Passenger Count by Price Category')
plt.ylabel('')
plt.savefig('price_category_pie.png')

# 4. EDA: Terminal and Activity Type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Terminal', y='Passenger Count')
plt.yscale('log') # Use log scale because of high variance
plt.title('Passenger Count Distribution per Terminal (Log Scale)')
plt.savefig('terminal_boxplot.png')

# Prepare for modeling: Select features and encode
features = ['Year', 'Month', 'GEO Summary', 'GEO Region', 'Activity Type Code', 'Price Category Code', 'Terminal', 'Boarding Area']
X = df[features].copy()
y = df['Passenger Count']

# Encoding categorical features
X_encoded = pd.get_dummies(X, columns=['GEO Summary', 'GEO Region', 'Activity Type Code', 'Price Category Code', 'Terminal', 'Boarding Area'], drop_first=True)

# Save preprocessed data for review or further use
# X_encoded['Passenger Count'] = y
# X_encoded.to_csv('preprocessed_data.csv', index=False)

print("EDA completed and plots saved.")
print(f"Feature set shape: {X_encoded.shape}")



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- Supervised Learning ---
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)

# 2. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) # Reduced estimators for speed
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"Linear Regression: R2={lr_r2:.4f}, MAE={lr_mae:.2f}")
print(f"Random Forest: R2={rf_r2:.4f}, MAE={rf_mae:.2f}")

# --- Unsupervised Learning ---
# Sample data for silhouette score as it's computationally expensive
X_scaled = StandardScaler().fit_transform(X_encoded)
X_sample = X_scaled[np.random.choice(X_scaled.shape[0], 5000, replace=False)]

# 1. K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)
clusters = kmeans.labels_
sil_score = silhouette_score(X_sample, kmeans.predict(X_sample))

print(f"K-Means Silhouette Score (Sample): {sil_score:.4f}")

# Comparison Table
supervised_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R-squared': [lr_r2, rf_r2],
    'MAE': [lr_mae, rf_mae]
})

unsupervised_results = pd.DataFrame({
    'Model': ['K-Means (k=4)'],
    'Silhouette Score': [sil_score]
})

supervised_results.to_csv('supervised_comparison.csv', index=False)
unsupervised_results.to_csv('unsupervised_comparison.csv', index=False)



# Visualize Supervised Model Performance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='R-squared', data=supervised_results)
plt.title('R-squared Comparison')

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='MAE', data=supervised_results)
plt.title('MAE Comparison (Lower is Better)')
plt.tight_layout()
plt.savefig('model_comparison.png')

# Cluster Analysis for Unsupervised Report
df['Cluster'] = clusters
cluster_summary = df.groupby('Cluster')['Passenger Count'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='Cluster', y='Passenger Count', data=cluster_summary)
plt.title('Average Passenger Count per Cluster')
plt.savefig('cluster_analysis.png')


