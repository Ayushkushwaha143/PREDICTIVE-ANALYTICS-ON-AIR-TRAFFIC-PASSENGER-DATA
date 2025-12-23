#1 Air Traffic Passenger Analysis & Prediction using Python

#2 IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

#3 LOAD DATASET
df = pd.read_csv("Air_Traffic_Passenger_Statistics.csv")
df.head()

#4 DATA PREPROCESSING
# Convert date column
df['Activity Period Start Date'] = pd.to_datetime(df['Activity Period Start Date'])

# Feature extraction
df['Year'] = df['Activity Period Start Date'].dt.year
df['Month'] = df['Activity Period Start Date'].dt.month

# Handle missing values
df.ffill(inplace=True)


#5 EXPLORATORY DATA ANALYSIS (EDA)
#Passenger Trend
df.groupby('Year')['Passenger Count'].sum().plot(kind='line', title='Passenger Trend')
plt.show()
#Correlation
sns.heatmap(df[['Passenger Count','Year','Month']].corr(), annot=True)
plt.show()


#EDA: Passenger Count by GEO Region
plt.figure(figsize=(12, 6))

region_passengers = (
    df.groupby('GEO Region')['Passenger Count']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

sns.barplot(
    data=region_passengers,
    x='Passenger Count',
    y='GEO Region',
    hue='GEO Region',
    palette='viridis',
    legend=False
)

plt.title('Total Passenger Count by GEO Region')
plt.tight_layout()
plt.savefig('passenger_by_region.png')
plt.show()


#EDA: Price Category Distribution
plt.figure(figsize=(8, 6))

df.groupby('Price Category Code')['Passenger Count'].sum().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140
)

plt.title('Passenger Count by Price Category')
plt.ylabel('')
plt.axis('equal')     # makes pie circular
plt.tight_layout()
plt.savefig('price_category_pie.png')
plt.show()


#EDA: Terminal and Activity Type
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x='Terminal',
    y='Passenger Count'
)

plt.yscale('log')  # Log scale for high variance
plt.title('Passenger Count Distribution per Terminal (Log Scale)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('terminal_boxplot.png')
plt.show()




#SUPERVISED LEARNING (REGRESSION)
#6 FEATURE & TARGET SELECTION
X = df[['Year', 'Month']]
y = df['Passenger Count']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#7 MODEL 1: LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

#Evaluation
print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 :", r2_score(y_test, lr_pred))

#8 MODEL 2: DECISION TREE REGRESSION
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

#Evaluation
print("Decision Tree")
print("MAE:", mean_absolute_error(y_test, dt_pred))
print("MSE:", mean_squared_error(y_test, dt_pred))
print("R2 :", r2_score(y_test, dt_pred))


#UNSUPERVISED LEARNING (CLUSTERING)
#9 K-MEANS CLUSTERING
X_cluster = df[['Passenger Count']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

#10 CLUSTER VISUALIZATION
sns.scatterplot(
    x=df.index,
    y=df['Passenger Count'],
    hue=df['Cluster']
)
plt.show()

#11 CLUSTER EVALUATION
silhouette = silhouette_score(X_scaled, df['Cluster'])
print("Silhouette Score:", silhouette)
#(K-Means Clustering (k=3)(reason: good silhouette score))









