# Import necessary libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Import and view dataset the first few rows of the dataset to understand its structure
file_path = r"C:\Users\relevant_location"
file_name = r"\Onlineretail.csv"
online_sales = pd.read_csv(file_path + file_name,
                           sep=",", encoding="ISO-8859-1", header=0,
                           index_col=None)
print(online_sales.head())

#Check data for missing values and remove if any
print("Number of missing values: \n",online_sales.isna().sum(axis=0))
online_sales = online_sales.dropna(axis=0, how="any")
print("Number of missing values: \n",online_sales.isna().sum(axis=0))

#Remove Duplicates
#Duplicate removal not applicable here

#Data Formatting: Convert CustomerID to string and InvoiceDate to datetime format
print("Original data: \n", online_sales.describe(),"\n")
online_sales["CustomerID"] = online_sales["CustomerID"].astype(str)
online_sales["InvoiceDate"] = pd.to_datetime(online_sales["InvoiceDate"],format='%d-%m-%Y %H:%M')

#Detect and remove outliers
print("Original data: \n", online_sales.describe(),"\n")

#Negative Quantity and Price are not possible we will remove these
online_sales.drop(online_sales[online_sales["UnitPrice"] <=0].index, inplace=True)
online_sales.drop(online_sales[online_sales["Quantity"] <=1].index, inplace=True)
print("Original data: \n", online_sales.describe(),"\n")

#Apply the Z-score method to identify and remove outliers
print("Original data: \n", online_sales.describe(),"\n")
z_upper = online_sales["UnitPrice"].mean() + 3*online_sales["UnitPrice"].std()
z_lower = online_sales["UnitPrice"].mean() - 3*online_sales["UnitPrice"].std()
online_sales_z = online_sales[(online_sales["UnitPrice"] < z_upper) & (
    online_sales["UnitPrice"] > z_lower)]

z_upper = online_sales["Quantity"].mean() + 3*online_sales["Quantity"].std()
z_lower = online_sales["Quantity"].mean() - 3*online_sales["Quantity"].std()
online_sales_z = online_sales_z[(online_sales_z["Quantity"] < z_upper) & (
    online_sales_z["Quantity"] > z_lower)]

print("Outliers removed using z-score: \n", online_sales_z.describe(),"\n")

## Apply the Interquartile Range (IQR) method to further identify and remove outliers for UnitPrice
print("Original data: \n", online_sales.describe(),"\n")
q1 = np.percentile(online_sales["UnitPrice"],25)
q3 = np.percentile(online_sales["UnitPrice"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_iqr = online_sales[(online_sales["UnitPrice"] < upperiqr) & (
    online_sales["UnitPrice"] > lowiqr)]

# Apply IQR for Quantity as well
q1 = np.percentile(online_sales_iqr["Quantity"],25)
q3 = np.percentile(online_sales_iqr["Quantity"],75)
iqr = q3 - q1
lowiqr1 = q1 - 1.5 * iqr
upperiqr1 = q3 + 1.5* iqr
online_sales_iqr = online_sales_iqr[(online_sales_iqr["Quantity"] < upperiqr1) & (
    online_sales_iqr["Quantity"] > lowiqr1)]
print("Outliers removed using IQR: \n", online_sales_iqr.describe(),"\n")
