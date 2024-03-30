# Assuming the steps in Data_import_&_cleaning were followed 

#Feature Engineering - create a new feature 'TotalCost' which is the product of Quantity and UnitPrice
online_sales["TotalCost"] = online_sales["Quantity"]*online_sales["UnitPrice"]

#Irrelevant Features - Visualize the correlation between Quantity, UnitPrice, and TotalCost using a heatmap
correlation_matrix = online_sales[["Quantity","UnitPrice","TotalCost"]].corr()
sns.heatmap(correlation_matrix, vmin=0.0, vmax=1.0, annot=True)
plt.title("Correlation Matrix",fontweight="bold")
plt.show()
#We can use new TotalCost feature rather than Quantity and UnitPrice

#Feature Engineering
#RFM analysis - Recency, Calculating how many days ago each customer made their last purchase
most_recent_purchase = max(online_sales["InvoiceDate"])
online_sales["TimeDiff"] = most_recent_purchase - online_sales["InvoiceDate"]
online_sales_rfm_r = online_sales.groupby("CustomerID")["TimeDiff"].min()
online_sales_rfm_r.head()
#Above is using customer ID as index - needs to be reset with below
online_sales_rfm_r = online_sales_rfm_r.reset_index()
#Time difference is showing hours and miutes, this much detail is not needed
online_sales_rfm_r["TimeDiff"] = online_sales_rfm_r["TimeDiff"].dt.days 
online_sales_rfm_r.head()

#RFM analysis - Frequency, Count how many times each customer made a purchase
online_sales_rfm_f = online_sales.groupby("CustomerID")["InvoiceNo"].count()
online_sales_rfm_f.head()
#Above is using customer ID as index - needs to be reset with below
online_sales_rfm_f = online_sales_rfm_f.reset_index()
online_sales_rfm_f.head()

#RFM analysis - Monetary - Calculating and grouping customers by their total spend
online_sales_rfm_m = online_sales.groupby("CustomerID")["TotalCost"].sum()
online_sales_rfm_m.head()
#Above is using customer ID as index - needs to be reset with below
online_sales_rfm_m = online_sales_rfm_m.reset_index()
online_sales_rfm_m.head()

#RFM - Creating dataframe for rfm analysis by combining Recency, Frequency, and Monetary into one DataFrame
online_sales_rfm = pd.merge(online_sales_rfm_r,
                            online_sales_rfm_f,
                            on="CustomerID",
                            how="inner")
online_sales_rfm = pd.merge(online_sales_rfm,
                            online_sales_rfm_m, 
                            on="CustomerID",
                            how="inner")
online_sales_rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
print("RFM Table: \n",online_sales_rfm.head())

#Visulaisation of outlier distribution using boxplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
axes[0].boxplot(online_sales_rfm["Recency"], labels=["Recency"])
axes[0].set_title("Recency")
axes[1].boxplot(online_sales_rfm["Frequency"], labels=["Frequency"])
axes[1].set_title("Frequency")
axes[2].boxplot(online_sales_rfm["Monetary"], labels=["Monetary"])
axes[2].set_title("Monetary")
for ax in axes:
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Range", fontweight="bold")
fig.suptitle("Outlier Distribution", fontweight="bold")
plt.show()

#Removing outliers using IQR
original_len = len(online_sales_rfm)
#Rfm Recency
q1 = np.percentile(online_sales_rfm["Recency"],25)
q3 = np.percentile(online_sales_rfm["Recency"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_rfm = online_sales_rfm[(online_sales_rfm["Recency"] < upperiqr) & (
    online_sales_rfm["Recency"] > lowiqr)]


q1 = np.percentile(online_sales_rfm["Frequency"],25)
q3 = np.percentile(online_sales_rfm["Frequency"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_rfm = online_sales_rfm[(online_sales_rfm["Frequency"] < upperiqr) & (
    online_sales_rfm["Frequency"] > lowiqr)]

q1 = np.percentile(online_sales_rfm["Monetary"],25)
q3 = np.percentile(online_sales_rfm["Monetary"],75)
iqr = q3 - q1
lowiqr = q1 - 1.5 * iqr
upperiqr = q3 + 1.5* iqr
online_sales_rfm = online_sales_rfm[(online_sales_rfm["Monetary"] < upperiqr) & (
    online_sales_rfm["Monetary"] > lowiqr)]
online_sales_rfm["CustomerID"] = online_sales_rfm["CustomerID"].astype(str)
print("Outliers removed from RFM table using IQR: ", original_len - len(online_sales_rfm))
#Resetting and removing current index
online_sales_rfm = online_sales_rfm.reset_index(drop=True)