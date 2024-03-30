# Assuming the steps in Feature_engineering were followed 

#Using z-score to standardise data
online_sales_rfm_df = online_sales_rfm[["Recency","Frequency","Monetary"]]
scaler = StandardScaler()
online_sales_rfm_scaled = scaler.fit_transform(online_sales_rfm_df)
online_sales_rfm_scaled = pd.DataFrame(online_sales_rfm_scaled)
online_sales_rfm_scaled.columns = ["Recency","Frequency","Monetary"]
print("Standardised RFM Table: \n",online_sales_rfm_scaled.head())

#Creating K-Means Clustering algorithm
#Finding n clusters - Elbow graph - Determining the optimal number of clusters using the elbow method
cluster_range = [*range(2,11)]
lst = []
for i in cluster_range:
    kmeans = KMeans(n_clusters=i,max_iter=50)
    kmeans.fit(online_sales_rfm_scaled)
    lst.append(kmeans.inertia_)

plt.plot(lst,marker="o",linestyle=":",c="black")
plt.title("Elbow Method",fontweight="bold")
plt.xlabel("Number of Clusters",fontweight="bold")
plt.ylabel("Inertia", fontweight="bold")
plt.show()

#K-Means algorithm
kmeans = KMeans(n_clusters=3,max_iter=50)
kmeans.fit(online_sales_rfm_scaled) 

online_sales_rfm["ClusterID"] = kmeans.labels_
print("RFM Table with Cluster ID: \n", online_sales_rfm.head())
online_sales_rfm_scaled["ClusterID"] = kmeans.labels_

#3d Graph to visualise clustering
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(online_sales_rfm_scaled.iloc[:, 0],
                     online_sales_rfm_scaled.iloc[:, 1],
                     online_sales_rfm_scaled.iloc[:, 2],
                     c=kmeans.labels_, cmap='viridis', s=50)

ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           kmeans.cluster_centers_[:, 2],
           s=200, marker='X', c='red')

ax.set_xlabel('Recency',fontweight="bold")
ax.set_ylabel('Frequency',fontweight="bold")
ax.set_zlabel('Monetary',fontweight="bold")
plt.colorbar(scatter, ax=ax, label='Clusters')
plt.title('K-Means Clustering Results in 3D',fontweight="bold")
plt.show()

#Understanding clusters with boxplots
sns.boxplot(data=online_sales_rfm, x="ClusterID", y="Recency")
plt.title("Clusters by Recency",fontweight="bold")
plt.xlabel("Clusters",fontweight="bold")
plt.ylabel("Recency (days)",fontweight="bold")
plt.show()

sns.boxplot(data=online_sales_rfm, x="ClusterID", y="Frequency")
plt.title("Clusters by Frequency",fontweight="bold")
plt.xlabel("Clusters",fontweight="bold")
plt.ylabel("Frequency",fontweight="bold")
plt.show()

sns.boxplot(data=online_sales_rfm, x="ClusterID", y="Monetary")
plt.title("Clusters by Monetary",fontweight="bold")
plt.xlabel("Clusters",fontweight="bold")
plt.ylabel("Monetary ($)",fontweight="bold")
plt.show()