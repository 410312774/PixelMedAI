#计算亚区域轮廓系数的核心代码

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# 生成示例数据
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.0, random_state=42)

# 用KMeans进行聚类
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# 计算轮廓系数
silhouette_avg = silhouette_score(X, labels)
print(f"轮廓系数为: {silhouette_avg}")

# 如果你想查看每个样本点的轮廓系数，可以使用silhouette_samples
from sklearn.metrics import silhouette_samples

sample_silhouette_values = silhouette_samples(X, labels)
print(f"前十个样本点的轮廓系数: {sample_silhouette_values[:10]}")
