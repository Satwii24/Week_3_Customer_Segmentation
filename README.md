# Customer Segmentation Using K-Means Clustering

## Week 3 – Unsupervised Learning and Clustering Analysis

### Project Overview

This project demonstrates the application of unsupervised machine learning for customer segmentation using the K-Means clustering algorithm.

The project uses the publicly available Mall Customer Segmentation dataset. The primary objective is to identify meaningful groups of customers based on their annual income and spending score.

Since this is an unsupervised learning problem, the dataset does not contain predefined cluster labels. Instead, K-Means identifies groups of customers according to similarities in the selected features.

---

## Objectives

- Analyze a publicly available customer dataset.
- Perform basic data exploration and preprocessing.
- Select appropriate features for clustering.
- Standardize the selected features.
- Apply the K-Means clustering algorithm.
- Determine a suitable number of clusters using the Elbow Method.
- Evaluate the clustering using Silhouette Score.
- Visualize the identified customer segments.
- Analyze the characteristics of each cluster.
- Discuss potential business applications.

---

## Dataset

The project uses the Mall Customer Segmentation dataset.

### Main Features

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Customer spending score |

For clustering, the following features are primarily used:

- Annual Income (k$)
- Spending Score (1-100)

---

## Methodology

The project follows these steps:

1. Load the dataset using Pandas.
2. Inspect the dataset structure.
3. Check missing values and duplicate records.
4. Select Annual Income and Spending Score.
5. Standardize the selected features using StandardScaler.
6. Apply the Elbow Method for different values of K.
7. Select an appropriate number of clusters.
8. Apply K-Means clustering.
9. Calculate the Sil
