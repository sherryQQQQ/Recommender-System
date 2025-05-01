# Clustering Algorithms Cheatsheet

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Clustering** | Unsupervised learning task of grouping similar data points |
| **Similarity Measure** | Distance function that quantifies how similar two data points are |
| **Centroid** | Center point of a cluster (typically mean of points in the cluster) |
| **Inertia** | Sum of squared distances of samples to their closest centroid |
| **Silhouette Score** | Measure of how similar objects are to their own cluster compared to other clusters (range: -1 to 1) |
| **Curse of Dimensionality** | Phenomena that make clustering difficult in high-dimensional spaces |

## K-Means Clustering

### Algorithm
1. Select K points as initial centroids
2. Assign each data point to nearest centroid
3. Recompute centroids as mean of assigned points
4. Repeat steps 2-3 until convergence

### Mathematical Formulation
$$\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$
- $S = \{S_1, S_2, ..., S_k\}$ are the k clusters
- $\mu_i$ is the centroid of cluster $S_i$

### Properties
- Time Complexity: O(n·k·d·i)
  - n = number of points, k = number of clusters, 
  - d = dimensions, i = iterations
- Space Complexity: O(n + k)

### Advantages & Limitations
✓ Simple, fast implementation  
✓ Works well with large datasets  
✓ Linear complexity in n  
✗ Requires specifying K beforehand  
✗ Sensitive to initial centroids  
✗ Limited to spherical clusters  
✗ Sensitive to outliers

### Initialization Methods
- **Random**: Select K random points as centroids
- **K-Means++**: Select centroids with probability proportional to distance from existing centroids
  ```
  1. Choose first centroid randomly
  2. For each point x, compute D(x) = distance to nearest centroid
  3. Choose next centroid with probability proportional to D(x)²
  4. Repeat until K centroids selected
  ```

### Python Implementation
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
```

### Finding Optimal K
- **Elbow Method**: Plot inertia vs K, look for "elbow" point
- **Silhouette Method**: Choose K that maximizes silhouette score

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### Algorithm
1. For each point, find all points within distance ε (epsilon)
2. Identify core points (points with ≥ MinPts neighbors)
3. Connect core points that are within ε of each other
4. Assign border points to clusters of their neighboring core points
5. Label points not in any cluster as noise

### Parameters
- **ε (eps)**: Maximum distance between two points to be considered neighbors
- **MinPts**: Minimum number of points required to form a dense region

### Point Types
- **Core Point**: Has at least MinPts points within ε
- **Border Point**: Has fewer than MinPts within ε but is neighbor to a core point
- **Noise Point**: Neither a core nor a border point

### Properties
- Time Complexity: O(n²) naive, O(n log n) with spatial indexing
- Space Complexity: O(n)

### Advantages & Limitations
✓ No need to specify number of clusters  
✓ Can find arbitrarily shaped clusters  
✓ Robust to outliers (labeled as noise)  
✓ Can handle clusters of varying densities with careful parameter tuning  
✗ Sensitive to parameters ε and MinPts  
✗ Struggles with varying density clusters  
✗ Higher computational complexity than K-Means

### Python Implementation
```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

## Agglomerative Hierarchical Clustering

### Algorithm
1. Start with each point as its own cluster
2. Compute distances between all pairs of clusters
3. Merge the closest pair of clusters
4. Repeat steps 2-3 until only one cluster remains or a stopping condition is met

### Linkage Methods
- **Single Linkage**: Minimum distance between points in clusters
  $d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$
- **Complete Linkage**: Maximum distance between points in clusters
  $d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$
- **Average Linkage**: Average distance between all pairs of points
  $d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$
- **Ward's Linkage**: Minimizes variance of merged clusters
  $d(C_i, C_j) = \sqrt{\frac{|C_i||C_j|}{|C_i|+|C_j|}} \|c_i - c_j\|$

### Properties
- Time Complexity: O(n³) naive, O(n² log n) optimized
- Space Complexity: O(n²)

### Advantages & Limitations
✓ Produces hierarchical representation (dendrogram)  
✓ No need to specify number of clusters in advance  
✓ Can use various distance metrics and linkage criteria  
✓ Can find clusters of complex shapes with appropriate linkage  
✗ Computationally expensive for large datasets  
✗ Cannot undo previous merges  
✗ Sensitive to outliers

### Python Implementation
```python
from sklearn.cluster import AgglomerativeClustering
agg_cluster = AgglomerativeClustering(n_clusters=k, linkage='ward')
labels = agg_cluster.fit_predict(X)
```

## K-Modes (for Categorical Data)

### Algorithm
1. Select k initial modes
2. Assign each point to the closest mode using Hamming distance
3. Update modes to most frequent category in each cluster
4. Repeat steps 2-3 until convergence

### Hamming Distance
- Count number of positions where categories differ
- Each mismatch = 1, match = 0
- Sum total mismatches

### Properties
- Time Complexity: Similar to K-Means
- Only works with categorical data

### Python Implementation
```python
from kmodes.kmodes import KModes
kmode = KModes(n_clusters=k, init='Huang')
labels = kmode.fit_predict(X_categorical)
```

## Evaluation Metrics

### Internal Metrics (no ground truth)
- **Silhouette Coefficient**:
  $s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$
  - a(i) = average distance to points in same cluster
  - b(i) = average distance to points in nearest different cluster
  - Range: [-1, 1] (higher is better)

- **Dunn Index**:
  $DI = \frac{\min_{i \neq j} d(C_i, C_j)}{\max_k diam(C_k)}$
  - d(Ci, Cj) = distance between clusters i and j
  - diam(Ck) = diameter of cluster k
  - Higher values indicate better clustering

- **Davies-Bouldin Index**:
  $DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$
  - σi = average distance of points in cluster i to centroid
  - d(ci, cj) = distance between centroids
  - Lower values indicate better clustering

- **Calinski-Harabasz Index**:
  $CH = \frac{tr(B_k)/(k-1)}{tr(W_k)/(n-k)}$
  - Bk = between-cluster dispersion matrix
  - Wk = within-cluster dispersion matrix
  - Higher values indicate better clustering

### External Metrics (with ground truth)
- **Adjusted Rand Index**:
  $ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}]/\binom{n}{2}}{\frac{1}{2}[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}] - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}]/\binom{n}{2}}$
  - Range: [-1, 1] (higher is better)

- **Normalized Mutual Information**:
  $NMI(U, V) = \frac{2 \cdot I(U, V)}{H(U) + H(V)}$
  - I(U,V) = mutual information between U and V
  - H(U), H(V) = entropies
  - Range: [0, 1] (higher is better)

## Typical Use Cases

| Algorithm | Best For |
|-----------|----------|
| **K-Means** | • Well-separated, spherical clusters<br>• Large datasets where efficiency is important<br>• When number of clusters is known |
| **DBSCAN** | • Arbitrary-shaped clusters<br>• Datasets with outliers/noise<br>• When number of clusters is unknown<br>• When clusters have similar densities |
| **Agglomerative** | • When hierarchical structure is needed<br>• Small to medium datasets<br>• When cluster visualization is important |
| **K-Modes** | • Datasets with categorical variables<br>• Market segmentation with demographic data |