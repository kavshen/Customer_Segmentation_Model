# Customer_Segmentation_Model

This project uses the K-Means clustering algorithm to segment customers based on their annual income and spending score. The aim is to identify different customer segments to better understand their behaviors and tailor marketing strategies accordingly.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer segmentation is the practice of dividing a customer base into groups of individuals that are similar in specific ways relevant to marketing, such as age, gender, interests, and spending habits. This project focuses on clustering customers based on their annual income and spending score using the K-Means clustering algorithm.

## Dataset

The dataset used in this project is the `Mall_Customers.csv` file which contains the following columns:

- `CustomerID`: Unique ID assigned to the customer
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Annual Income (k$)`: Annual income of the customer in thousand dollars
- `Spending Score (1-100)`: Spending score assigned to the customer based on their behavior and spending nature

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/customer-segmentation-model.git
    ```

2. Navigate to the project directory:
    ```bash
    cd customer-segmentation-model
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load the dataset:
    ```python
    import pandas as pd
    df = pd.read_csv('Mall_Customers.csv')
    ```

2. Select relevant features:
    ```python
    X = df.iloc[:, [3, 4]].values
    ```

3. Calculate WCSS (Within-Cluster Sum of Squares) for a range of cluster numbers to determine the optimal number of clusters:
    ```python
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 21):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    ```

4. Plot the elbow method to find the optimal number of clusters:
    ```python
    import matplotlib.pyplot as plt
    plt.plot(range(1, 11), wcss[:10])
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS value')
    plt.show()
    ```

5. Apply K-Means clustering to the dataset with the optimal number of clusters (in this case, 5):
    ```python
    kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
    y_kmeans = kmeansmodel.fit_predict(X)
    ```

6. Visualize the clusters:
    ```python
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=80, c='red', label='Customer 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=80, c='blue', label='Customer 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=80, c='yellow', label='Customer 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=80, c='cyan', label='Customer 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=80, c='black', label='Customer 5')
    plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=100, c='magenta', label='Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    ```

## Results

The resulting plot shows the different clusters of customers based on their annual income and spending score. Each cluster represents a segment of customers with similar behaviors, which can be used to tailor marketing strategies and improve customer satisfaction.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
