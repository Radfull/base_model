import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import cluster
from collections import deque

class DBSCAN:
    def __init__(self, eps: float, min_samples: int, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.X = None
        self.labels_ = None
        self.visited = None
        self.neighbors = None
    
    def __euclid_dist(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))
    
    def __get_neighbors(self, point_idx):
        neighbors = []
        for i in range(len(self.X)):
            if i != point_idx and self.__euclid_dist(self.X[point_idx], self.X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit(self, X):
        self.X = np.array(X)
        n_samples = len(self.X)
        self.labels_ = np.full(n_samples, -1)
        self.visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        
        for i in range(n_samples):
            if not self.visited[i]:
                self.visited[i] = True
                neighbors = self.__get_neighbors(i)
                
                if len(neighbors) < self.min_samples:
                    self.labels_[i] = -1
                else:
                    self.__expand_cluster(i, neighbors, cluster_id)
                    cluster_id += 1
        return self
    
    def __expand_cluster(self, point_idx, neighbors, cluster_id):
        queue = deque(neighbors)
        self.labels_[point_idx] = cluster_id
        
        while queue:
            curr_point = queue.popleft()
            
            if not self.visited[curr_point]:
                self.visited[curr_point] = True
                current_neighbors = self.__get_neighbors(curr_point)
                
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            
            if self.labels_[curr_point] == -1:
                self.labels_[curr_point] = cluster_id
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

def plot_scatters(data_to_plot):
    data = data_to_plot['data']
    target = data_to_plot['target']
    fig, ax = plt.subplots(2, 3, figsize=(12,7))
    fig.tight_layout(h_pad=3)

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    for idx, (i, j) in enumerate(pairs):
        row = idx // 3
        col = idx % 3
        ax[row, col].scatter(data[:, i], data[:, j], c=target)
        ax[row, col].set_title(f'{i+1} и {j+1} признаки')
        ax[row, col].set_xlabel(f'{i+1} признак')
        ax[row, col].set_ylabel(f'{j+1} признак')

    plt.show()
        
def main():
    df = load_iris()
    X = df['data'][:,2:]
    
    our_model = DBSCAN(eps=0.5, min_samples=5)
    our_pred = our_model.fit_predict(X)
    
    # plot_scatters(df)
    sklearn_model = cluster.DBSCAN(eps=0.5, min_samples=5)
    sklearn_pred = sklearn_model.fit_predict(X)
    
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].scatter(X[:,0], X[:,1], c=df['target'])
    ax[0].set_title('True')
    ax[1].scatter(X[:,0], X[:,1], c=our_pred)
    ax[1].set_title('My DBSCAN')
    ax[2].scatter(X[:,0], X[:,1], c=sklearn_pred)
    ax[2].set_title('Sklearn DBSCAN')
    plt.show()

if __name__ == "__main__":
    main()