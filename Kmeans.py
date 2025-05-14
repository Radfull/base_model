import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class Kmeans():
    def __init__(self, n_clusters:int = 3, max_iter:int=300, n_init:int=1, eps:float = 1e-6):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = []
        self.clusters = [[] for _ in range(n_clusters)]
        self.n_init = n_init
        self.errors = np.zeros(n_init)
        self.centers_all = None
        self.eps = eps
    
    def __center_mass(self):
        center_mass = []
        for i in range(len(self.clusters)):
            center_mass_cluter = np.zeros(len(self.centers[0]))
            for point in self.clusters[i]:
                for j in range(len(point)):
                    center_mass_cluter[j] += point[j]
            center_mass_cluter = center_mass_cluter / (len(self.clusters[i])+1)
            center_mass.append(center_mass_cluter)
        return center_mass
    
    def __euclid_dist_count(self,point):
        ind = 0
        min_dist = float('inf')
        for i in range(len(self.centers)):
            dist = sum((point-self.centers[i])**2)
            if dist < min_dist:
                min_dist = dist
                ind = i
        return ind

    def __calc_WCSS(self):
        wcss = 0
        for i in range(len(self.clusters)):
            for point in self.clusters[i]:
                wcss += sum((point - self.centers[i])**2)
        return wcss
    
    def fit(self, X):
        X = np.array(X)
        centers_all = [[] for _ in range(self.n_init)]
        for epoch in range(self.n_init):
            self.centers = np.array([X[np.random.choice(len(X))] for _ in range(self.n_clusters)])
            prev_wcss = float('inf')
            
            for iter in range(self.max_iter):
                self.clusters = [[] for _ in range(self.n_clusters)]
                for point in X:
                    self.clusters[self.__euclid_dist_count(point)].append(point)
                self.centers = self.__center_mass()
                centers_all[epoch].append(self.centers)
                
                current_wcss = self.__calc_WCSS()
                if abs(prev_wcss - current_wcss) < self.eps:
                    # print(iter)
                    break 
                prev_wcss = current_wcss

            wcss = self.__calc_WCSS()
            self.errors[epoch] = wcss
        
        error_min = np.argmin(self.errors)
        self.centers = centers_all[error_min][-1]
        self.centers_all = centers_all[error_min]


    def predict(self,X):
        distances = np.array([[sum((x - c)**2) for c in self.centers] for x in X])
        # print(distances)
        return np.argmin(distances, axis=1)

    def __predict_with_centroid(self,X, centers):
        distances = np.array([[sum((x - c)**2) for c in centers] for x in X])
        # print(distances)
        return np.argmin(distances, axis=1)

    def get_wcss(self):
        return min(self.errors)
    
    def iter_plot(self, X):
        
        plt.ion()
        fig, ax = plt.subplots()

        for i, data in enumerate(self.centers_all):
            ax.clear()  
            plt.scatter(X[:,0], X[:,1], c=self.__predict_with_centroid(X, self.centers_all[i]))
            for j in range(self.n_clusters):
                plt.scatter(data[j][0],data[j][1] , marker="x", color="red", s=50)
            ax.set_title(f'График {i+1}')
            plt.draw()
            plt.pause(0.5) 

        plt.ioff()
        plt.show()  

def plot_scatters(data_to_plot):

    data = data_to_plot['data']
    target = data_to_plot['target']
    fig, ax = plt.subplots(2, 3, figsize = (12,7))
    fig.tight_layout(h_pad=3)

    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    for idx, (i, j) in enumerate(pairs):
        row = idx // 3
        col = idx % 3
        ax[row, col].scatter(data[:, i], data[:, j],c=target )
        ax[row, col].set_title(f'{i+1} и {j+1} признаки')
        ax[row, col].set_xlabel(f'{i+1} признак')
        ax[row, col].set_ylabel(f'{j+1} признак')

    plt.show()

def elbow_method(data, variants):
    wcss = []
    
    for var in variants:
        model = Kmeans(n_clusters=var, max_iter =50, n_init=5)
        model.fit(data)
        wcss.append(model.get_wcss())

    plt.plot(wcss)
    plt.grid()
    plt.xlabel('Итерация')
    plt.ylabel('WCSS')
    plt.title('Elbow method(метод логтя)')
    plt.show()

    mins = []
    min_ind = float('inf')
    for i in range(2,len(wcss)):
        mins.append(abs(wcss[i-1] - wcss[i] + 1e-6) / abs(wcss[i-2] - wcss[i] + 1e-6))

    print(np.argmin(mins) + 1)

    

def main():
    df_iric = load_iris()
    # print(df_iric['data'])
    data = df_iric['data'][:, :2]
    # print(data[:,1])
    # plot_scatters(df_iric)
    # print(data)

    mdl = Kmeans(n_clusters=3, max_iter=50, n_init=5)
    mdl.fit(data)
    pred = mdl.predict(data)
    # mdl.iter_plot(data)
    fig, ax = plt.subplots(1, 2, figsize = (12,7))
    ax[0].scatter(data[:,0], data[:,1], c=df_iric['target'])
    ax[0].set_title(f'True clusters')
    ax[1].scatter(data[:,0], data[:,1], c=pred)
    ax[1].set_title(f'Predict clusters')
    plt.show()

    # elbow_method(data, range(1,21))

if __name__ == "__main__":
    main()