import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, n_component):
        self.n_component = n_component
    
    def fit(self, X):
        self.X = X
        self.mean_ = X.mean(axis=0)
        self.covariance_ = np.cov(X, rowvar=False)
        
        eig, eigv = np.linalg.eig(self.covariance_)
        idx = eig.argsort()[::-1]
        eig = eig[idx]
        eigv = eigv[:, idx]
        self.explained_variance_ = eig[:self.n_component]
        self.components_ = eigv[:, :self.n_component]   # self.component will be transpose to scikit-learn PCA 
        
        self.explained_variance_ratio_ = (eig / eig.sum())[:self.n_component]
        
    def transform(self, X):
        X = X - self.mean_
        Iv_t = self.components_.T     # eigen vectors are orthogonal, so the inverse is equal to transpose
        transed_x = np.matmul(Iv_t, X.T).T
        return transed_x[:, :self.n_component]
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        Iv = self.components_
        return np.matmul(X, Iv.T) + self.mean_