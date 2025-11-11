"""
CS 634 Data Mining - Final Term Project
Option 1: Supervised Data Mining (Classification)

Student Name: Madison Rose Lucas
NJIT UCID: MRL58
Email: MRL58@njit.edu

Algorithm 1: LIBSVM with RBF Kernel (Category 1: Support Vector Machines)
Algorithm 2: Random Forest (Category 2: Random Forests)

Implementation: FROM SCRATCH (using only NumPy for basic operations)
"""
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Algorithm 1: SVM with RBF Kernel (From Scratch)
# ============================================================================

class SVMFromScratch:
    """
    Support Vector Machine with RBF kernel implemented from scratch.
    Uses Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, C=1.0, gamma='scale', max_iter=1000, tol=1e-3):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        
    def _rbf_kernel(self, X1, X2):
        """Compute RBF (Gaussian) kernel between X1 and X2."""
        if self.gamma == 'scale':
            gamma_val = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma_val = self.gamma
            
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma_val * sq_dists)
    
    def _decision_function(self, X):
        """Compute decision function for samples in X."""
        kernel_matrix = self._rbf_kernel(X, self.X_train)
        return np.dot(kernel_matrix, self.alphas * self.y_train) + self.b
    
    def fit(self, X, y):
        """Train SVM using simplified SMO algorithm. For multiclass, uses One-vs-Rest."""
        self.classes_ = np.unique(y)
        
        if len(self.classes_) == 2:
            y_binary = np.where(y == self.classes_[0], -1, 1)
            self._fit_binary(X, y_binary)
            self.binary = True
        else:
            self.binary = False
            self.classifiers = []
            for cls in self.classes_:
                y_binary = np.where(y == cls, 1, -1)
                clf = SVMFromScratch(C=self.C, gamma=self.gamma, 
                                    max_iter=self.max_iter, tol=self.tol)
                clf._fit_binary(X, y_binary)
                self.classifiers.append(clf)
        return self
    
    def _fit_binary(self, X, y):
        """Fit binary SVM using simplified SMO."""
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.alphas = np.zeros(n_samples)
        self.b = 0
        K = self._rbf_kernel(X, X)
        
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alphas)
            
            for i in range(n_samples):
                Ei = self._decision_function(X[i].reshape(1, -1))[0] - y[i]
                
                if (y[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alphas[i] > 0):
                    
                    j = np.random.choice([k for k in range(n_samples) if k != i])
                    Ej = self._decision_function(X[j].reshape(1, -1))[0] - y[j]
                    
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            if np.linalg.norm(self.alphas - alpha_prev) < self.tol:
                break
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.binary:
            decision = self._decision_function(X)
            return np.where(decision >= 0, self.classes_[1], self.classes_[0])
        else:
            decisions = np.array([clf._decision_function(X) for clf in self.classifiers])
            return self.classes_[np.argmax(decisions, axis=0)]
