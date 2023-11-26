import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Perform LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Plot the result
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], color=color,
                alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()

# Let's do it manually
# 1. Compute the d-dimensional mean vectors for the different classes from the dataset.
# 2. Compute the scatter matrices (in-between-class and within-class scatter matrix).
# 3. Compute the eigenvectors (ee1,ee2,...,eed) and corresponding eigenvalues (λλ1,λλ2,...,λλd) for the scatter matrices.
# 4. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k
#   dimensional matrix WW(where every column represents an eigenvector).
# 5. Use this d×k eigenvector matrix to transform the samples onto the new subspace. This can be summarized by the matrix
#   multiplication: YY=X×W (where XX is a n×d-dimensional matrix representing the n samples, and yy are the transformed n×k-dimensional
#   samples in the new subspace).


# 1. Compute the d-dimensional mean vectors for the different classes from the dataset.
np.set_printoptions(precision=4)
mean_vectors = []
for cl in range(0, 3):
    mean_vectors.append(np.mean(X[y == cl], axis=0))
    print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl]))

# 2. Compute the scatter matrices (in-between-class and within-class scatter matrix).
S_W = np.zeros((4, 4))
for cl, mv in zip(range(0, 3), mean_vectors):
    class_sc_mat = np.zeros((4, 4))
    for row in X[y == cl]:
        row, mv = row.reshape(4, 1), mv.reshape(4, 1)
        class_sc_mat += (row - mv).dot((row - mv).T)
    S_W += class_sc_mat
print('within-class Scatter Matrix:\n', S_W)

overall_mean = np.mean(X, axis=0)

S_B = np.zeros((4, 4))

for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i, :].shape[0]
    mean_vec = mean_vec.reshape(4, 1)
    overall_mean = overall_mean.reshape(4, 1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('between-class Scatter Matrix:\n', S_B)

# 3. Compute the eigenvectors (ee1,ee2,...,eed) and corresponding eigenvalues (λλ1,λλ2,...,λλd) for the scatter matrices.
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# 4. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k
# dimensional matrix WW(where every column represents an eigenvector).
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
             for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# 5. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
# This can be summarized by the matrix multiplication: YY=X×W (where XX is a n×d-dimensional matrix representing the n samples,
# and yy are the transformed n×k-dimensional samples in the new subspace).
W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1)))
print('Matrix W:\n', W.real)

X_lda = X.dot(W)
assert X_lda.shape == (150, 2), "The matrix is not 150x2 dimensional."

# Plot the result
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color,
                alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()

# Quality of the projections
quality = np.linalg.inv(S_W).dot(S_B)
print('Quality of the projections:\n', quality)

# Explained variance
tot = sum(eig_vals.real)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals.real, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('Explained variance:\n', var_exp)
print('Cumulative explained variance:\n', cum_var_exp)

# Implementing the decision funtion using previous results
decision = X.dot(quality)
print('Decision function:\n', decision)
