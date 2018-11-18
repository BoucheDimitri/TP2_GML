import numpy as np
import scipy
import os
import helper


def one_hot_target(Y):
    num_samples = Y.shape[0]
    classes = np.unique(Y)
    Yoh = np.zeros((num_samples, classes.shape[0]), dtype=int)
    count = 0
    for c in classes:
        Yoh[:, count] = (Y == c).astype(int)
        count += 1
    return Yoh


def mask_labels(Y, l):
    # Draw at random index of labels
    num_samples = Y.shape[0]
    l_idx = np.random.choice(range(num_samples), l, replace=False)

    # mask labels
    Y_masked = np.zeros((num_samples, ), dtype = int)
    Y_masked[l_idx] = Y[l_idx]
    return Y_masked


def hard_hfs(X, Y, laplacian_regularization=0.000001, var=1, eps=0, k=10, laplacian_normalization=""):
#  a skeleton function to perform hard (constrained) HFS,
#  needs to be completed
#
#  Input
#  X:
#      (n x m) matrix of m-dimensional samples
#  Y:
#      (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
#
#  Output
#  labels:
#      class assignments for each (n) nodes

    num_samples = np.size(X, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    # Indices of labelled and unlabelled data
    l_idx = np.argwhere(Y != 0)[:, 0]
    u_idx = np.argwhere(Y == 0)[:, 0]
    
    # Build similarity graph and Laplacian
    W = helper.build_similarity_graph(X, var, eps, k)
    L = helper.build_laplacian(W, laplacian_normalization)
    
    # Extract blocks corresponding to unlabelled and labelled data
    Luu = L[u_idx, :][:, u_idx]
    Wul = W[u_idx, :][:, l_idx]
    
    # fl with one hot encoding
    fl = one_hot_target(Y[l_idx])
    
    # Compute fu using regularized Laplacian
    Q = Luu + laplacian_regularization * np.eye(u_idx.shape[0])
    Q_inv = np.linalg.inv(Q)
    fu = np.dot(Q_inv, np.dot(Wul, fl))
    
    # Infer label from computed fu using thresholding
    fu_lab = np.argmax(fu, axis=1)
    
    # Consolidate labels from fu and fl
    labels = np.zeros((num_samples, ), dtype=int)
    # +1 because labels start at 1
    labels[u_idx] = fu_lab + 1
    labels[l_idx] = Y[l_idx]

    return labels


def two_moons_hfs_hard(path, dataset = '/data/data_2moons_hfs', l = 4, var=1, eps=0, k=10, laplacian_normalization=""):
    # a skeleton function to perform HFS, needs to be completed

    # load the data
    in_data =scipy.io.loadmat(path + dataset)
    X = in_data['X']
    Y = in_data['Y'][:, 0]

    # Draw at random index of labels
    Y_masked = mask_labels(Y, l)

    labels = hard_hfs(X, Y_masked, var=var, eps=eps, k=k, laplacian_normalization=laplacian_normalization)

    helper.plot_classification(X, Y, labels,  var=var, eps=eps, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))
    
    return accuracy


def soft_hfs(X, Y, c_l, c_u, laplacian_regularization=0.000001 ,var=1, eps=0, k=10, laplacian_normalization=""):
    #  a skeleton function to perform soft (unconstrained) HFS,
    #  needs to be completed
    #
    #  Input
    #  X:
    #      (n x m) matrix of m-dimensional samples
    #  Y:
    #      (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
    #  c_l,c_u:
    #      coefficients for C matrix

    #
    #  Output
    #  labels:
    #      class assignments for each (n) nodes
    num_samples = np.size(X, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    # Indices of labelled and unlabelled data
    l_idx = np.argwhere(Y != 0)[:, 0]
    u_idx = np.argwhere(Y == 0)[:, 0]
    
    # Build C matrix
    diagC = np.zeros((num_samples, ))
    diagC[l_idx] = c_l
    diagC[u_idx] = c_u
    C = np.diag(diagC)
    C_inv = np.diag(1 / diagC)
    
    # Target vector
    y = one_hot_target(Y)[:, 1:]
    
    # Build similarity graph and Laplacian
    W = helper.build_similarity_graph(X, var, eps, k)
    L = helper.build_laplacian(W, laplacian_normalization)
    
    # Q matrix (regularized laplacian)
    Q = L + laplacian_regularization * np.eye(num_samples)
    
    # Computation of fstar
    D = np.dot(C_inv, Q) + np.eye(num_samples)
    D_inv = np.linalg.inv(D)
    fstar = np.dot(D_inv, y)
    
    # +1 : labels start at 1
    labels = np.argmax(fstar, axis=1) + 1
    
    return labels 


def two_moons_hfs_soft(path, cl, cu, l = 4, var=1, eps=0, k=10, laplacian_normalization=""):
    # a skeleton function to perform HFS, needs to be completed
    # load the data
    in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs')
    X = in_data['X']
    Y = in_data['Y'][:, 0]

    # automatically infer number of labels from samples
    num_samples = X.shape[0]
    num_classes = len(np.unique(Y))

    # Draw at random index of labels
    Y_masked = mask_labels(Y, l)

    labels = soft_hfs(X, Y_masked, cl, cu, var=var, eps=eps, k=k, laplacian_normalization=laplacian_normalization)

    helper.plot_classification(X, Y, labels,  var=var, eps=eps, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))
    
    return accuracy


def hard_vs_soft_hfs(path, cl, cu, l = 20, var=1, eps=0, k=10):
    # a skeleton function to confront hard vs soft HFS, needs to be completed

    # load the data
    in_data =scipy.io.loadmat(path+'/data/data_2moons_hfs')
    X = in_data['X']
    Y = in_data['Y'][:, 0]

    # automatically infer number of labels from samples
    num_samples = X.shape[0]
    num_classes = len(np.unique(Y))

    # Draw at random index of labels
    l_idx = np.random.choice(range(num_samples), l, replace=False)

    # mask labels
    Y_masked = np.zeros((num_samples, ), dtype = int)
    Y_masked[l_idx] = Y[l_idx]
    
    
    Y_masked[Y_masked != 0] = helper.label_noise(Y_masked[Y_masked != 0], 4)
    
    l_idx = np.argwhere(Y_masked != 0)
    
    soft_labels = soft_hfs(X, Y_masked, cl, cu, var=var, eps=eps, k=k)
    hard_labels = hard_hfs(X, Y_masked, var=var, eps=eps, k=k)

    Y_masked[Y_masked == 0] = np.squeeze(Y)[Y_masked == 0]

    helper.plot_classification_comparison(X, Y, hard_labels, soft_labels, var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy