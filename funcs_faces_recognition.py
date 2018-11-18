import matplotlib.pyplot as plt
import scipy.misc as sm
import numpy as np
import cv2
import os
import sys
import funcs_hfs


def resize_rectangle(rect, lx, ly, xbounds, ybounds):
    """Reshape rectangle around its center"""
    center = (int(rect[0] + rect[2]/2), int(rect[1] + rect[3]/2))
    x = center[0] - int(lx/2)
    y = center[1] - int(ly/2)
    if x + lx > xbounds[1]:
        x = xbounds[1] - lx
    elif x - lx < xbounds[0]:
        x = xbounds[0]
    if y + ly > ybounds[1]:
        y = ybounds[1] - ly
    elif y - ly < ybounds[0]:
        y = ybounds[0]
    return x, y, lx, ly


def load_faces(path, dataadd = "/10faces/", xmladd=""):

    # Parameters
    cc = cv2.CascadeClassifier(path + xmladd + 'haarcascade_frontalface_default.xml')
    frame_size = 96

    # List folders
    datapath = path + dataadd
    folders = os.listdir(datapath)
    folders.sort()

    # Infer nlabels and nimages per labels
    nlabels = len(folders)
    nimages_perlabel = len(os.listdir(datapath + "/" + folders[0] + "/"))

    # Loading images
    images = np.zeros((nlabels * nimages_perlabel, frame_size ** 2))
    labels = np.zeros(nlabels * nimages_perlabel)

    # Initialize counter variable
    j = 0

    # Loop over all files
    for folder in os.listdir(datapath):
        files = os.listdir(datapath + folder + "/")
        files.sort()
        i = 0
        for file in files:
            # Read image
            im = cv2.imread(datapath + folder + "/" + file)
            # Detect face
            # (Catch exception for some image that raise error)
            box = cc.detectMultiScale(im)
            box = box[0]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # Resize rectangle so that it fits in frame size
            xbounds = (0, im.shape[0] - 1)
            ybounds = (0, im.shape[1] - 1)
            x, y, lx, ly = resize_rectangle(box, frame_size, frame_size, xbounds, ybounds)
            # Convert to grayscale
            gray_face = gray_im[y:y + ly, x:x + lx]
            # resize the face and reshape it to a row vector, record labels
            gf = gray_face.copy()
            images[j * nlabels + i] = gf.reshape((frame_size ** 2, ))
            labels[j * nlabels + i] = int(folder) + 1
            i += 1
        j += 1
    return images, labels


def load_faces_extended(path, dataadd = "/extended_dataset/", xmladd="/data/", nbimgs=50):

    # Parameters
    cc = cv2.CascadeClassifier(path + xmladd + 'haarcascade_frontalface_default.xml')
    frame_size = 96

    # List folders
    datapath = path + dataadd
    folders = os.listdir(datapath)
    folders.sort()

    # Infer nlabels and nimages per labels
    nlabels = len(folders)

    # Loading images
    images = np.zeros((nlabels * nbimgs, frame_size ** 2))
    labels = np.zeros(nlabels * nbimgs)

    # Initialize counter variable
    j = 0

    # Loop over all files
    for folder in os.listdir(datapath):
        files = os.listdir(datapath + folder + "/")
        # So as to get more different pictures (pictures that are following one another are very similar)
        np.random.shuffle(files)
        i = 0
        file_ind = 0
        while i < nbimgs:
            # Read image
            im = cv2.imread(datapath + folder + "/" + files[file_ind])
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # Detect face
            # (Catch exception for some image that raise error)
            box = cc.detectMultiScale(gray_im)
            try:
                box = box[0]
                # Resize rectangle so that it fits in frame size
                xbounds = (0, im.shape[0] - 1)
                ybounds = (0, im.shape[1] - 1)
                x, y, lx, ly = resize_rectangle(box, frame_size, frame_size, xbounds, ybounds)
                # Convert to grayscale
                gray_face = gray_im[y:y + ly, x:x + lx]
                # resize the face and reshape it to a row vector, record labels
                gf = gray_face.copy()
                try:
                    images[j * nbimgs + i] = gf.reshape((frame_size ** 2, ))
                    labels[j * nbimgs + i] = int(folder) + 1
                    i += 1
                    file_ind += 1
                except ValueError:
                    file_ind += 1
                # print(i)
            except IndexError:
                file_ind += 1
        j += 1
    return images, labels


def pre_process(X):
    """Skeleton function for trying diverse pre processing operations"""
    dim = int(np.sqrt(X.shape[1]))
    nims = X.shape[0]
    Xproc = [X[i, :].reshape((dim, dim)).astype(np.uint8) for i in range(0, nims)]
    # Gaussian Blur
    Xproc = [cv2.GaussianBlur(Xproc[i], ksize=(5, 5), sigmaX=1) for i in range(0, nims)]
    # Histogram equalization
#     Xproc = [cv2.equalizeHist(Xproc[i]) for i in range(0, nims)]
    Xproc1d = np.array([Xproc[i].reshape((-1)) for i in range(0, nims)])
    return Xproc1d


def mask_labels_faces(Y, lper_person):
    labels = np.unique(Y)
    Y_masked = np.zeros(Y.shape, dtype=int)
    for lab in labels:
        inds = np.argwhere(Y == lab)[:, 0]
        choice = np.random.choice(inds, lper_person, replace=False)
        Y_masked[choice] = lab
    return Y_masked


def offline_face_recognition_hard(X, Y, l=4, 
                                  laplacian_regularization=0.0001, 
                                  var=1e3, eps=0, k=10, 
                                  laplacian_normalization="", bigfigure=False):

    nbimgs = int(Y.shape[0] / 10)
    Y_masked = mask_labels_faces(Y, l)
    labels = Y
    rlabels = funcs_hfs.hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    u_idx = np.argwhere(Y_masked == 0)
    perfs_unlabelled = np.equal(rlabels[u_idx], labels[u_idx]).mean()
    
    # Plots #
    if bigfigure:
        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(121)
    plt.imshow(labels.reshape((10, nbimgs)))

    plt.subplot(122)
    plt.imshow(rlabels.reshape((10, nbimgs)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()
    return perfs_unlabelled


def offline_face_recognition_soft(X, Y, cl=10, cu=1, l=4, 
                                  laplacian_regularization=0.0001, 
                                  var=1e3, eps=0, k=9, 
                                  laplacian_normalization="", bigfigure=False):

    nbimgs = int(Y.shape[0] / 10)
    Y_masked = mask_labels_faces(Y, l)
    labels = Y
    rlabels = funcs_hfs.soft_hfs(X, Y_masked, cl, cu, laplacian_regularization, var, eps, k, laplacian_normalization)

    u_idx = np.argwhere(Y_masked == 0)
    perfs_unlabelled = np.equal(rlabels[u_idx], labels[u_idx]).mean()

    # Plots #
    if bigfigure:
        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(121)
    plt.imshow(labels.reshape((10, nbimgs)))

    plt.subplot(122)
    plt.imshow(rlabels.reshape((10, nbimgs)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()
    return perfs_unlabelled


def merge_datasets(X1, X2, Y1, Y2):
    return np.concatenate((X1, X2), axis=0), np.concatenate((Y1, Y2))