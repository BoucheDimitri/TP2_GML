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
    gamma = .95
    var=10000

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

    # Initialize counter variables
    i = 0
    j = 0

    # Loop over all files
    for folder in os.listdir(datapath):
        files = os.listdir(datapath + folder + "/")
        files.sort()
        for file in files:
            print(file)
            # Read image
            im = cv2.imread(datapath + folder + "/" + file)
            # Detect face
            box = cc.detectMultiScale(im)[0]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # Resize rectangle so that it fits in frame size
            xbounds = (0, im.shape[0] - 1)
            ybounds = (0, im.shape[1] - 1)
            x, y, lx, ly = resize_rectangle(box, frame_size, frame_size, xbounds, ybounds)
            # Convert to grayscale
            gray_face = gray_im[y:y + ly, x:x + lx]
            #resize the face and reshape it to a row vector, record labels
            gf = gray_face.copy()
            try:
                images[j * nlabels +  i % 10] = gf.reshape((frame_size ** 2, ))
                labels[j * nlabels + i % 10] = j + 1
                i += 1
            except ValueError:
                print("Reshaping problem encountered, leave out the problematic image")
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
                                  laplacian_normalization=""):

    Y_masked = mask_labels_faces(Y, l)
    labels = Y
    rlabels = funcs_hfs.hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    
    # Plots #
    plt.subplot(121)
    plt.imshow(labels.reshape((10, 10)))

    plt.subplot(122)
    plt.imshow(rlabels.reshape((10, 10)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()


def offline_face_recognition_soft(X, Y, cl=10, cu=1, l=4, 
                                  laplacian_regularization=0.0001, 
                                  var=1e3, eps=0, k=9, 
                                  laplacian_normalization=""):

    Y_masked = mask_labels_faces(Y, l)
    labels = Y
    rlabels = funcs_hfs.soft_hfs(X, Y_masked, cl, cu, laplacian_regularization, var, eps, k, laplacian_normalization)
    # Plots #
    plt.subplot(121)
    plt.imshow(labels.reshape((10, 10)))

    plt.subplot(122)
    plt.imshow(rlabels.reshape((10, 10)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()


def add_extension_expanded(datapath):
    folders = os.listdir(datapath)
    folders.sort()
    for folder in os.listdir(datapath):
        files = os.listdir(datapath + folder + "/")
        files.sort()
        for file in files:
            os.rename(datapath + "/" + folder + "/" + file, datapath + "/" + folder + "/" + file + ".jpg")