import cv2, os
import math
import scipy
import numpy as np
import scipy.misc
import math
import re
import pandas as pd
import operator
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt


import numpy as np

def compute_pca(data):
    m = np.mean(data, axis=0)
    datac = np.array([obs - m for obs in data])
    stddev = np.std(data, axis=0)
    datac = datac/stddev
    T = np.dot(datac, datac.T)
    [u,s,v] = np.linalg.svd(T)

    eigenvalues, temp = np.linalg.eigh(T)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]

    # here iteration is over rows but the columns are the eigenvectors of T
    pcs = [np.dot(datac.T, item) for item in u.T ]

    # note that the eigenvectors are not normed after multiplication by T^T
    pcs = np.array([d / np.linalg.norm(d) for d in pcs])

    return pcs, m, eigenvalues #, s, T, u

# def compute_projections(I,pcs,m):
#     projections = []
#     for i in I:
#         w = []
#         for p in pcs:
#             w.append(np.dot(i - m, p))
#         projections.append(w)
#     return projections
#
# def reconstruct(w, X, m,dim = 5):
#     return np.dot(w[:dim],X[:dim,:]) + m
#
# def normalize(samples, maxs = None):
#     # Normalize data to [0,1] intervals. Supply the scale factor or
#     # compute the maximum value among all the samples.
#
#     if not maxs:
#         maxs = np.max(samples)
#     return np.array([np.ravel(s) / maxs for s in samples])

def plot_eigenfaces(EV):
    for i in range (10):
        figure = plt.figure()
        figure.suptitle('Eigenface')
        plt.gray()
        plt.imshow(EV[i].reshape(243,320))
        # plt.show()
        # figure.savefig("Eigenfaces/" + str(i) + ".png")


def plot_eigenValues(E):
  plot = plt.plot(E)
  plt.title('EigenValues')
  # plt.savefig("EigenValuePlot.png")
  # plt.show()

def CrossValidation(imgData, k_eigen):
    totalrows = len(imgData)

    bestk = 0
    bestaccuracy = 0
    avg_accuracy = 0
    kplot = []
    accuracyplot = []

    for k in range(1,25,2):
        #print(k)
        accuracysum = 0
        for i in range(0,5):
            teststartrow = int((i*totalrows)/5)
            testendrow = int(((i+1)*totalrows)/5)
            localtestdata = imgData.iloc[teststartrow:testendrow]
            localtraindata = imgData.drop(imgData.index[teststartrow:testendrow], inplace=False)
            accuracysum += calcAccuracy(localtraindata, localtestdata, k, k_eigen)

        avg_accuracy = accuracysum/5;

        kplot.append(k)
        accuracyplot.append(avg_accuracy)
        print(avg_accuracy)

        if avg_accuracy > bestaccuracy:
            bestaccuracy = avg_accuracy
            bestk = k

    # plt.plot(kplot,accuracyplot)
    # plt.ylabel('Accuracy')
    # plt.xlabel('K')
    # plt.show()

    return bestk

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def findKneighbors(traindf, testdfForRegression, k, k_eigen):
    distances = []
    length = len(testdfForRegression) - 1
    for x in range(len(traindf)):
        dist = euclideanDistance(testdfForRegression, traindf.iloc[x,:k_eigen], length)
        distances.append((traindf.iloc[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def predict(neighbors, k_eigen):
    countsForImageClassification = [0,0,0,0,0,0,0,0,0,0,0,0]

    for x in range(len(neighbors)):
        response = neighbors[x][k_eigen]
        countsForImageClassification[int(response)] +=1

    return countsForImageClassification.index(max(countsForImageClassification))


def getAccuracy(test_output, predictions):
    correct = 0

    for x in range(0, len(test_output) - 1):
        test_val = test_output.iloc[x].astype(str).astype(int)

        if test_val == int(predictions[x]):
            correct += 1

    return (correct / float(len(test_output))) * 100.0


def normalize_column(A, col):
    A.iloc[:, col] = (A.iloc[:, col] - np.mean(A.iloc[:, col])) /(np.std(A.iloc[:, col]))


def calcAccuracy(traindata, testdata, k, k_eigen):
    traindfForRegression = traindata.iloc[:,:k_eigen]
    testdfForRegression = testdata.iloc[:,:k_eigen]

    totallen = len(testdata)
    predictions = []

    for x in range(0, totallen - 1):
        neighbors = findKneighbors(traindata, testdfForRegression.iloc[x], k, k_eigen)
        result = predict(neighbors, k_eigen)
        predictions.append(result)
        #print('> predicted= ' + repr(result) + ', actual= ' + repr(test_output.iloc[x])) # 'Area']))

    accuracy = getAccuracy(testdata.iloc[:,k_eigen], predictions)
    return accuracy

    print('Accuracy:'+ repr(accuracy) + '%')


def pca_n_faceRecog():
    directory = 'yalefaces'
    im_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.gif')]

    imnbr = len(im_paths) #get the number of images

    immatrix = np.array([np.array(Image.open(im_paths[i]).convert('L')).flatten() for i in range(imnbr)], 'f')

    # perform PCA
    EV, mean_X, E = compute_pca(np.array(immatrix,'f'))

    # plot_eigenfaces(EV)
    # plot_eigenValues(E)

    #capture 50% energy
    E_cumsum = np.cumsum(E)
    i = 0
    while((E_cumsum[i]/E_cumsum[E_cumsum.size - 1]) < 0.50):
        i +=1
    print(i+1)

    # k eigen-vectors
    k_eigen = 106

    EV = EV.T[:,:k_eigen]

    # projected matrix
    U = np.dot(immatrix - mean_X,EV)

    recnstr_img = np.dot(U, EV.T)

    for i in range(recnstr_img.shape[0]):
        img = recnstr_img[i].reshape(243,320)
        figure = plt.figure()
        figure.suptitle('Reconstructed Image')
        plt.gray()
        plt.imshow(img)
        # plt.show()
        figure.savefig("reconstImages/" + im_paths[i].replace('.gif', '.png'))

    #im_output = np.array([int(s) for s in im_paths.split() if s.isdigit()])

    tstMatrix = []
    trnMatrix = []
    trnOutput = []
    tstOutput = []

    img_output = np.array([re.findall(r'\d+', str(im_paths[i])) for i in range(imnbr)])
    #img_Data = U.append(img_output, axis = 1)

    for i in range(11):
        X_train, X_test, Y_train, Y_test = train_test_split(U[i*11:i*11+11], img_output[i*11:i*11+11], test_size=0.33)

        for i in range(len(X_train)):
            trnMatrix.append(X_train[i])
            trnOutput.append(Y_train[i])
        for i in range(len(X_test)):
            tstMatrix.append(X_test[i])
            tstOutput.append(Y_test[i])


    trnMatrix = pd.DataFrame(trnMatrix)
    trnOutput = pd.DataFrame(trnOutput)
    tstMatrix = pd.DataFrame(tstMatrix)
    tstOutput = pd.DataFrame(tstOutput)

    trnData = np.append(trnMatrix, trnOutput, axis = 1)
    np.random.shuffle(trnData)
    trnData = pd.DataFrame(trnData)
    #print(trnData.iloc[:,100])

    k = 1 # CrossValidation(trnData, k_eigen)
    # print(k)

    predictions = []
    totallen = len(tstMatrix)

    for x in range(0, totallen - 1):
        neighbors = findKneighbors(trnData, tstMatrix.iloc[x], k, k_eigen)
        result = predict(neighbors, k_eigen)
        predictions.append(result)
        #print('> predicted= ' + repr(result) + ', actual= ' + repr(test_output.iloc[x])) # 'Area']))

    accuracy = getAccuracy(tstOutput.iloc[:,0], predictions)
    print('Accuracy:'+ str(accuracy) + '%')

pca_n_faceRecog()