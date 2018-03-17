import cv2, os
import scipy
import numpy as np
import scipy.misc
from PIL import Image
import glob
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA, RandomizedPCA

def plotEigen(E):
  plt.plot(E)
  plt.title('EigenValues')
  plt.show()

# def pca(X):
#   # input: X, matrix with training data as flattened arrays in rows
#   # return: projection matrix (with important dimensions first), variance and mean
#
#   #center data
#   normal_X = X
#   mean_X = X.mean(axis = 0)
#   normal_X -= mean_X
#
#   M = np.dot(normal_X, normal_X.T) # covariance matrix, AA', not the A'A like usual
#   #M = np.dot(normal_X.T, normal_X) / (normal_X.shape[0]) #covariance matrix
#
#   E,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
#   tmp = np.dot(normal_X.T, EV)  # this is the compact trick
#   print(tmp.shape)
#
#   for i in range(165):
#       tmp[:,i] = tmp[:,i]/ np.linalg.norm(tmp[:,i])
#
#   EV = tmp #[::-1]  # reverse since last eigenvectors are the ones we want
#
#   # plot eigen faces
#   plt.figure()
#   plt.gray()
#   plt.imshow(EV[:,1].reshape(243, 320))
#   plt.show()
#
#   return E, EV, mean_X

def pca (X , num_components = 165):
    [n , d] = X.shape
    if (num_components <= 0) or ( num_components >n):
        num_components = n
    mu = X.mean( axis = 0)
    X = X - mu

    # if n > d:
    #     C = np.dot(X.T,X)
    #     [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    # else:

    C = np.dot(X ,X .T)
    [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    eigenvectors = np.dot(X .T, eigenvectors)

    for i in range(n):
        eigenvectors [: , i ] = eigenvectors [: , i ]/ np.linalg.norm( eigenvectors[: , i ])

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues [ idx ]
    eigenvectors = eigenvectors [: , idx ]
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy()
    eigenvectors = eigenvectors [: ,0: num_components ]. copy ()
    return [ eigenvalues , eigenvectors , mu ]


directory = 'yalefaces'
im_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.gif')]

labels = []

im = np.array(scipy.misc.imresize(Image.open(im_paths[0]), (100, 100))) #open one image to get the size
m,n = im.shape[0:2] #get the size of the images
imnbr = len(im_paths) #get the number of images


#create matrix to store all flattened images
# i_height = 100
# i_width = 100
# immatrix = np.array([scipy.misc.imresize(np.array(Image.open(im_paths[i])), (i_height, i_width)).flatten() for i in range(imnbr)], 'f')
immatrix = np.array([np.array(Image.open(im_paths[i]).convert('L')).flatten() for i in range(imnbr)], 'f')


#perform PCA
E, EV, mean_X = pca(np.array(immatrix,'f'))

# k eigen-vectors
k = 164

EV = EV[:,:k]

U = np.dot(immatrix - mean_X,EV)

recnstr_img = np.dot(U, EV.T)

for i in range(recnstr_img.shape[0]):
    first_img = recnstr_img[i].reshape(243,320)
    plt.figure()
    plt.gray()
    #plt.imshow(X[0].reshape(243,320))
    plt.imshow(first_img)
    plt.show()

