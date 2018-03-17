from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
import glob
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
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt



# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen = []
#datagen.append(ImageDataGenerator(zca_whitening=True))
datagen.append(ImageDataGenerator(zoom_range=0.2))
datagen.append(ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2))
datagen.append(ImageDataGenerator(horizontal_flip=True, vertical_flip=True))

directory = 'yalefaces'
im_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.gif')]

img_matrix = []
imnbr = len(im_paths)

for filename in glob.glob('yalefaces/*.*'):
	im = Image.open(filename).convert('L')
	pixels = list(im.getdata())
	width, height = im.size
	pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
	img_as_np = np.array(pixels)

	img_as_np = img_as_np.flatten()
	img_matrix.append(img_as_np)


img_output = np.array([re.findall(r'\d+', str(im_paths[i])) for i in range(imnbr)])
img_matrix = np.array(img_matrix)

i = 0

# for j in range(165):
# 	figure = plt.figure()
# 	figure.suptitle('AugmentedImage')
# 	plt.imshow(img_matrix[j].reshape(243,320), cmap=plt.get_cmap('gray'))
# 	plt.show()
# 	# figure.savefig('yalefaces_aug/' + im_paths[j].replace('.gif', '.png'))

# for i in range(3):
# 	for j in range(165):
# 		img = img_matrix[j].reshape(1, 1, 243, 320)
# 		img = np.array(img)
#
# 		# configure batch size and retrieve one batch of images
# 		batch = datagen[i].flow(img, batch_size=1)[0]
#
# 		figure = plt.figure()
# 		figure.suptitle('AugmentedImage')
# 		plt.imshow(batch.reshape(243,320), cmap=plt.get_cmap('gray'))
# 		plt.show()
# 		# figure.savefig('yalefaces_aug/' + im_paths[j].replace('.gif', '') + '_' + str(i) + '.png')

directory1 = 'yalefaces_aug'
im_paths1 = [os.path.join(directory1, filename) for filename in os.listdir(directory1) if(filename.endswith('.png'))]

imnbr1 = len(im_paths1) #get the number of images
immatrix1 = np.array([scipy.misc.imresize((Image.open(im_paths1[i]).convert('L')),(100,100)).flatten() for i in range(imnbr1)], 'f')
img_output1 = np.array([re.findall(r'\d+', str(im_paths1[i]))[0] for i in range(imnbr1)])

tstMatrix = []
trnMatrix = []
trnOutput = []
tstOutput = []

#print(im_paths1[0:44])

for i in range(11):
	X_train, X_test, Y_train, Y_test = train_test_split(immatrix1[i * 44:i * 44 + 44], img_output1[i * 44:i * 44 + 44], test_size=0.33)

	# train_image, test_image = arrange_dataset(im_paths[i*11:i*11+11])
	for i in range(len(X_train)):
		trnMatrix.append(X_train[i])
		trnOutput.append(Y_train[i])

	for i in range(len(X_test)):
		tstMatrix.append(X_test[i])
		tstOutput.append(Y_test[i])


trnMatrix = np.array(trnMatrix)
trnOutput = np.array(trnOutput)
tstMatrix = np.array(tstMatrix)
tstOutput = np.array(tstOutput)

trnData = np.append(pd.DataFrame(trnMatrix), pd.DataFrame(trnOutput), axis=1)
np.random.shuffle(trnData)
trnData = pd.DataFrame(trnData)
# print(trnData.iloc[:,100])

#Create model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), strides = (1,1), activation='relu', input_shape=(100,100,1)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid', data_format=None))
model.add(keras.layers.Dropout(0.2))
model.add(Flatten())
model.add(Dense(12))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

trnData1 = np.array(trnMatrix.reshape(len(trnMatrix), 100, 100, 1))
trnOutput = keras.utils.to_categorical(trnOutput.astype(str).astype(int))

#Fit model
model.fit(trnData1, trnOutput, verbose=1)
tstOutput = keras.utils.to_categorical(tstOutput.astype(str).astype(int))

eval = model.evaluate(tstMatrix.reshape(len(tstMatrix),100,100,1), tstOutput, verbose=1)

print('Test loss:', eval[0])
print('Test accuracy:', eval[1])