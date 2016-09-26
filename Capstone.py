######################################################################################################################
# TODO : Import all used module out of the box to avoid looping importation

import cv2
import os
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

######################################################################################################################

def booler(str):
    if str == 'True' or str == 'true' or str == 't' or str == 'yes' or str == 'y' or str == 'ok' or str == '':
        return True
    else: return False


######################################################################################################################
# TODO : Get the absolute path and the user mode

# Absolute path make the using of cv2 easier
PATH = os.path.abspath("Capstone.py")[:-12]
# Check if user wish to run manually
Flag = booler(input('Do you wish to run automatic (or manual) ? (True, False) : '))

######################################################################################################################
# TODO : Convert TIFF images from both Xtrain and ytrain folders into PNG

# Images were convert from TIFF to PNG using this function
def TIFFtoPNG(path):
    temp = path
    for folderName in ['/Xtrain', '/ytrain']:
       path = temp + folderName
       for file in os.listdir(path):
           if 'tif' in file:
               name, ext = os.path.splitext(file)
               os.rename(os.path.join(path, file), os.path.join(path, name+'.png'))
               # Change TIFF to PNG if TIFF there is
TIFFtoPNG(PATH)

######################################################################################################################
# TODO : If mode is manual, allow globals to be constructed from the answers of the user with standard stepping for Gradient Descent

# input from user, answer is based on user hardware, if he had a NVDIA GPU card, user can allow bigger subset and

if not Flag:
   OVERALLSIZE = int(input('Choose the size of data : '))
   TESTSIZE = int(input('Choose the number of data point you want to use as test : '))
   BLACK = float(input('Choose the maximum portion of all-black masks : '))
   STEPS = 0.5

# Show standard unchanged images and relative mask using PIL
print('')
print('Showing standard image 1_1.png ...')
imgShow = Image.open("/Users/Jules/GoogleDrive/P5Capstone/Xtrain/1_1.png")
imgShow.show()
time.sleep(1)
imgShow.close()

print('Showing standard mask 1_1_mask.png')
print('')
mskShow = Image.open("/Users/Jules/GoogleDrive/P5Capstone/ytrain/1_1_mask.png")
mskShow.show()
time.sleep(1)
mskShow.close()

######################################################################################################################
# TODO : Check number of all-black masks within subset

# Check the number of all black mask
def blacker(masksList):
    counter = 0
    for mask in masksList:
        if not np.count_nonzero(mask): counter += 1
    return counter

######################################################################################################################
# TODO : Uses a MAIN function to allow repetition in automatic mode

def MAIN(OVERALLSIZE, TESTSIZE, STEPS, BLACK):
   print('######################################################')
   print('')
   print('Overall size of dataset is ' + str(OVERALLSIZE))
   print('Test set is 40% of dataset')
   print('Steps for Gradient Descent is ' + str(STEPS))
   print('')

   start = time.time()

   IMAGES = [img for img in os.listdir(PATH + '/Xtrain') if img.endswith('png')]

   def imageMaker():
       # Shuffle the images name
       imagesIn = shuffle(IMAGES)

       # Take on the relative masks name based directly on the images name
       masks = [name[:-4]+'_mask.png' for name in imagesIn]

       # Crop the list to user given size
       images, masks = imagesIn[:OVERALLSIZE], masks[:OVERALLSIZE]

       # Import the image left as grayscale numpy array using cv2.imread
       images_, masks_  = [cv2.imread(PATH + '/Xtrain/' + img, cv2.IMREAD_GRAYSCALE).astype(np.int) for img in images], \
                          [cv2.imread(PATH + '/ytrain/' + msk, cv2.IMREAD_GRAYSCALE).astype(np.int) for msk in masks]

       return images_, masks_

   # Avoid too non-informative dataset from which bad performances can be unleached
   ugod = imageMaker()
   images_, masks_ = ugod[0], ugod[1]
   crumpy = blacker(masks_)
   timeTemp = 0
   while crumpy > OVERALLSIZE * BLACK:
       ugod = imageMaker()
       image_, masks_ = ugod[0], ugod[1]
       crumpy = blacker(masks_)
       timeTemp += 1
       print('Too many all-black masks only ' + str(BLACK) + ' of mask can be all-black ... Reprocessing shuffling for the ' + str(timeTemp) + ' time')

   print('')
   print('Percentage of all-black masks in subset : ' + str(100.0 * crumpy / OVERALLSIZE) + '%')
   print('')

######################################################################################################################

   print('')
   print('Shape of images : ' + str(np.shape(images_[0])))
   print('Shape of masks : ' + str(np.shape(masks_[0])))
   print('')

######################################################################################################################

   # Find where is the extrema of ROI in masks given a list of them
   # Uses of prewritten library was very disappointing with lots of compatibilities issues, named : Scikit-Image, PIL, OpenCV. Writing a specialized function for problem
   def looker(listOfMatrix):

       up,down,left,right = [],[],[],[]
       for matrix in listOfMatrix:

           if np.count_nonzero(matrix) != 0:
               temp1 = np.argwhere(matrix)
               up.append(temp1[0][0])
               down.append(temp1[-1][0])

               temp2 = np.argwhere(np.transpose(matrix))
               left.append(temp2[0][0])
               right.append(temp2[-1][0])

       return np.min(up), np.max(down), np.min(left), np.max(right)

   cookie = looker(masks_)

   print('ROI is skewed withing boundaries : ')
   print('up-most row: ' + str(cookie[0]))
   print('down-most row: ' + str(cookie[1]))
   print('left-most column: ' + str(cookie[2]))
   print('right-most column: ' + str(cookie[3]))
   print('')

######################################################################################################################

   print('CALCULATING...')

######################################################################################################################

   # Split the dataset in test train and normalizing to avoid too big number which are hard to handle within TensorFlow
   X_train, y_train, X_test, y_test = np.asarray(images_[TESTSIZE:])/255., \
                                      np.asarray(masks_[TESTSIZE:])/255, \
                                      np.asarray(images_[:TESTSIZE])/255., \
                                      np.asarray(masks_[:TESTSIZE])/255

######################################################################################################################
# TODO : Create the placeholder for Tensorflow to work correctly

   # Create 3D placeholder to contain the inputs [batch, height+grayscale, width+grascale] (tensor will further be change to 4D [batch, height, width, grayscale])
   x = tf.placeholder(tf.float32, shape=[None, 420, 580], name='x')

   # Create a 4D placeholder to contain label inputs [batch, height, width, grascale]
   y_ = tf.placeholder(tf.float32, shape=[None, 420, 580,1], name='y_')

######################################################################################################################
# TODO : start an interactive session to be able to evaluate variable at every time

   # Create a evaluation variable with interactive value allowing to .run (evaluate) a TensorFlow statement at any time
   sess = tf.InteractiveSession()

######################################################################################################################
# TODO : Create weights and biases tensors generality form

   # Create a weight array from a standard deviation stddev and with shape given
   def weight_variable(shape):
       initial = tf.truncated_normal(shape, stddev = 0.1)
       return tf.Variable(initial)

   # Create a bias array from given shape
   def bias_variable(shape):
       initial = tf.constant(0.1, shape=shape)
       return tf.Variable(initial)

######################################################################################################################
# TODO : Create low-end general convolution and pooling

   # Create a convolutional 2D function with strides (window sliding) over 1 batch at a time (other would get strange results) of 1x1 area and on 1 channel
   def conv2d(x, W):
       # padding SAME give a same dimensionality (by not overlapping the sliding windows)
       return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

   # Create a max pool operation over 2x2 region
   def max_pool_2x2(x):
       return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

######################################################################################################################
# TODO : Create high-end UNS problem-related convolution and pooling

   # Create a convolution pooling step
   def convoer(inputs, shape, flag):
       # Create parameters
       W = weight_variable(shape)
       b = bias_variable([shape[3]])

       temp = shape
       temp[2] = shape[3]

       Wa = weight_variable(temp)
       ba = bias_variable([shape[3]])

       # Step convolution with inputs
       conv = tf.nn.relu(conv2d(inputs, W) + b)
       # Step another convolution with conv to allow further tuning
       conv = tf.nn.relu(conv2d(conv, Wa) + ba)
       # Max pool over 2x2 section (allow model to be rotational, sliding and basic transformations independent, also reduce dimensionality without giving up too much of the informations gained)
       pool = max_pool_2x2(conv)

       # Allow convenience to conv10 and conv6
       if flag: return pool
       elif not flag: return conv

   # Create a deconvolution pooling step
   def upconvoer(inputs, shape, height, width):
       # Create parameters
       W = weight_variable(shape)
       b = bias_variable([shape[3]])

       temp = shape
       temp[2] = shape[3]

       Wa = weight_variable(temp)
       ba = bias_variable([shape[3]])

       # Resize the matrix to upper given dimensions
       up = tf.image.resize_images(inputs, height, width)
       # Allow the model to learn by conving on it
       conv = tf.nn.relu(conv2d(up, W) + b)
       # Conv a second time to allow further learning
       conv = tf.nn.relu(conv2d(conv, Wa) + ba)

       return conv

######################################################################################################################
# TODO : Construct the Unet

   # Idea for Unet was taken from Kera submission : https://github.com/jocicmarko/ultrasound-nerve-segmentation
   def U():
       # reshape the x placeholder (now a tensor) into a more convenient shape
       inputs = tf.reshape(x, [-1,420,580,1])

       # allow a 1st step of convolution-pooling method as previously described
       pool1 = convoer(inputs, [3,3,1,32], True)

       # Images are decrease by a factor ~2 while their number of features by pixel is gaining a factor ~2
       # allow a convolution with pooling (flag True)
       pool2 = convoer(pool1, [3,3,32,64], True)
       pool3 = convoer(pool2, [3,3,64,128], True)
       pool4 = convoer(pool3, [3,3,128,256], True)
       # allow a convolution with no pooling (flag False)
       conv5 = convoer(pool4, [3,3,256,512], False)
       # deconvolution and image resize by factor ~2
       conv6 = upconvoer(conv5, [3,3,512,256], 53, 73)
       conv7 = upconvoer(conv6, [3,3,256,128], 105, 145)
       conv8 = upconvoer(conv7, [3,3,128,64], 210, 290)
       # deconvolution leads to same size image
       conv9 = upconvoer(conv8, [3,3,64,32], 420, 580)

       # Create manually 2 parameters
       W10 = weight_variable([1,1,32,1])
       b10 = bias_variable([1])

       # Sigmoid function ensure answer will be between 0 and 1 (binary mask)
       conv10 = tf.nn.sigmoid(conv2d(conv9, W10) + b10)

       return conv10

######################################################################################################################
# TODO : Run the Unet and check for accuracy

   y = U()

   # Initializing all the variable before further attribution is the key to avoid 'variable is attributed before defined' problem very common in the non-linear structure of tensorflow (due to C compiling)
   sess.run(tf.initialize_all_variables())

   # Cross-entropy definition
   cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

   # Use gradient descent as a reductor of cross-entropy cost function
   train_step = tf.train.GradientDescentOptimizer(STEPS).minimize(cross_entropy)

   # resize the y_train to meet its placeholder dimension of 4D
   joe = np.resize(y_train, [len(y_train), 420, 580, 1])

   # train the model with no dropout, might be existent in further implementation
   sess.run(train_step, feed_dict={x: X_train, y_: joe})

   # Calculate y_true
   correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

   # Compute accuracy
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

   # resize y_test to meet its placeholder dimension of 4D
   joey = np.resize(y_test, [len(y_test), 420, 580, 1])

   # train over optimizing accuracy, no dropout
   train_accuracy = accuracy.eval(feed_dict = {x : X_test, y_: joey})

######################################################################################################################
# TODO : Give user a look at accuracy

   # give back accuracy
   print('')
   print('accuracy from training set size : ' + str(OVERALLSIZE - TESTSIZE) + ' and test size : ' + str(TESTSIZE))
   print(str(train_accuracy))

   end = time.time()-start
   print('')

   print('time to run the whole program : ' + str(end) + ' seconds')
   print('')

   return train_accuracy, end

######################################################################################################################
# TODO : Call for manual mode

if not Flag:
   MAIN(OVERALLSIZE, TESTSIZE, STEPS, BLACK)

######################################################################################################################
# TODO : Call for automatic mode with plot showing

else:

   def plotter(x,y,xlabel,ylabel,title):
       plt.plot(x, y)
       plt.ylabel(ylabel)
       plt.xlabel(xlabel)
       plt.title(title)
       plt.show()

   SIZES = [10, 30, 50, 70, 100]
   STEPS = [0.3, 0.5, 0.8, 1, 10]
   BLACKS = [0.4, 0.5, 0.6, 0.7, 0.8]

   print('')
   print('TESTING DIFFERENT SIZES FOR DATASET...')
   print('')

   plot_size_accuracy = []
   plot_size_time = []
   for size in SIZES:
       temp = MAIN(size, int(0.4*size), 0.5, 0.6)
       plot_size_accuracy.append(temp[0])
       plot_size_time.append(temp[1])



   print('')
   print('TESTING DIFFERENT STEPPING FOR GRADIENT DESCENT...')
   print('')

   plot_steps_accuracy = []
   plot_steps_time = []
   for step in STEPS:
       temp = MAIN(10, 4, step, 0.6)
       plot_steps_accuracy.append(temp[0])
       plot_steps_time.append(temp[1])


   print('')
   print('TESTING DIFFERENT PORTIONS FOR ALL-BLACK MASK PORTION...')
   print('')

   plot_black_accuracy = []
   plot_black_time = []
   for black in BLACKS:
       temp = MAIN(10, 4, 0.5, black)
       plot_black_accuracy.append(temp[0])
       plot_black_time.append(temp[1])
   BLACKS = [black*100 for black in BLACKS]

   plotter(SIZES, plot_size_accuracy, 'Number of data points used in training', 'Accuracy',
           'Accuracy against size of data given test size of 40% and gradient steps of 0.5')
   plotter(SIZES, plot_size_time, 'Time', 'Number of data points used in training',
           'Time in seconds against size of data given test size of 40% and gradient steps of 0.5')

   plotter(STEPS, plot_steps_accuracy, 'Number of steps taken for Gradient Descent', 'Accuracy',
           'Accuracy against number of steps given a data set of 100 and a test set of 40')
   plotter(STEPS, plot_steps_time, 'Time', 'Number of steps taken for Gradient Descent',
           'Time against number of steps given a data set of 100 and a test set of 40')

   plotter(BLACKS, plot_black_accuracy, 'Percent of all-black masks allowed', 'Accuracy',
           'Accuracy against portion of all-black masks allowed given a data set of 100 and a test set of 40')
   plotter(BLACKS, plot_black_time, 'Percent of all-black masks allowed', 'Time',
           'Time against portion of all-black masks allowed given a data of 100 and a test set of 40')
#-----------------------------------------------------END--------------------------------------------------------------#


# Down is a way of finding the presence/absence of ROI (region of interset) brachial plexus. It can be used as a
# pretrainer to avoid very costly False Positive prediction but it is not included in present code since it boost the time waiting by a great factor
# For now the 2 programs are not linked but further implementations could take care of that.

# OVERALLSIZE = int(float(input('Choose the number of images you want (<5635) : '))) # OVERALLSIZE = int(float(input('Choose the percent of the data you want to use (recommended 10) : ')) * 5635 / 100)
# TESTSIZE = int(float(input('Choose the number of the data you want to use as test (<5635) : '))) # TESTSIZE = int(float(input('Choose the percent of the data you want to use as test (recommended 30) : ')) * 5635 / 100)
# PATH = '/Users/Jules/Google Drive/P5Capstone'
#
# ######################################################################################################################
#
# images = [img for img in os.listdir(PATH + '/Xtrain') if img.endswith('png')]
# # put random_state to 1
# images = shuffle(images,random_state = 0)
# masks = [name[:-4]+'_mask.png' for name in images]
#
# images, masks = images[:OVERALLSIZE], masks[:OVERALLSIZE]
# images_, masks_  = [cv2.imread(PATH + '/Xtrain/' + img, cv2.IMREAD_GRAYSCALE).flatten().astype(np.int) for img in images], \
#                    [cv2.imread(PATH + '/ytrain/' + msk, cv2.IMREAD_GRAYSCALE).flatten().astype(np.int) for msk in masks]
#
# ######################################################################################################################
#
#
# X_train, y_train, X_test, y_test = np.asarray(images_[TESTSIZE:])/255., \
#                                    np.asarray(masks_[TESTSIZE:]), \
#                                    np.asarray(images_[:TESTSIZE])/255., \
#                                    np.asarray(masks_[:TESTSIZE])
#
# def blacker(listOmatrix):
#     y = []
#     for narray in listOmatrix:
#         if not np.count_nonzero(narray): y.append([0, 1])
#         else : y.append([1, 0])
#     return y
#
# y_train = blacker(y_train)
# y_test = blacker(y_test)
#
# sess = tf.InteractiveSession()
#
# x = tf.placeholder(tf.float32, shape=[None, 243600])
# y_ = tf.placeholder(tf.float32, shape=[None, 2])
#
# def batcher(X, y, batch_size):
#     listPicker = random.sample(range(len(X)), batch_size)
#     a = [X[index] for index in listPicker]
#     b = [y[index] for index in listPicker]
#     return a, b
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev = 0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding='SAME')
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')
#
# W_conv1 = weight_variable([42,58,1,32])
# b_conv1 = bias_variable([32])
#
# x_image = tf.reshape(x, [-1,580,420,1])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# W_conv2 = weight_variable([5,5,32,64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# W_fc1 = weight_variable([105 * 145 * 64 , 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 105 * 145 * 64 ])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# W_fc2 = weight_variable([1024, 2])
# b_fc2 = bias_variable([2])
#
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables())
#
# train_accuracy = accuracy.eval(feed_dict = {x : X_test, y_: y_test, keep_prob: 1.0})
# print(train_accuracy)
