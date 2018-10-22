import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

# Please change them into your own directory
train_data = '/Users/yxk/Documents/Python/601/Miniproject2/Train'
test_data = '/Users/yxk/Documents/Python/601/Miniproject2/Test'
validation_data = '/Users/yxk/Documents/Python/601/Miniproject2/Validation'

def one_hot(labellist, label):
    '''
    make label into one hot representation according to the order in labellist
    '''
    labelset = [0] * len(labellist)
    labelset[labellist.index(label)] = 1
    ohl = np.array(labelset)
    return ohl

def getlabel(data_path):
    '''
    make all images into matrices and label them use one_hot according to their folder's name
    '''
    images = []
    foldername = [element for element in os.listdir(data_path) if element != '.DS_Store']

    for folder in foldername:

        img_path = os.path.join(data_path, folder)
        imgnames = os.listdir(img_path)
        imglabel = one_hot(foldername, folder)

        for imgname in imgnames:

            if imgname == '.DS_Store':
                continue

            path = os.path.join(img_path, imgname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            try:
                img = cv2.resize(img, (128, 128))
            except:
                print(imgname, 'This image may have some problems, please check it')
                quit()
            images.append([np.array(img), imglabel])

    shuffle(images)
    return images

def flatten(data):
    img_data = np.array([i[0] for i in data]).reshape(-1, 128, 128, 1)
    lbl_data = np.array([i[1] for i in data])
    return img_data, lbl_data

def convmodel3(training_image, training_label, validation_data, validation_label):
    '''
    build and train a 3 convolutional layers neural network
    '''
    dir = [element for element in os.listdir(train_data) if element != '.DS_Store']
    classes = len(dir)

    model = Sequential()

    model.add(InputLayer(input_shape = [128, 128, 1]))
    model.add(Conv2D(filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 50, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(classes, activation = 'softmax'))

    # if during each epoch, accuracy doesn't change, please change 4e-4 into 5e-4 or change back
    optimizer = Adam(lr = 4e-4)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Please change save_model_path into your own directory
    save_model_path = '/Users/yxk/Documents/Python/601/Miniproject2/best_model3.hdf5'
    # Save the best model into save_model_path during each epoch
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, save_best_only=True, verbose=1)
    # stop training if the val_acc don't improve for 10 epochs in order to prevent overfitting
    es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
    # start training this model
    history = model.fit(x = training_image, y = training_label, epochs = 50, batch_size = 64, validation_data = (validation_data, validation_label), callbacks = [es, cp])
    model.summary()
    return model, history # if you use it as API and just want the best_model.hdf5, you can delete it

def convmodel5(training_image, training_label, validation_data, validation_label):
    '''
    build and train a 5 convolutional layers neural network
    '''
    dir = [element for element in os.listdir(train_data) if element != '.DS_Store']
    classes = len(dir)

    model = Sequential()

    model.add(InputLayer(input_shape = [128, 128, 1]))
    model.add(Conv2D(filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 50, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(classes, activation = 'softmax'))

    optimizer = Adam(lr = 5e-4)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Please change save_model_path into your own directory
    save_model_path = '/Users/yxk/Documents/Python/601/Miniproject2/best_model5.hdf5'
    # Save the best model into save_model_path during each epoch
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, save_best_only=True, verbose=1)
    # stop training if the val_acc don't improve for 10 epochs in order to prevent overfitting
    es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
    # start training this model
    history = model.fit(x = training_image, y = training_label, epochs = 50, batch_size = 64, validation_data = (validation_data, validation_label), callbacks = [es, cp])
    model.summary()
    return model, history # if you use it as API and just want the best_model.hdf5, you can delete it

def test_plot(model, test_img):
    '''
    pick some test images and use trained model to predict them and show the result using matplotlib
    '''
    fig = plt.figure(figsize = (10, 10))
    labelname = [element for element in os.listdir(train_data) if element != '.DS_Store']

    for cnt, data in enumerate(test_img[10:40]):

        y = fig.add_subplot(6, 5, cnt+1)
        img = data[0]
        data = img.reshape(-1, 128, 128, 1)
        model_out = model.predict([data])
        str_label = labelname[np.argmax(model_out)]

        y.imshow(img, cmap = 'gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()

def evaluate_plot(history):
    '''
    plot the Training and validation loss and Training and validation accuracy's trend into line chart
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (10, 4))

    plt.subplot(121)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = 'upper right')

    plt.subplot(122)
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend(loc = 'lower right')

    plt.show()

def main():

    train_img = getlabel(train_data) # label all train images
    test_img = getlabel(test_data) # label all test images
    val_img = getlabel(validation_data) # label all test images

    tr_img_data, tr_lbl_data  = flatten(train_img) # flatten all data to perpare it for training
    tst_img_data, tst_lbl_data = flatten(test_img) # flatten all data to perpare it for training
    val_img_data, val_lbl_data = flatten(val_img) # flatten all data to perpare it for training

    # build and train a 3 layers convolutional neural network based on train data and validation data
    model3, history3 = convmodel3(tr_img_data, tr_lbl_data, val_img_data, val_lbl_data)
    # build and train a 5 layers convolutional neural network based on train data and validation data
    model5, history5 = convmodel5(tr_img_data, tr_lbl_data, val_img_data, val_lbl_data)

    # load a existing model's weight
    #model3.load_weights('best_model3.hdf5')

    # plot the Training and validation loss and Training and validation accuracy's trend into line chart
    evaluate_plot(history3)
    evaluate_plot(history5)

    # pick some test images and use trained model to predict them and show the result using matplotlib
    #test_plot(model3, test_img)

if __name__ == '__main__':
    main()
