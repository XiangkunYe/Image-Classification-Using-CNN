# EC601_Deep-Learning

Hello! Here's Xiangkun Ye again, and the ***image_claffication.py*** is for ***miniproject2***
for ***EC601***.

## Befor start
* Please make sure to install ***tensorflow***, ***keras***, ***cv2***, ***numpy*** and ***matploblib***.  
* Please change those directories in the head of ***image_classification*** in it into your own path.

## Brief introduction
This program can automatic build and train a ***3/5 lays convolutional neural network*** 
to classify any kind and any number of objects you want to recognize using tensorflow
and keras based on datasets you provide.  

You need to prepare 3 datasets: ***Train, Test and Validation***.
Each of them should contain several folders and each folder shoud contain a kind of object
you want to classify and the folder will be use as label for images in it. 

## Usage  
### 1. Run directly  
You can run it directly after preparing those datasets, and it will automatically build and train a
***3 lays convolutional neural network*** and a ***5 lays convolutional neural network*** and solve the best model as ***best_model3/5.hdf5***, after finishing, you could either choose some
test images and show the prediction results using ***test_plot*** or plot the Training and validation loss and Training and validation accuracy's trend into line chart using ***evaluate_plot*** or both.
### 2. Use as API
It can also be used as API. There're several functions: ***rename, one_hot, getlabel, flatten, convmodel3, convmodel5, test_plot and evaluate_plot***, most of them will have a introduction in the beginning, you can also use ***main***
as API's example code.

### Hope you have fun with it! If you find any bugs or anything else, please tell me!
