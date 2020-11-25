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
***3 lays convolutional neural network*** and a ***5 lays convolutional neural network*** and solve the best model as ***best_model3/5.hdf5***. After finishing, you could either choose some
test images and show the prediction results using ***test_plot*** or plot the Training and validation loss and Training and validation accuracy's trend into line chart using ***evaluate_plot*** or both.
### 2. Use as API
It can also be used as API. There're several functions: ***one_hot, getlabel, flatten, convmodel3, convmodel5, test_plot and evaluate_plot***, most of them will have a introduction in the beginning, you can also use ***main***
as API's example code.

## Comparison
### 1. Original model
Here're two line charts about the trend of loss and accuracy during training process of 3/5 lays convolutional neural network ***(First for 3, second for 5)*** .
![image](https://github.com/XiangkunYe/EC601_Deep-Learning/blob/master/Chart/Original_Chart3.png)
![image](https://github.com/XiangkunYe/EC601_Deep-Learning/blob/master/Chart/Original_Chart5.png)
I used about 700 images of car and cat to train them. As you can see, in this amount of datasets, the 3 lays convolutional neural network perform well while overfitting occured just after several epoches in the 5 lays convolutional neural network model.
### 2. Optimal model
To deal with the overfitting problem, I read several papers and tutorials and find a few approaches:
* Get more training data.
* Reduce the capacity of the network.
* Add weight regularization.
* Add early stop.
* Add dropout.    
  
I choosed to add an early stop and set the patience as 10 (since we only run 50 epoches). Also, I added a dropout in each model (the amount is 0.25 for 3 convolutional neural network and 0.5 for 5) and changed some parameters like learning rate. Here's line charts after change and it performed much better ***(Also first for 3, second for 5)*** .
![image](https://github.com/XiangkunYe/EC601_Deep-Learning/blob/master/Chart/Optimal_Chart3.png)
![image](https://github.com/XiangkunYe/EC601_Deep-Learning/blob/master/Chart/Optimal_Chart5.png)
### 3. Conclusion
After optimization, the 5 lays convolutional neural network perform better and the overfitting problem is less significant even if it still occur. Also, the accuracy arises faster and higher than 3 lays convolutional neural network. The advantages of 3 lays convolutional neural network is that it learns faster and basically no overfitting and more suitable for small dataset. ***So for large dataset, we should use 5 lays convolutional neural network, and for small dataset or in order to prevent overfitting, we could use 3 lays convolutional neural network.***

### Hope you have fun with it! If you find any bugs or anything else, please tell me!
