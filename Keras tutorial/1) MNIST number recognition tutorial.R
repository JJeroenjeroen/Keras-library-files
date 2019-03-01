#####################################################
# Date:      01-03-2019                             #
# Author:    Jeroen Meij                            #
# File:      Keras library tutorial                 #
# Version:   1.0                                    #    
#####################################################


#In this file the keras library is used to model neural networks for image recognition.
#The dataset used are the numbers from the MNIST dataset, available in the keras library. 
#For the full tutorial, visit https://tensorflow.rstudio.com/keras/
#########################################################################################

#call package and set seed
library(keras)
set.seed(123)

#take the MNIST number dataset
mnist <- dataset_mnist()

#Make training and testing sets of the x data (3-d array (images,width,height) of grayscale values).
x_train <- mnist$train$x
x_test <- mnist$test$x

#Make training and testing sets of the y data (integer vector with values ranging from 0 to 9).
y_train <- mnist$train$y
y_test <- mnist$test$y

#Reshape the image pixel matrix into a long vector (24x 24 becomes rowvector of 784)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))


#Turn greyscale levels (values of 0-255) into floats inbetween 0-1
x_train <- x_train / 255
x_test <- x_test / 255


#turn the actual outcomes into a binary matrix
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


#Create an empty keras model to build on 
model <- keras_model_sequential()

#add layers to the model
model %>% 
  
#this layer gives the length of the INPUT data (784), together with the amount different color values (256)
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 

#This is the second layer in the model, for which the algorithm finds itself chooses what patterns it is looking for  
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%

#this layer gives the length of the OUTPUT data (a value between 0-9). 
#The softmax parameter tells the model to give a probability for each of these 10 values. The highest probability wins
  layer_dense(units = 10, activation = 'softmax')

#Provide the parameters for the loss function, optimizer and performance metrics
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


#Train the model 30 times, each time on a batch of 128 images (vectors and outcomes)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


#Plot a graph of how the model has learned on each new iteration
plot(history)

#test the model on the testsets
model %>% evaluate(x_test, y_test)


