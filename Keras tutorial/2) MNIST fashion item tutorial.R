#####################################################
# Date:      01-03-2019                             #
# Author:    Jeroen Meij                            #
# File:      Keras library tutorial                 #
# Version:   1.0                                    #    
#####################################################


#In this file the keras library is used to model neural networks for image recognition.
#The dataset used is the Fashion MNIST dataset, available in the keras library. 
#For the full tutorial, visit https://tensorflow.rstudio.com/keras/
#########################################################################################
rm(list = ls())
library(keras)
library(tidyr)
library(ggplot2)
fashion_mnist <- dataset_fashion_mnist()



#Make training and testing sets for both the images (24x24 images, with 256 greyscale) 
#And sets for the labels (integers ranging from 0-9, corresponding to itemtype)
#The training sets contains 60,000 images & labels. The test sets contains 10,000 images & labels.
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test


#Generate a vector which corresponds to the labeled data
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

#plot one of the training set images
#choose one of images. This gives a matrix of 28 by 28 filled with greyscale values
image_1 <- as.data.frame(train_images[2582, , ])

#Change the names of the columns to a sequence ranging from 1 to 28
colnames(image_1) <- seq_len(ncol(image_1))

#add a column for the y-axis, which is a vector ranging from 1 to 28 
image_1$y <- seq_len(nrow(image_1))

#Tranfsorm matrix to a long df with 3 columns: x-axis, y-axis and greyscale value
image_1 <- gather(image_1, "x", "value", -y)

#change x-axis class from character to integer to make it useable for the plot
image_1$x <- as.integer(image_1$x)

#Use ggplot to plot the image
ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")


#Turn greyscale levels (values of 0-255) into floats inbetween 0-1
train_images <- train_images / 255
test_images <- test_images / 255


par("mar")

#createa grid to display multiple plots
par(mfcol=c(5,5))

#set margins for the graph where text will be displayed (respectivaly bottom, left, top and right)
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')




#Loop through 25 images, and display these umages in the 5x5 grid just created
for (i in 51:75) { 
  
#choose which specific image in the training graph is selected during the loop
  img <- train_images[i, , ]


#make all images rotate clockwise     
  img <- t(apply(img, 2, rev)) 
  
#provides each image of the loop, without tick marks on the x-axis  
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
#add name of the label for each image       
        main = paste(class_names[train_labels[i] + 1]))
}



############################

#setup empty keras model
model <- keras_model_sequential()

#start adding layers to th model
model %>%

#transform the 2d 28 by 28 images to a long vector of 756 values    
  layer_flatten(input_shape = c(28, 28)) %>%
  
#Generate the middle and final nodes in the model. 
#The middle node will contain 128 different patterns the model finds within the fashion images and their labels 
  layer_dense(units = 128, activation = 'relu') %>%
  
#The final node contains 10 labels and their probability scores 
  layer_dense(units = 10, activation = 'softmax')


#Provide the parameters for the loss function, optimizer and performance metrics
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

#train the model
model %>% fit(train_images, train_labels, epochs = 30)


#Apply the model to the test set and print the performance statistics
score <- model %>% evaluate(test_images, test_labels)
cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")


#make a df of the testset with probabilities related to each of the 10 labels
predictions <- model %>% predict(test_images)

#make a df of the testset stating which label is most likely to be correct
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]




#make another 5x5 grid and display the images of the test set, their predicted label and their actual label.
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 51:75) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}


