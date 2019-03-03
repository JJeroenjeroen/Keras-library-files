#####################################################
# Date:      01-03-2019                             #
# Author:    Jeroen Meij                            #
# File:      Keras library tutorial                 #
# Version:   1.0                                    #    
#####################################################


#In this file the keras library is used to model neural networks for text classification.
#The dataset contains the text of 50,000 movie reviews from the Internet Movie Database.
#For the full tutorial, visit https://tensorflow.rstudio.com/keras/
#########################################################################################

rm(list = ls())

#set seed and call packages
set.seed(123)
library(keras)
library(dplyr)
library(ggplot2)
library(purrr)

#call dataset
#num_words = 10000 keeps the top 10,000 most frequently occurring words in the training data. 
#Rare words are discarded to keep the size of the data manageable.
imdb <- dataset_imdb(num_words = 10000)


#specify the training set & labels, and the testing set & labels
#the trainsets are lists of integers each representing a word in the dictionary. 
#the testsets contains lists of binary values representing whether the review was positive or negative
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test

#obtain list of words that represent each integer of the dataset
word_index <- dataset_imdb_word_index()


#turn the list or words into a usaeble long df format
word_index_df <- data.frame(
  word = names(word_index),
  idx = unlist(word_index, use.names = FALSE),
  stringsAsFactors = FALSE
)

# add extra indices to the df indicating specific/special parts within a review
#these indices are already used in the reviews but have not been labelled in the word list yet
word_index_df <- word_index_df %>% mutate(idx = idx + 3)
word_index_df <- word_index_df %>%
  add_row(word = "<PAD>", idx = 0)%>%
  add_row(word = "<START>", idx = 1)%>%
  add_row(word = "<UNK>", idx = 2)%>%
  add_row(word = "<UNUSED>", idx = 3)


#arrange the word index by ID
word_index_df <- word_index_df %>% arrange(idx)

#make function that decodes the integers into words
#map works like lapply, and changes the number into a word
decode_review <- function(text){
  paste(map(text, function(number) word_index_df %>%
              filter(idx == number) %>%
              select(word) %>% 
              pull()),
        collapse = " ")
}


#standardize review size to a max length of 256 words, and reviews with less words are filled to 256
#the padding for sequences than 256 is determined by the "padding" argument 
train_data <- pad_sequences(
  train_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

test_data <- pad_sequences(
  test_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

# input the vocabulary count used for the movie reviews (10,000 words)
vocab_size <- 10000

#create the empty keras model
model <- keras_model_sequential()

#implement layers to the empty model
model %>% 
#first layer turns the vocabulary integers into vectors of fixed size.
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
#second layer returns a fixed-length output vector for each example by averaging over the sequence dimension  
  layer_global_average_pooling_1d() %>%
#third later is a dense layer containing 16 hidden units.  
  layer_dense(units = 16, activation = "relu") %>%
#final layer is the output layer, which provides 1 probabiltiy for the input either being positive or negative  
  layer_dense(units = 1, activation = "sigmoid")

#add performance metrics and loss function to the model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

#Create a validation set by setting apart 10,000 examples from the original training data. 
x_val <- train_data[1:10000, ]
partial_x_train <- train_data[10001:nrow(train_data), ]

y_val <- train_labels[1:10000]
partial_y_train <- train_labels[10001:length(train_labels)]

#train the model using the partial training datasets. 40 times over mini batches of 512 samples of data
#add vallidation data to see at what part the model starts to overfit
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val),
  verbose=1
)

#use the model to test whether it is as accurate on the testset (hence not overfitted)
results <- model %>% evaluate(test_data, test_labels)



