# Conditional Variational Autoencoder

# With TF-2, you can still run this code due to the following line:
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
K <- keras::backend()

#library(keras)
library(tidyverse)
library(imager)
library(recolorize)
library(OpenImageR)
library(readxl)

files_train <- list.files("Dataset/Images/Train_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)
#files_train_labels <- list.files("Dataset/Features/Features_Class_Train/",  full.names = TRUE, pattern = ".txt", all.files = TRUE)
labels_train <- read_excel('Dataset/Features/Features_Dummy_Train/Labels_Dummy.xlsx')

files_test <- list.files("Dataset/Images/Test_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)
#files_test_labels <- list.files("Dataset/Features/Features_Class_Test/",  full.names = TRUE, pattern = ".txt", all.files = TRUE)
labels_test <- read_excel('Dataset/Features/Features_Dummy_Test/Labels_Dummy_test.xlsx')

## Note: The labels are already in format of dummy variables (One-hot encoding). No need to use "to_categorical" function. 

# Training data
Results_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  Resized <- resizeImage(Image, width = 240, height = 320) #uniform size of images
  Results_train[[i]] <- Resized
}

#Taking out images for trying to fit the model.
#Results_train <- Results_train[-c(11:114)]

# #Convert labels into a "classification". Only images that appear people. 
#labels_train$X.person. <- as.numeric(labels_train$X.person.)
labels_train$X.person. <- as.integer(labels_train$X.person.)
y_train <- data.frame(labels_train[-c(1:3)]) %>% 
  as.matrix()


#Use all the labels in the image. 

#labels_train$X.car. <- as.integer(labels_train$X.car.)
#labels_train$X.bus. <- as.integer(labels_train$X.bus.)
#labels_train$X.truck. <- as.integer(labels_train$X.truck.)


#y_train <- to_categorical(y_t, num_classes = NULL) #%>%
  #as.integer() %>% 
  #array() #%>% 
  #expand_dims(which_dim = 1L)

#y_train <- to_categorical(y_t, num_classes = NULL) #%>%
  #array()
  #as.integer() %>% 
  #array() #%>% 
#expand_dims(which_dim = 1L)




#labels_test$X.car. <- as.integer(labels_test$X.car.)
#labels_test$X.bus. <- as.integer(labels_test$X.bus.)
#labels_test$X.truck. <- as.integer(labels_test$X.truck.)
labels_test$X.person. <- as.integer(labels_test$X.person.)
y_test <- data.frame(labels_test[-c(1:3)]) %>% 
 as.matrix() #%>% 


y_test <- to_categorical(y_te, num_classes = NULL, dtype = "float32") #%>% 

#%>% 
  #as.integer() %>% 
  #array() #%>% 
  #expand_dims(which_dim = 1L)

##Taking out images for trying to fit the model.
#y_train <- array(y_train[-c(11:114),])


#Check the number of dimensions
dim(Results_train[[2]])
# 240(height-row) x 320(width-column) x 3(channels - RGB) 


#show images
imageShow(Results_train[[300]])

# Convert list of images into arrays
#train_array <- array()
#for (i in 1:length(Results_train)){
  #train_array[i] <- image_to_array(Results_train[[i]], data_format = "channels_last")
#}

train_array <- array(unlist(Results_train), dim=c(1265,240,320,3))
dim(train_array)


dim(train_array)
print(train_array[1])
# Test data
Results_test <- list()
for(i in seq_along(files_test)){
  Image <- readImage(files_test[i])
  Resized <- resizeImage(Image, width = 240, height = 320)
  Results_test[[i]] <- Resized
}

#Check the number of dimensions
dim(Results_test[[2]])

#show images
imageShow(Results_test[[3]])

# Convert list of images into arrays
#for (i in seq_along(Results_test)){
  #test_array <- image_to_array(Results_test[[i]], dim(c(240, 320, 3)))
#}

test_array <- array(unlist(Results_test), dim=c(243,240,320,3))

# VARIATIONAL AUTOENCODER

## Input image dimensions

img_rows <- 240L
img_cols <- 320L
img_channels <- 3L # Greyscale = 1 and RGB = 3


# Data preparation --------------------------------------------------------

library(listarrays)

x_train <- train_array %>% 
  `/` (255) #%>% #normalize values of pixels, between (0-1) 
  #array_reshape(c(240, 320, 3)) %>%
  #expand_dims(which_dim = 1L)  #Include batch_size to be able to go in the model

#x_train <- train_array %>% 
  #`/` (255) 

#n_pixels <- prod(240,320,3)
#x_train <-  array_reshape(c(img_channels, n_pixels)) %>%
  #expand_dims(which_dim = 1L)


# Note: keras.Conv2D layers expect input with 4D shape

dim(x_train)

x_test <- test_array %>% 
  `/` (255) # %>% 
  #array_reshape(c(240, 320, 3)) %>%
  #expand_dims(which_dim = 1L) 

#x_test <- test_array %>% 
  #`/` (255)  
#x_test <-  array_reshape(list(img_channels, n_pixels)) %>%
  #expand_dims(which_dim = 1L)
  
# Parameters --------------------------------------------------------------


## Traning parameters
batch_size <- 2L   # number of training samples used in one iteration
#Note: Popular batch sizes include 32, 64, and 128 samples.

epochs <- 50L  # number of times that the learning algorithm will work through the entire training dataset

original_dim <- c(img_rows, img_cols, img_channels)

## Number of convolutional filters to use
filters <- 1L #Detects the patterns on the data.

# Convolution kernel size
num_conv <- 3L

#Dimensionality
intermediate_dim <- 128L
latent_dim <- 2L
epsilon_std <- 1.0

#Length of y_train
x_train_shape <- dim(x_train)[1]


y_train_shape <- dim(y_train)[2] 
#y_train_shape <- col(y_train)

#Model definition --------------------------------------------------------

# ENCODER

x <- layer_input(shape = original_dim, name = 'x')

conv_1 <- layer_conv_2d(
  x,
  filters = img_channels,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_2 <- layer_conv_2d(
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d(
  conv_2,
  filters = filters,
  kernel_size = c(3L, 3L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d(
  conv_3,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)


#flattern the layer to be in same dimension as the label
flat <- layer_flatten(conv_4, input_shape = x_train_shape)

#Label
label <- layer_input(shape = y_train_shape, name = 'label') 



inputs <- layer_concatenate(list(flat,label))

h <- layer_dense(inputs, intermediate_dim, activation = "relu") #hidden layer
z_mean <- layer_dense(h, latent_dim, name = 'latent_mu')
z_log_var <- layer_dense(h, latent_dim, name = 'latent_sigma')

# Reparameterization trick

#We define a sampling function to sample from the distribution.
# z = z_mean + sigma*epsilon
#where, sigma = exp(log_var/2)
#z_mean (mu) is a vector that represents the the mean point of the distribution.
#log_var is a vector that represents the logarithm of the variance of each dimension.
#epsilon is a point sampled from the standard normal distribution

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=epsilon_std
  )
  
  z_mean + k_exp(z_log_var/2)*epsilon
}

# Note: This trick allows us to train both mu and sigma in the training process.

# We define the z by creating a sample vector from the latent distribution
z <- layer_concatenate(list(z_mean, z_log_var)) %>%   
  layer_lambda(sampling)

#Merge latent space with label

zc <- layer_concatenate(list(z, label))

#"layer_concatenate" takes a list of tensors and functions and returns a single vector!
#Note: z is the lambda custom layer we are adding from gradient descent calculations

#Define and summarize the encoder model
encoder <- keras_model(c(x,label),c(z_mean,z_log_var)) #Input and label
summary(encoder)


#DECODER

output_shape <- c(batch_size, 120L, 160L, filters) # For the encoder to have the same dimensions as the decoder

decoder_input <- layer_input(shape = latent_dim+y_train_shape, name = 'decoder_input')
# We need to start with a shape that can be remapped to the original image shape.

decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")
decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])   #output_shape[-1]



decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = img_channels,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "sigmoid"
)


hidden_decoded <- decoder_hidden(zc)   # Z with the label
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
outputs <- decoder_deconv_3_upsample(deconv_2_decoded)



#Defining decoder separately
#hidden_decoded <- decoder_hidden(decoder_input)   # Z with the label
#up_decoded <- decoder_upsample(hidden_decoded)
#reshape_decoded <- decoder_reshape(up_decoded)
#deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
#deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
#generator <- decoder_deconv_3_upsample(deconv_2_decoded)

#decoder <- keras_model(decoder_input, generator)
#summary(decoder)

# Loss function
## We want to measure how different our normal distribution with parameters mu and log_var is from...
## the standard normal distribution. In this special case, the KL divergence has a closed form. 
library(magrittr)

vae_loss <- function(x, outputs){
  xent_loss <- 1.0 * img_rows * img_cols *loss_binary_crossentropy(x, outputs, axis = -1L)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae_loss <- function(x, outputs) {
  x <- k_flatten(x)
  outputs <- k_flatten(outputs)
  xent_loss <- 1.0 * img_rows * img_cols * loss_binary_crossentropy(x, outputs)
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  k_mean(xent_loss + kl_loss)
}


cvae <- keras_model(list(x,label), outputs)
cvae %>% compile(optimizer = "adam", loss = vae_loss)

summary(cvae)

# Model training ----------------------------------------------------------

cvae_final <- fit(cvae,
  list(x = x_train, label = y_train), x_train,
  verbose = 1,
  epochs = epochs, 
  batch_size = batch_size, 
  #validation_data = list(x = x_test, label = y_test), x_test
)

summary(cvae_final)
plot(cvae_final)

imageShow(x_train[3,,,])


