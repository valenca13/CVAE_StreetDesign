# Computer Vision Concepts

# - An image is the combination of red, green and blue channels; 
# - The computer reads the images as numbers (scalars, vectors, arrays or matrices);
# - The channels are converted to three dimensional arrays (row, column, channel);
# - Each cell of the array corresponds to a pixel. Therefore, the arrays dimensions are equal to the resolution;
# - The values stored in each cell of the array represent the intensity of that channel for the corresponding pixel;
# - Therefore, each pixel has three channels. 
# - The maximum value of the "intensity of brightness" in each channel is 255. 
# These values are on a scale of 0 (no light) to 255 (the brightest). 
# Thus, if a pixel is perfectly black, its values would be {Red: 0, Green: 0, Blue: 0}. 
# Conversely, if the pixel is white, their stored values on the arrays would be {R: 255, G: 255, B: 255}
# That is why we divide by 255, for the values to be between 0-1.


#Import Libraries
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


files_train <- list.files("Dataset/Train/Images",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

files_test <- list.files("Dataset/Test/Images",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

## Note: The labels are already in format of dummy variables (One-hot encoding). No need to use "to_categorical" function. 

# Training data
Results_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  Resized <- resizeImage(Image, width = 240, height = 320) #uniform size of images
  Results_train[[i]] <- Resized
}

#Check the number of dimensions
dim(Results_train[[2]])
# 240(height-row) x 320(width-column) x 3(channels - RGB) 


#show an example of an image
imageShow(Results_train[[8]])

# Convert list of images into arrays

img_rows <- 240L
img_cols <- 320L
img_channels <- 3L

n_images_train <- length(Results_train)

train_array <- array(unlist(Results_train), dim=c(n_images_train,img_rows, img_cols, img_channels)) 
dim(train_array)

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

n_images_test <- length(Results_test)
test_array <- array(unlist(Results_test), dim=c(n_images_test,img_rows, img_cols, img_channels)) 

# Data preparation --------------------------------------------------------

library(listarrays)

x_train <- train_array %>% 
  `/` (255) %>% 
  as.array()

# Note: keras.Conv2D layers expect input with 4D shape

dim(x_train)

x_test <- test_array %>% 
  `/` (255) %>% 
  as.array()

# VARIATIONAL AUTOENCODER

# Parameters --------------------------------------------------------------

  ## Traning parameters
batch_size <- 100L   # number of training samples used in one iteration
#Note: Popular batch sizes include 32, 64, and 128 samples.

epochs <- 30L  # number of times that the learning algorithm will work through the entire training dataset

original_dim <- c(img_rows, img_cols, img_channels)

  ## Number of convolutional filters to use
filters <- 1L #Detects the patterns on the data.

  # Convolution kernel size
num_conv <- 3L
latent_dim <- 2L
intermediate_dim <- 128L
epsilon_std <- 1.0


#Model definition --------------------------------------------------------

# ENCODER

x <- layer_input(shape = original_dim, name = 'encoder_input')

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
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d(
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)



#flattern the layer
flat <- layer_flatten(conv_4)

h <- layer_dense(flat, intermediate_dim, activation = "relu") #hidden layer
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

#"layer_concatenate" takes a list of tensors and unction returns a single vector!
#Note: z is the lambda custom layer we are adding from gradient descent calculations

#Define and summarize the encoder model
encoder <- keras_model(x, c(z_mean,z_log_var))
summary(encoder)


#DECODER

output_shape <- c(batch_size, 120L, 160L, filters) # For the encoder to have the same dimensions as the decoder

decoder_input <- layer_input(shape = latent_dim, name = 'decoder_input')
# We need to start with a shape that can be remapped to the original image shape.

decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")
decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])

decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = img_channels,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "sigmoid"
)


hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_mean <- decoder_deconv_3_upsample(deconv_2_decoded)

#Defining decoder separately
hidden_decoded <- decoder_hidden(decoder_input)   
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
generator <- decoder_deconv_3_upsample(deconv_2_decoded)

decoder <- keras_model(decoder_input, generator)
summary(decoder)

# Loss function
## We want to measure how different our normal distribution with parameters mu and log_var is from...
## the standard normal distribution. In this special case, the KL divergence has a closed form. 

vae_loss <- function(x, x_decoded_mean){
  recon_loss <- 1.0 * img_rows * img_cols *mse(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  recon_loss + kl_loss
}

vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)

# Model training ----------------------------------------------------------

vae %>% fit(
  x_train, x_train,
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)

summary(vae)

imageShow(Results_train[[1]])
imageShow(x_train[1,,,])
#-----------------------------------------------------------------------------
 # Using MNIST dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")

vae %>% fit(
  x_train, x_train,
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)
