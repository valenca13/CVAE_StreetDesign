
#Import libraries
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

# Parameters --------------------------------------------------------------

batch_size <- 100L
original_dim <- 784L
latent_dim <- 2L
intermediate_dim <- 256L
epochs <- 50L
epsilon_std <- 1.0

# Model definition --------------------------------------------------------

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


vae_loss <- function(x, x_decoded_mean){
  recon_loss <- 1.0 * img_rows * img_cols *mse(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  recon_loss + kl_loss
}

vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)












#x <- layer_input(shape = c(original_dim))
#h <- layer_dense(x, intermediate_dim, activation = "relu")
#z_mean <- layer_dense(h, latent_dim)
#z_log_var <- layer_dense(h, latent_dim)

#sampling <- function(arg){
 # z_mean <- arg[, 1:(latent_dim)]
 # z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
 # epsilon <- k_random_normal(
  #  shape = c(k_shape(z_mean)[[1]]), 
  #  mean=0.,
 #   stddev=epsilon_std
 # )
  
 # z_mean + k_exp(z_log_var/2)*epsilon
#}

# note that "output_shape" isn't necessary with the TensorFlow backend
#z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  #layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
#decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
#decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
#h_decoded <- decoder_h(z)
#x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder
#vae <- keras_model(x, x_decoded_mean)

# encoder, from inputs to latent space
#encoder <- keras_model(x, z_mean)

# generator, from latent space to reconstructed inputs
#decoder_input <- layer_input(shape = latent_dim)
#h_decoded_2 <- decoder_h(decoder_input)
#x_decoded_mean_2 <- decoder_mean(h_decoded_2)
#generator <- keras_model(decoder_input, x_decoded_mean_2)


#vae_loss <- function(x, x_decoded_mean){
  #xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
 # kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
 # xent_loss + kl_loss
#}

#vae %>% compile(optimizer = "rmsprop", loss = vae_loss)


# Data preparation --------------------------------------------------------

mnist <- dataset_mnist()
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 784), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 784), order = "F")


# Model training ----------------------------------------------------------

vae %>% fit(
  x_train, x_train, 
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)

imageShow(x_train[1,])
imageShow(x_test[1,])

# Visualizations ----------------------------------------------------------

library(ggplot2)
library(dplyr)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(mnist$test$y)) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

# display a 2D manifold of the digits
n <- 15  # figure with 15x15 digits
digit_size <- 28

# we will sample n points within [-4, 4] standard deviations
grid_x <- seq(-4, 4, length.out = n)
grid_y <- seq(-4, 4, length.out = n)

rows <- NULL
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    column <- rbind(column, predict(generator, z_sample) %>% matrix(ncol = 28) )
  }
  rows <- cbind(rows, column)
}
rows %>% as.raster() %>% plot()

