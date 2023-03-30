Variational Autoencoder
================

##### 1. Import Libraries

With TF-2, you can still run this code due to the following line:

``` r
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
K <- keras::backend()

library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ## ✔ ggplot2 3.4.0      ✔ purrr   0.3.4 
    ## ✔ tibble  3.1.8      ✔ dplyr   1.0.10
    ## ✔ tidyr   1.2.0      ✔ stringr 1.4.0 
    ## ✔ readr   2.1.2      ✔ forcats 0.5.1

    ## Warning: package 'ggplot2' was built under R version 4.2.2

    ## Warning: package 'dplyr' was built under R version 4.2.2

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(imager)
```

    ## Loading required package: magrittr
    ## 
    ## Attaching package: 'magrittr'
    ## 
    ## The following object is masked from 'package:purrr':
    ## 
    ##     set_names
    ## 
    ## The following object is masked from 'package:tidyr':
    ## 
    ##     extract
    ## 
    ## 
    ## Attaching package: 'imager'
    ## 
    ## The following object is masked from 'package:magrittr':
    ## 
    ##     add
    ## 
    ## The following object is masked from 'package:stringr':
    ## 
    ##     boundary
    ## 
    ## The following object is masked from 'package:tidyr':
    ## 
    ##     fill
    ## 
    ## The following objects are masked from 'package:stats':
    ## 
    ##     convolve, spectrum
    ## 
    ## The following object is masked from 'package:graphics':
    ## 
    ##     frame
    ## 
    ## The following object is masked from 'package:base':
    ## 
    ##     save.image

``` r
library(recolorize)
library(OpenImageR)
```

    ## Warning: package 'OpenImageR' was built under R version 4.2.2

    ## 
    ## Attaching package: 'OpenImageR'
    ## 
    ## The following object is masked from 'package:recolorize':
    ## 
    ##     readImage

``` r
library(readxl)
```

    ## Warning: package 'readxl' was built under R version 4.2.2

##### 2. Import datasets

``` r
files_train <- list.files("Dataset/Train_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

files_test <- list.files("Dataset/Test_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)
```

##### Resize images and assign to a list.

**a) Training data**

``` r
Results_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  Resized <- resizeImage(Image, width = 240, height = 320) #uniform size of images
  Results_train[[i]] <- Resized
}
```

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

    ## Warning in resize_nearest_array(image, width, height): When resizing the 'width'
    ## becomes 239 therefore we have to adjust it to the input width parameter of
    ## 240.000000 !

###### Check the number of dimensions

``` r
dim(Results_train[[2]])
```

    ## [1] 240 320   3

# 240(height-row) x 320(width-column) x 3(channels - RGB)

###### Show an example of an image in training dataset

``` r
imageShow(Results_train[[8]])
```

![](VAE_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

###### Convert list of images into arrays

``` r
img_rows <- 240L
img_cols <- 320L
img_channels <- 3L

n_images_train <- length(Results_train)

train_array <- array(unlist(Results_train), dim=c(n_images_train,img_rows, img_cols, img_channels)) 
dim(train_array)
```

    ## [1] 1269  240  320    3

# Test data

Results_test \<- list() for(i in seq_along(files_test)){ Image \<-
readImage(files_test\[i\]) Resized \<- resizeImage(Image, width = 240,
height = 320) Results_test\[\[i\]\] \<- Resized }

\#Check the number of dimensions dim(Results_test\[\[2\]\])

\#show images imageShow(Results_test\[\[3\]\])

# Convert list of images into arrays

n_images_test \<- length(Results_test) test_array \<-
array(unlist(Results_test), dim=c(n_images_test,img_rows, img_cols,
img_channels))

# Data preparation ——————————————————–

library(listarrays)

x_train \<- train_array %\>% `/` (255) %\>% as.array()

# Note: keras.Conv2D layers expect input with 4D shape

dim(x_train)

x_test \<- test_array %\>% `/` (255) %\>% as.array()

# VARIATIONAL AUTOENCODER

# Parameters ————————————————————–

\## Traning parameters batch_size \<- 100L \# number of training samples
used in one iteration \#Note: Popular batch sizes include 32, 64, and
128 samples.

epochs \<- 30L \# number of times that the learning algorithm will work
through the entire training dataset

original_dim \<- c(img_rows, img_cols, img_channels)

\## Number of convolutional filters to use filters \<- 1L \#Detects the
patterns on the data.

\# Convolution kernel size num_conv \<- 3L latent_dim \<- 2L
intermediate_dim \<- 128L epsilon_std \<- 1.0

\#Model definition ——————————————————–

# ENCODER

x \<- layer_input(shape = original_dim, name = ‘encoder_input’)

conv_1 \<- layer_conv_2d( x, filters = img_channels, kernel_size = c(2L,
2L), strides = c(1L, 1L), padding = “same”, activation = “relu” )

conv_2 \<- layer_conv_2d( conv_1, filters = filters, kernel_size = c(2L,
2L), strides = c(2L, 2L), padding = “same”, activation = “relu” )

conv_3 \<- layer_conv_2d( conv_2, filters = filters, kernel_size =
c(num_conv, num_conv), strides = c(1L, 1L), padding = “same”, activation
= “relu” )

conv_4 \<- layer_conv_2d( conv_3, filters = filters, kernel_size =
c(num_conv, num_conv), strides = c(1L, 1L), padding = “same”, activation
= “relu” )

\#flattern the layer flat \<- layer_flatten(conv_4)

h \<- layer_dense(flat, intermediate_dim, activation = “relu”) \#hidden
layer z_mean \<- layer_dense(h, latent_dim, name = ‘latent_mu’)
z_log_var \<- layer_dense(h, latent_dim, name = ‘latent_sigma’)

# Reparameterization trick

\#We define a sampling function to sample from the distribution. \# z =
z_mean + sigma\*epsilon \#where, sigma = exp(log_var/2) \#z_mean (mu) is
a vector that represents the the mean point of the distribution.
\#log_var is a vector that represents the logarithm of the variance of
each dimension. \#epsilon is a point sampled from the standard normal
distribution

sampling \<- function(arg){ z_mean \<- arg\[, 1:(latent_dim)\] z_log_var
\<- arg\[, (latent_dim + 1):(2 \* latent_dim)\]

epsilon \<- k_random_normal( shape = c(k_shape(z_mean)\[\[1\]\]),
mean=0., stddev=epsilon_std )

z_mean + k_exp(z_log_var/2)\*epsilon }

# Note: This trick allows us to train both mu and sigma in the training process.

# We define the z by creating a sample vector from the latent distribution

z \<- layer_concatenate(list(z_mean, z_log_var)) %\>%  
layer_lambda(sampling)

\#“layer_concatenate” takes a list of tensors and unction returns a
single vector! \#Note: z is the lambda custom layer we are adding from
gradient descent calculations

\#Define and summarize the encoder model encoder \<- keras_model(x,
c(z_mean,z_log_var)) summary(encoder)

\#DECODER

output_shape \<- c(batch_size, 120L, 160L, filters) \# For the encoder
to have the same dimensions as the decoder

decoder_input \<- layer_input(shape = latent_dim, name =
‘decoder_input’) \# We need to start with a shape that can be remapped
to the original image shape.

decoder_upsample \<- layer_dense(units = prod(output_shape\[-1\]),
activation = “relu”) decoder_hidden \<- layer_dense(units =
intermediate_dim, activation = “relu”)

decoder_reshape \<- layer_reshape(target_shape = output_shape\[-1\])

decoder_deconv_1 \<- layer_conv_2d_transpose( filters = filters,
kernel_size = c(num_conv, num_conv), strides = c(1L, 1L), padding =
“same”, activation = “relu” )

decoder_deconv_2 \<- layer_conv_2d_transpose( filters = filters,
kernel_size = c(num_conv, num_conv), strides = c(1L, 1L), padding =
“same”, activation = “relu” )

decoder_deconv_3\_upsample \<- layer_conv_2d_transpose( filters =
img_channels, kernel_size = c(3L, 3L), strides = c(2L, 2L), padding =
“same”, activation = “sigmoid” )

hidden_decoded \<- decoder_hidden(z) up_decoded \<-
decoder_upsample(hidden_decoded) reshape_decoded \<-
decoder_reshape(up_decoded) deconv_1\_decoded \<-
decoder_deconv_1(reshape_decoded) deconv_2\_decoded \<-
decoder_deconv_2(deconv_1\_decoded) x_decoded_mean \<-
decoder_deconv_3\_upsample(deconv_2\_decoded)

\#Defining decoder separately hidden_decoded \<-
decoder_hidden(decoder_input)  
up_decoded \<- decoder_upsample(hidden_decoded) reshape_decoded \<-
decoder_reshape(up_decoded) deconv_1\_decoded \<-
decoder_deconv_1(reshape_decoded) deconv_2\_decoded \<-
decoder_deconv_2(deconv_1\_decoded) generator \<-
decoder_deconv_3\_upsample(deconv_2\_decoded)

decoder \<- keras_model(decoder_input, generator) summary(decoder)

# Loss function

## We want to measure how different our normal distribution with parameters mu and log_var is from…

## the standard normal distribution. In this special case, the KL divergence has a closed form.

vae_loss \<- function(x, x_decoded_mean){ recon_loss \<- 1.0 \* img_rows
\* img_cols *mse(x, x_decoded_mean) kl_loss \<- -0.5*k_mean(1 +
z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
recon_loss + kl_loss }

vae \<- keras_model(x, x_decoded_mean) vae %\>% compile(optimizer =
“rmsprop”, loss = vae_loss) summary(vae)

# Model training ———————————————————-

vae %\>% fit( x_train, x_train, shuffle = TRUE, epochs = epochs,
batch_size = batch_size, validation_data = list(x_test, x_test) )

summary(vae)

imageShow(Results_train\[\[1\]\]) imageShow(x_train\[1,,,\])
\#—————————————————————————–
