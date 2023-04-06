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
library(imager)
library(recolorize)
library(OpenImageR)
library(readxl)
library(listarrays)
```

##### 2. Import datasets

###### Download the dataset and create data set folder

``` r
Train_filtered_files = "https://github.com/valenca13/CVAE_StreetDesign/releases/download/1.0/Train_Images_filtered.zip"
Train_filtered = download.file(Train_filtered_files, destfile = "Dataset/files.zip")
unzip(zipfile = "Dataset/files.zip", exdir = "Dataset/Images")

Test_filtered_files = "https://github.com/valenca13/CVAE_StreetDesign/releases/download/1.0/Test_Images_filtered.zip"
Test_filtered = download.file(Test_filtered_files, destfile = "Dataset/files2.zip")
unzip(zipfile = "Dataset/files2.zip", exdir = "Dataset/Images")
```

###### List files in folder

``` r
files_train <- list.files("Dataset/Images/Train_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

files_test <- list.files("Dataset/Images/Test_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)
```

##### 3. Resize images and assign to a list.

**a) Training data**

``` r
Results_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  Resized <- resizeImage(Image, width = 240, height = 320) #uniform size of images
  Results_train[[i]] <- Resized
}
```

###### Check dimensionality

``` r
dim(Results_train[[2]])
```

    ## [1] 240 320   3

> **Note**: 240(height-row) x 320(width-column) x 3(channels - RGB)

###### Show an example of an image in the training dataset

``` r
imageShow(Results_train[[35]])
```

![](Fig_Guidelines/VAE_unnamed-chunk-6-1.png)<!-- -->

###### Convert list of images into arrays

``` r
img_rows <- 240L
img_cols <- 320L
img_channels <- 3L

n_images_train <- length(Results_train)

train_array <- array(unlist(Results_train), dim=c(n_images_train,img_rows, img_cols, img_channels)) 
dim(train_array)
```

    ## [1] 1265  240  320    3

**b) Test data**

``` r
Results_test <- list()
for(i in seq_along(files_test)){
  Image <- readImage(files_test[i])
  Resized <- resizeImage(Image, width = 240, height = 320)
  Results_test[[i]] <- Resized
}
```

###### Check dimensionality

``` r
dim(Results_test[[2]])
```

    ## [1] 240 320   3

###### Show an example of an image in the test dataset

``` r
imageShow(Results_test[[3]])
```

![](Fig_Guidelines/VAE_unnamed-chunk-10-1.png)<!-- -->

###### Convert list of images into arrays

``` r
n_images_test <- length(Results_test)
test_array <- array(unlist(Results_test), dim=c(n_images_test,img_rows, img_cols, img_channels)) 
```

##### 4. Data preparation

``` r
x_train <- train_array %>% 
  `/` (255) %>% 
  as.array()

x_test <- test_array %>% 
  `/` (255) %>% 
  as.array()
```

> **Note**: keras.Conv2D layers expect input with 4D shape

###### Check dimensionality

``` r
dim(x_train)
```

    ## [1] 1265  240  320    3

``` r
dim(x_test)
```

    ## [1] 243 240 320   3

##### 5. VARIATIONAL AUTOENCODER

**a) Training parameters**

``` r
batch_size <- 25L   # number of training samples used in one iteration
#Note: Popular batch sizes include 32, 64, and 128 samples.

epochs <- 100L  # number of times that the learning algorithm will work through the entire training dataset

original_dim <- c(img_rows, img_cols, img_channels)

filters <- 1L #Detects the patterns on the data.

# Convolution kernel size
num_conv <- 3L
intermediate_dim <- 128L
latent_dim <- 2L
epsilon_std <- 1.0
```

> **Note**: The values of the parameter were modified during the
> experiment. These are just an example. Please check Table 1 of the
> paper.

**b) Model definition**

##### ENCODER

``` r
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
```

###### Flattern the layer

``` r
flat <- layer_flatten(conv_4)
```

###### Define dense layers

``` r
h <- layer_dense(flat, intermediate_dim, activation = "relu") #hidden layer
z_mean <- layer_dense(h, latent_dim, name = 'latent_mu')
z_log_var <- layer_dense(h, latent_dim, name = 'latent_sigma')
```

###### Reparameterization trick

``` r
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
```

> **Note**: The reparameterization trick allows the model to train both
> **mu** and **sigma** of the latent space.

> **Note**: We define a sampling function to sample from the
> distribution. z = z_mean + sigma\*epsilon;  
> where, sigma = exp(log_var/2)

-   z_mean (mu) is a vector that represents the the mean point of the
    distribution;
-   log_var is a vector that represents the logarithm of the variance of
    each dimension;
-   epsilon is a point sampled from the standard normal distribution.

##### LATENT SPACE: Define “z” by creating a sample vector from the latent distribution

``` r
z <- layer_concatenate(list(z_mean, z_log_var)) %>%   
  layer_lambda(sampling)
```

###### Define and summarize the encoder model

``` r
encoder <- keras_model(x, c(z_mean,z_log_var))
summary(encoder)
```

    ## Model: "model"
    ## ________________________________________________________________________________
    ##  Layer (type)             Output Shape      Param #  Connected to               
    ## ================================================================================
    ##  encoder_input (InputLaye  [(None, 240, 320  0       []                         
    ##  r)                       , 3)]                                                 
    ##  conv2d (Conv2D)          (None, 240, 320,  39       ['encoder_input[0][0]']    
    ##                            3)                                                   
    ##  conv2d_1 (Conv2D)        (None, 120, 160,  13       ['conv2d[0][0]']           
    ##                            1)                                                   
    ##  conv2d_2 (Conv2D)        (None, 120, 160,  10       ['conv2d_1[0][0]']         
    ##                            1)                                                   
    ##  conv2d_3 (Conv2D)        (None, 120, 160,  10       ['conv2d_2[0][0]']         
    ##                            1)                                                   
    ##  flatten (Flatten)        (None, 19200)     0        ['conv2d_3[0][0]']         
    ##  dense (Dense)            (None, 128)       2457728  ['flatten[0][0]']          
    ##  latent_mu (Dense)        (None, 2)         258      ['dense[0][0]']            
    ##  latent_sigma (Dense)     (None, 2)         258      ['dense[0][0]']            
    ## ================================================================================
    ## Total params: 2,458,316
    ## Trainable params: 2,458,316
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

##### DECODER

``` r
output_shape <- c(batch_size, 120L, 160L, filters) # For the encoder to have the same dimensions as the decoder

decoder_input <- layer_input(shape = latent_dim, name = 'decoder_input')

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
```

> **Note**: The decoder builds up from the latent space until it
> achieves the shape of the input.

##### c) Define Loss function

``` r
vae_loss <- function(x, x_decoded_mean){
  x <- k_flatten(x)
  x_decoded_mean <- k_flatten(x_decoded_mean)
  recon_loss <- 1.0 * img_rows * img_cols *loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  recon_loss + kl_loss
}

vae <- keras_model(x, x_decoded_mean)
vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
summary(vae)
```

    ## Model: "model_1"
    ## ________________________________________________________________________________
    ##  Layer (type)             Output Shape      Param #  Connected to               
    ## ================================================================================
    ##  encoder_input (InputLaye  [(None, 240, 320  0       []                         
    ##  r)                       , 3)]                                                 
    ##  conv2d (Conv2D)          (None, 240, 320,  39       ['encoder_input[0][0]']    
    ##                            3)                                                   
    ##  conv2d_1 (Conv2D)        (None, 120, 160,  13       ['conv2d[0][0]']           
    ##                            1)                                                   
    ##  conv2d_2 (Conv2D)        (None, 120, 160,  10       ['conv2d_1[0][0]']         
    ##                            1)                                                   
    ##  conv2d_3 (Conv2D)        (None, 120, 160,  10       ['conv2d_2[0][0]']         
    ##                            1)                                                   
    ##  flatten (Flatten)        (None, 19200)     0        ['conv2d_3[0][0]']         
    ##  dense (Dense)            (None, 128)       2457728  ['flatten[0][0]']          
    ##  latent_mu (Dense)        (None, 2)         258      ['dense[0][0]']            
    ##  latent_sigma (Dense)     (None, 2)         258      ['dense[0][0]']            
    ##  concatenate (Concatenate  (None, 4)        0        ['latent_mu[0][0]',        
    ##  )                                                    'latent_sigma[0][0]']     
    ##  lambda (Lambda)          (None, 2)         0        ['concatenate[0][0]']      
    ##  dense_2 (Dense)          (None, 128)       384      ['lambda[0][0]']           
    ##  dense_1 (Dense)          (None, 19200)     2476800  ['dense_2[0][0]']          
    ##  reshape (Reshape)        (None, 120, 160,  0        ['dense_1[0][0]']          
    ##                            1)                                                   
    ##  conv2d_transpose (Conv2D  (None, 120, 160,  10      ['reshape[0][0]']          
    ##  Transpose)                1)                                                   
    ##  conv2d_transpose_1 (Conv  (None, 120, 160,  10      ['conv2d_transpose[0][0]'] 
    ##  2DTranspose)              1)                                                   
    ##  conv2d_transpose_2 (Conv  (None, 240, 320,  30      ['conv2d_transpose_1[0][0]'
    ##  2DTranspose)              3)                        ]                          
    ## ================================================================================
    ## Total params: 4,935,550
    ## Trainable params: 4,935,550
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

##### d) Model training

``` r
vae %>% fit(
  x_train, x_train,
  shuffle = TRUE, 
  epochs = epochs, 
  batch_size = batch_size, 
  validation_data = list(x_test, x_test)
)
```

##### e) Present examples of images generated by the generative model

###### Generated image from the the training database

``` r
imageShow(x_train[1,,,])
```

![](Fig_Guidelines/VAE_unnamed-chunk-24-1.png)<!-- -->

###### Generated image from the test database

``` r
imageShow(x_test[1,,,])
```

![](Fig_Guidelines/VAE_unnamed-chunk-25-1.png)<!-- -->

> **Note**: In this example we are printing images with id=1 of the
> training and test datasets.
