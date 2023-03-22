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



# With TF-2, you can still run this code due to the following line:
#if (tensorflow::tf$executing_eagerly())
  #tensorflow::tf$compat$v1$disable_eager_execution()

#library(keras)
#K <- keras::backend()

#library(keras)
library(tidyverse)
#library(imager)
#library(recolorize)
library(OpenImageR)
#library(readxl)
#library(listarrays)


#IMAGE AUGMENTATION


#train_dir <-  file.path("Train/Images")
#test_dir <-  file.path("Test/Images")

files_train <- list.files("Train/Images",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)
#files_test <- list.files("Test/Images",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

# Resize images

Results_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  Resized <- resizeImage(Image, width = 240, height = 320) #uniform size of images
  Results_train[[i]] <- Resized
}

imageShow(Results_train[[1]])

# Image rotation

Rotation_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  Rotated <- resizeImage(Image, width = 240, height = 320) %>% 
    Augmentation(
      flip_mode = NULL,
      crop_width = NULL,
      crop_height = NULL,
      resiz_width = 0,
      resiz_height = 0,
      resiz_method = "nearest",
      shift_rows = 0,
      shift_cols = 0,
      rotate_angle = 30,
      rotate_method = "nearest",
      zca_comps = 0,
      zca_epsilon = 0,
      image_thresh = 0,
      padded_value = 0,
      verbose = FALSE
    ) 
  Rotation_train[[i]] <- Rotated
  writeImage(Rotated, paste0("Train/Rotated_Images/", i, ".jpg"))
  while (!is.null(dev.list()))  dev.off()
}


imageShow(Rotation_train[[70]])

#Horizontal flip (Mirror)

flip_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  flip <- resizeImage(Image, width = 240, height = 320) %>% 
    Augmentation(
      flip_mode = "horizontal",    #a character string ('horizontal', 'vertical')
      crop_width = NULL,
      crop_height = NULL,
      resiz_width = 0,
      resiz_height = 0,
      resiz_method = "nearest",
      shift_rows = 0,
      shift_cols = 0,
      rotate_angle = 0,
      rotate_method = "nearest",
      zca_comps = 0,
      zca_epsilon = 0,
      image_thresh = 0,
      padded_value = 0,
      verbose = FALSE
    ) 
  flip_train[[i]] <- flip
  #writeImage(flip, paste0("Train/flip_Images/", i, ".jpg"))
  #while (!is.null(dev.list()))  dev.off()
}

imageShow(flip_train[[10]])

#Width shifted (Shift columns)

w_shift_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  wshift <- resizeImage(Image, width = 240, height = 320) %>% 
    Augmentation(
      flip_mode = NULL,    #a character string ('horizontal', 'vertical')
      crop_width = NULL,
      crop_height = NULL,
      resiz_width = 0,
      resiz_height = 0,
      resiz_method = "nearest",
      shift_rows = 0,
      shift_cols = 50,
      rotate_angle = 0,
      rotate_method = "nearest",
      zca_comps = 0,
      zca_epsilon = 0,
      image_thresh = 0,
      padded_value = 0,
      verbose = FALSE
    ) 
  w_shift_train[[i]] <- wshift
  #writeImage(wshift, paste0("Train/wshift_Images/", i, ".jpg"))
  #while (!is.null(dev.list()))  dev.off()
}

imageShow(w_shift_train[[10]])

#height shifted (Shift rows)

h_shift_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  hshift <- resizeImage(Image, width = 240, height = 320) %>% 
    Augmentation(
      flip_mode = NULL,    #a character string ('horizontal', 'vertical')
      crop_width = NULL,
      crop_height = NULL,
      resiz_width = 0,
      resiz_height = 0,
      resiz_method = "nearest",
      shift_rows = 50,
      shift_cols = 0,
      rotate_angle = 0,
      rotate_method = "nearest",
      zca_comps = 0,
      zca_epsilon = 0,
      image_thresh = 0,
      padded_value = 0,
      verbose = FALSE
    ) 
  h_shift_train[[i]] <- hshift
  #writeImage(hshift, paste0("Train/hshift_Images/", i, ".jpg"))
  #while (!is.null(dev.list()))  dev.off()
}

imageShow(h_shift_train[[10]])

# ZCA Whitening

white_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  white <- resizeImage(Image, width = 240, height = 320) %>% 
    ZCAwhiten(30, 0.1) 
  white_train[[i]] <- white
  #writeImage(white, paste0("Train/hshift_Images/", i, ".jpg"))
  #while (!is.null(dev.list()))  dev.off()
}

imageShow(white_train[[10]])

# Edge detection

edgedet_train <- list()
for(i in seq_along(files_train)){
  Image <- readImage(files_train[i]) 
  edge <- resizeImage(Image, width = 240, height = 320) %>% 
    edge_detection(method = 'Prewitt', conv_mode = 'same') #Methods: 'Frei_chen', 'LoG', 'Prewitt', 'Roberts_cross', 'Scharr', 'Sobel'
  edgedet_train[[i]] <- edge
  #writeImage(edge, paste0("Train/edge_Images/", i, ".jpg"))
  #while (!is.null(dev.list()))  dev.off()
}

imageShow(edgedet_train[[20]])
