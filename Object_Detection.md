Object Detection
================

##### 1. Import libraries

``` r
#remotes::install_github("maju116/platypus")

# With TF-2, you can still run this code due to the following line:
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
library(platypus)
```

    ## Loading required package: tensorflow

    ## Warning: package 'tensorflow' was built under R version 4.2.2

    ## Welceome to platypus!

``` r
library(abind)
#purrr::set_names 
```

##### 2. “You Look Only Once” (YOLO)

``` r
test_yolo <- yolo3(
  net_h = 416, # Input image height. Must be divisible by 32
  net_w = 416, # Input image width. Must be divisible by 32
  grayscale = FALSE, # Images can be loaded as grayscale or RGB
  n_class = 80, # Number of object classes (80 for COCO dataset)
  anchors = coco_anchors)   # Anchor boxes


test_yolo
```

    ## Model: "yolo3"
    ## ________________________________________________________________________________
    ##  Layer (type)         Output Shape   Param #  Connected to           Trainable  
    ## ================================================================================
    ##  input_img (InputLaye  [(None, 416,   0       []                     Y          
    ##  r)                   416, 3)]                                                  
    ##  darknet53 (Functiona  [(None, None,  4062064  ['input_img[0][0]']   Y          
    ##  l)                    None, 256),   0                                          
    ##                        (None, None,                                             
    ##                        None, 512),                                              
    ##                        (None, None,                                             
    ##                        None, 1024)]                                             
    ##  yolo3_conv1 (Functio  (None, 13, 13  1102438  ['darknet53[0][2]']   Y          
    ##  nal)                 , 512)         4                                          
    ##  yolo3_conv2 (Functio  (None, 26, 26  2957312  ['yolo3_conv1[0][0]',  Y         
    ##  nal)                 , 256)                   'darknet53[0][1]']               
    ##  yolo3_conv3 (Functio  (None, 52, 52  741376  ['yolo3_conv2[0][0]',  Y          
    ##  nal)                 , 128)                   'darknet53[0][0]']               
    ##  grid1 (Functional)   (None, 13, 13  4984063  ['yolo3_conv1[0][0]']  Y          
    ##                       , 3, 85)                                                  
    ##  grid2 (Functional)   (None, 26, 26  1312511  ['yolo3_conv2[0][0]']  Y          
    ##                       , 3, 85)                                                  
    ##  grid3 (Functional)   (None, 52, 52  361471   ['yolo3_conv3[0][0]']  Y          
    ##                       , 3, 85)                                                  
    ## ================================================================================
    ## Total params: 62,001,757
    ## Trainable params: 61,949,149
    ## Non-trainable params: 52,608
    ## ________________________________________________________________________________

##### 3. You can now load YOLOv3 Darknet weights trained on COCO dataset. Download pre-trained weights from here and run:

``` r
test_yolo %>% load_darknet_weights("yolov3.weights")
```

##### 4. Calculate predictions for new images

``` r
train_img_paths <- data.frame(list.files("Example_Image",  full.names = TRUE, pattern = ".jpg", all.files = TRUE))

train_imgs <- train_img_paths %>%
  map(~ {
    image_load(., target_size = c(416, 416), grayscale = FALSE) %>%
      image_to_array() %>%
      `/`(255)
  }) %>%
  abind(along=4) %>% #along = 4
  aperm(c(4, 1:3))

train_preds <- test_yolo %>% 
  predict(train_imgs) 
```

> Note: The first element of the train_pred is the number of images

-   The Yolo network has 3 outputs:
    -   For large objects: (1:13, 1:13, 1:3);

    -   For medium objects: (1:26, 1:26, 1:3);

    -   For small objects: (1:52, 1:52, 1:52);
-   The output objects are vectors of length 85.

##### 5.
