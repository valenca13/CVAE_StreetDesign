Object Detection
================

##### 1. Install libraries

``` r
#remotes::install_github("maju116/platypus")
```

##### 2. Import libraries

``` r
# With TF-2, you can still run this code due to the following line:
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
K <- keras::backend()
library(tidyverse)
library(platypus)
library(abind)
#purrr::set_names 
```

##### 2. Import the “You Look Only Once” (YOLO) model for object detection

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

-   The first element of the train_pred is the number of images.

-   The Yolo network has 3 outputs:

    -   For large objects: (1:13, 1:13, 1:3);

    -   For medium objects: (1:26, 1:26, 1:3);

    -   For small objects: (1:52, 1:52, 1:52);

-   The output objects are vectors of length 85.

##### 3. [Download](https://pjreddie.com/darknet/yolo/) and import YOLOv3 Darknet weights trained on COCO dataset.

``` r
#test_yolo %>% load_darknet_weights("yolov3.weights")
```

> **Note**: The *yolov3.weights* file is too large to be loaded to
> github. Therefore, you should download and run the code with the
> weights.

##### 4. Calculate predictions for the images

``` r
train_img_paths <- list.files("Example_Images",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

#train_img_paths <- list.files("Dataset/Train_filtered",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

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

> **Note**: In this example, we use **small sample** of the original
> database.

> **Note**: The same code is used for training and test data. The
> crucial point that needs to be changed is in importing the files. For
> training data use *“Dataset/Train_filtered”* and for test data use
> *“Dataset/Test_filtered”*. Also we recommend adapting the name of
> variables according to the type of dataset.

##### 5. Transform raw predictions into bounding boxes:

``` r
train_boxes <- get_boxes(
  preds = train_preds, # Raw predictions form YOLOv3 model
  anchors = coco_anchors, # Anchor boxes
  labels = coco_labels, # Class labels
  obj_threshold = 0.55, # Object threshold
  nms = TRUE, # Should non-max suppression be applied (ensures that objects are only detected once)
  nms_threshold = 0.6, # Non-max suppression threshold
  correct_hw = FALSE # Should height and width of bounding boxes be corrected to image height and width
)
```

##### 6. Define the labels

``` r
labels <- array(train_boxes)

head(labels)
```

    ## [[1]]
    ## # A tibble: 6 × 7
    ##    xmin  ymin  xmax  ymax p_obj label_id label        
    ##   <dbl> <dbl> <dbl> <dbl> <dbl>    <int> <chr>        
    ## 1 0.197 0.667 0.211 0.695 0.705        1 person       
    ## 2 0.215 0.666 0.227 0.696 0.754        1 person       
    ## 3 0.231 0.664 0.244 0.698 0.673        1 person       
    ## 4 0.372 0.631 0.566 0.821 0.987        3 car          
    ## 5 0.257 0.626 0.366 0.751 0.997        8 truck        
    ## 6 0.419 0.542 0.425 0.558 0.759       10 traffic light
    ## 
    ## [[2]]
    ## # A tibble: 9 × 7
    ##      xmin  ymin  xmax  ymax p_obj label_id label
    ##     <dbl> <dbl> <dbl> <dbl> <dbl>    <int> <chr>
    ## 1 0.101   0.587 0.275 0.824 0.996        3 car  
    ## 2 0.591   0.606 0.914 0.911 1.00         3 car  
    ## 3 0.485   0.620 0.563 0.699 0.835        3 car  
    ## 4 0.535   0.617 0.640 0.714 0.891        3 car  
    ## 5 0.00278 0.559 0.158 0.922 0.999        3 car  
    ## 6 0.378   0.596 0.450 0.667 0.991        3 car  
    ## 7 0.427   0.601 0.511 0.680 0.837        3 car  
    ## 8 0.452   0.611 0.527 0.680 0.599        3 car  
    ## 9 0.466   0.611 0.556 0.687 0.804        3 car  
    ## 
    ## [[3]]
    ## # A tibble: 11 × 7
    ##      xmin  ymin   xmax  ymax p_obj label_id label 
    ##     <dbl> <dbl>  <dbl> <dbl> <dbl>    <int> <chr> 
    ##  1 0.858  0.624 0.925  0.777 0.969        1 person
    ##  2 0.758  0.606 0.835  0.834 0.920        1 person
    ##  3 0.497  0.648 0.668  0.782 1.00         3 car   
    ##  4 0.191  0.648 0.254  0.691 0.835        3 car   
    ##  5 0.327  0.659 0.353  0.683 0.924        3 car   
    ##  6 0.359  0.658 0.382  0.681 0.872        3 car   
    ##  7 0.0119 0.652 0.0823 0.707 0.958        3 car   
    ##  8 0.0546 0.655 0.116  0.703 0.832        3 car   
    ##  9 0.106  0.657 0.152  0.700 0.581        3 car   
    ## 10 0.150  0.659 0.210  0.697 0.627        3 car   
    ## 11 0.256  0.655 0.313  0.701 0.981        3 car

-   Each row corresponds to an object detected in the image (bounding
    box).

##### 7. Convert the labels into different formats for testing in the Conditional Variational Autoencoder

-   Extract the names of the images to match with the labels

``` r
images_names = gsub(pattern = "\\.jpg$", "", basename(train_img_paths))
```

> Note: In the paper we tested only the format *“c”*.

###### a) Tranform the labels in Yolo format and save in the Labels folder with the same name as the respective image.

``` r
Exportlabels_txt = function(){
  for(i in 1:length(labels)) {
table_yolo = data.frame(labels[i]) %>%
  mutate(class_id = label_id,
         x_centre = mean(c(xmin, xmax)), 
         y_centre = mean(c(ymin, ymax)),
         width = xmax - xmin,
         height = ymax - ymin) %>% 
  select(class_id, x_centre, y_centre, width, height) %>% 
  write.table(file = paste0("Dataset/Features/", images_names[i], ".txt"), sep = ",", row.names = FALSE, col.names = FALSE)   #Put the path of the folder where the images should be saved
  }}

#Exportlabels_txt()
```

-   In YOLO format, every image in the dataset has a single \*.txt file.

-   YOLO format = \[class_id, x_centre, y_centre, width, height\].

> **Note**: Adjust the path where the features should be saved and
> activate the “Exportlabels_txt()” function. This step should also be
> done when exporting other labels.

###### b) Export labels with a single vector presenting the classes

``` r
Exportlabels_Class_txt = function(){
  for(i in 1:length(labels)) {
    table_yolo = data.frame(labels[i]) %>%
      mutate(class_id = label_id) %>% 
      select(class_id) %>% 
      write.table(file = paste0("Train/Labels_Class/", images_names[i], ".txt"), sep = ",", row.names = FALSE, col.names = FALSE) ##Put the path of the folder where the images should be saved
  }}

#Exportlabels_Class_txt()
```

> **Note**: In the COCO dataset that the YOLOv3 was trained, the classes
> *Car*; *bus*; *truck*; *person*; *bicycle*; correspond to the
> following id: \[3, 6, 8, 1, 2\].

###### c) Export labels as a vector with dummy variable analyzing if the class is present or not in the image (dummy variable)

``` r
labels_df <- data.frame(labels) 

mode <- data.frame("car", "bus", "truck", "person")


for(i in 1:length(labels)) {
    mode[i,1] = ifelse(any(labels_df[[1]][[i]]$label_id == 3), 1, 0);
    mode[i,2] = ifelse(any(labels_df[[1]][[i]]$label_id == 6), 1, 0); 
    mode[i,3] = ifelse(any(labels_df[[1]][[i]]$label_id == 8), 1, 0); 
    mode[i,4] = ifelse(any(labels_df[[1]][[i]]$label_id == 1), 1, 0) 
    }
```

-   Export the labels in an Excel sheet to use as an input for the
    Conditional Variational Autoencoder

``` r
library(writexl)
#write_xlsx(mode, "Train/Labels_Dummy/Labels_Dummy.xlsx")
```

##### 8. Plot images with the objects detected

``` r
plot_boxes(
  images_paths = train_img_paths, # Images paths
  boxes = train_boxes, # Bounding boxes
  correct_hw = TRUE, # Should height and width of bounding boxes be corrected to image height and width
  labels = coco_labels, # Class labels
)
```

    ## Warning: The `<scale>` argument of `guides()` cannot be `FALSE`. Use "none" instead as
    ## of ggplot2 3.3.4.
    ## ℹ The deprecated feature was likely used in the platypus package.
    ##   Please report the issue at <https://github.com/maju116/platypus/issues>.

![](Object_Detection_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->![](Object_Detection_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->![](Object_Detection_files/figure-gfm/unnamed-chunk-14-3.png)<!-- -->
