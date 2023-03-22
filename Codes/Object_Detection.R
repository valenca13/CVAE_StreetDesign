#OBJECT DETECTION 

# Install package for object detection

#remotes::install_github("maju116/platypus")

# With TF-2, you can still run this code due to the following line:
if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

library(keras)
K <- keras::backend()
library(tidyverse)
library(platypus)
library(abind)
#purrr::set_names 


#Object Detection

#"You Look Only Once" (YOLO)

test_yolo <- yolo3(
  net_h = 416, # Input image height. Must be divisible by 32
  net_w = 416, # Input image width. Must be divisible by 32
  grayscale = FALSE, # Images can be loaded as grayscale or RGB
  n_class = 80, # Number of object classes (80 for COCO dataset)
  anchors = coco_anchors)   # Anchor boxes


test_yolo

#You can now load YOLOv3 Darknet weights trained on COCO dataset. Download pre-trained weights from here and run:

test_yolo %>% load_darknet_weights("yolov3.weights")

#Import images and generate predictions

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


#Note: 
# The first element of the train_pred is the number of images 
# The Yolo network has 3 outputs: 
# - For large objects: (1:13, 1:13, 1:3)
# - For medium objects: (1:26, 1:26, 1:3)
# - For small objects: (1:52, 1:52, 1:52)
# The output objects are vectors of length 85. 

#Transform raw predictions into bounding boxes:

train_boxes <- get_boxes(
  preds = train_preds, # Raw predictions form YOLOv3 model
  anchors = coco_anchors, # Anchor boxes
  labels = coco_labels, # Class labels
  obj_threshold = 0.55, # Object threshold
  nms = TRUE, # Should non-max suppression be applied (ensures that objects are only detected once)
  nms_threshold = 0.6, # Non-max suppression threshold
  correct_hw = FALSE # Should height and width of bounding boxes be corrected to image height and width
)

#Define the labels
labels <- array(train_boxes)

#Label format: xmin; ymin; xmax; ymax; p_obj; label_id; label.
#Each row is a bounding box. 


#Extract the names of the images to match with the labels
images_names = gsub(pattern = "\\.jpg$", "", basename(train_img_paths)) #pattern matching and replacement. Take out the ".jpg" of names

#Tranform the labels in Yolo format and save in the Labels folder with the same name as the respective image.
# In YOLO format, every image in the dataset has a single *.txt file.
# YOLO format = class_id, x_centre, y_centre, width, height

Exportlabels_txt = function(){
  for(i in 1:length(labels)) {
table_yolo = data.frame(labels[i]) %>%
  mutate(class_id = label_id,
         x_centre = mean(c(xmin, xmax)), 
         y_centre = mean(c(ymin, ymax)),
         width = xmax - xmin,
         height = ymax - ymin) %>% 
  select(class_id, x_centre, y_centre, width, height) %>% 
  write.table(file = paste0("Train/Labels/", images_names[i], ".txt"), sep = ",", row.names = FALSE, col.names = FALSE)
}}

#Exportlabels_txt()

# Export labels with a vector demonstrating the classes as categorical.
# Car, bus, truck, person, bicycle  = COCO dataset - label_id [3, 6, 8, 1, 2]

Exportlabels_Class_txt = function(){
  for(i in 1:length(labels)) {
    table_yolo = data.frame(labels[i]) %>%
      mutate(class_id = label_id) %>% 
      select(class_id) %>% 
      write.table(file = paste0("Train/Labels_Class/", images_names[i], ".txt"), sep = ",", row.names = FALSE, col.names = FALSE)
  }}

Exportlabels_Class_txt()

# Export labels - Vector "Yes or no" class (dummy)
  ## [car, bus, truck, person, bicycle]  = COCO - label_id [3, 6, 8, 1, 2]

labels_df <- data.frame(labels) 

mode <- data.frame("car", "bus", "truck", "person")


for(i in 1:length(labels)) {
    mode[i,1] = ifelse(any(labels_df[[1]][[i]]$label_id == 3), 1, 0);
    mode[i,2] = ifelse(any(labels_df[[1]][[i]]$label_id == 6), 1, 0); 
    mode[i,3] = ifelse(any(labels_df[[1]][[i]]$label_id == 8), 1, 0); 
    mode[i,4] = ifelse(any(labels_df[[1]][[i]]$label_id == 1), 1, 0) 
    }

  ## Export excel sheet
library(writexl)
write_xlsx(mode, "Train/Labels_Dummy/Labels_Dummy.xlsx")

#Plot images with the objects detected
plot_boxes(
  images_paths = train_img_paths, # Images paths
  boxes = train_boxes, # Bounding boxes
  correct_hw = TRUE, # Should height and width of bounding boxes be corrected to image height and width
  labels = coco_labels, # Class labels
)

#--------------------------------------------------------------------------------------------
#Test Dataset

# Import test images and generate predictions
test_img_paths <- list.files("Test/Images",  full.names = TRUE, pattern = ".jpg", all.files = TRUE)

test_imgs <- test_img_paths %>%
  map(~ {
    image_load(., target_size = c(416, 416), grayscale = FALSE) %>%
      image_to_array() %>%
      `/`(255)
  }) %>%
  abind(along=4) %>% #along = 4
  aperm(c(4, 1:3))

test_preds <- test_yolo %>% 
  predict(test_imgs) 

#Transform raw predictions into bounding boxes:

test_boxes <- get_boxes(
  preds = test_preds, # Raw predictions form YOLOv3 model
  anchors = coco_anchors, # Anchor boxes
  labels = coco_labels, # Class labels
  obj_threshold = 0.55, # Object threshold
  nms = TRUE, # Should non-max suppression be applied (ensures that objects are only detected once)
  nms_threshold = 0.6, # Non-max suppression threshold
  correct_hw = FALSE # Should height and width of bounding boxes be corrected to image height and width
)


#Define the labels
labels_test <- array(test_boxes)

#Extract the names of the images to match with the labels
images_names_test = gsub(pattern = "\\.jpg$", "", basename(test_img_paths))

#Tranform the labels in Yolo format and save in the Labels folder with the same name as the respective image.
# In YOLO format, every image in the dataset has a single *.txt file.
# YOLO format = class_id, x_centre, y_centre, width, height

Exportlabels_txt_test = function(){
  for(i in 1:length(labels_test)) {
    table_yolo = data.frame(labels[i]) %>%
      mutate(class_id = label_id,
             x_centre = mean(c(xmin, xmax)), 
             y_centre = mean(c(ymin, ymax)),
             width = xmax - xmin,
             height = ymax - ymin) %>% 
      select(class_id, x_centre, y_centre, width, height) %>% 
      write.table(file = paste0("Test/Labels/", images_names[i], ".txt"), sep = ",", row.names = FALSE, col.names = FALSE)
  }}

Exportlabels_txt_test()

# Export labels with a vector demonstrating the classes as categorical. 
# Car, bus, truck, person, bicycle  = COCO dataset - label_id [3, 6, 8, 1, 2]

Exportlabels_Class_txt = function(){
  for(i in 1:length(labels)) {
    table_yolo = data.frame(labels[i]) %>%
      mutate(class_id = label_id) %>% 
      select(class_id) %>% 
      write.table(file = paste0("Test/Labels_Class/", images_names[i], ".txt"), sep = ",", row.names = FALSE, col.names = FALSE)
  }}

Exportlabels_Class_txt()

# Export labels - Vector "Yes or no" class (dummy)
  ## [car, bus, truck, person, bicycle]  = COCO - label_id [3, 6, 8, 1, 2]

labels_df_test <- data.frame(labels_test) 

mode_test <- data.frame("car", "bus", "truck", "person")


for(i in 1:length(labels_test)) {
  mode_test[i,1] = ifelse(any(labels_df_test[[1]][[i]]$label_id == 3), 1, 0);
  mode_test[i,2] = ifelse(any(labels_df_test[[1]][[i]]$label_id == 6), 1, 0); 
  mode_test[i,3] = ifelse(any(labels_df_test[[1]][[i]]$label_id == 8), 1, 0); 
  mode_test[i,4] = ifelse(any(labels_df_test[[1]][[i]]$label_id == 1), 1, 0) 
}

## Export excel sheet

#write_xlsx(mode_test, "Labels_Dummy_test.xlsx")


#Plot images with the objects detected
plot_boxes(
  images_paths = test_img_paths, # Images paths
  boxes = test_boxes, # Bounding boxes
  correct_hw = TRUE, # Should height and width of bounding boxes be corrected to image height and width
  labels = coco_labels, # Class labels
)