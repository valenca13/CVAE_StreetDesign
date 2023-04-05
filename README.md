Generative deep modelling for street design
================

This page was created to reproduce some of the results and provide the
code used in the paper:

> \[Authors Anonymous\]. The road ahead of using AI for visualizations
> in street design: Learning from failures. Paper submitted to the
> International Journal of Information Management.

-   If any of the material is used, please cite the paper above.

### Motivation:

In this paper we propose the use of *generative deep models* for guiding
public meetings and workshops on street design and street space
allocation projects. As many urban projects are time-consuming and the
communication of innovative projects may not be clear without
visualizations, speeding up this process through AI may ease the
communication, public participation, and understanding of street space
reallocation projects. Using generative deep models to generate various
scenarios of street images by modifying characteristics of the images in
real-time, can be used as a tool to promote discussions on possible
solutions.

These models may have higher importance when explaining and proposing
innovative solutions to the street that are not straightforward to have
a mental map by the public, such as allocating road space dynamically
for different uses, or using technology to manage the space. Generating
various scenarios of a specific street is important because more than
one solution could be applied to the street. Consequently, public
managers may choose one solution for reallocating the space permanently,
or more than one solution, that could be used for changing the space in
different times of the day, week, or season.

### Methodology:

The methodology consists of three main tasks:

-   **Image collection and preprocessing**:

We extracted street images from the Mapillary database. Then performed a
manual filtering process, removing images of low quality. The database
after filtered can be found [here](Database/Images/). To avoid
overfitting, we performed image augmentation.

-   **Object detection/ segmentation**:

We perform object detection in this example, to use the vehicles as
features in the generative deep model. We suggest that future studies
use image segmentation.

-   **Generative deep modelling**:

We tested two types of models. The aim is to generate images conditioned
by specific features (Conditional Variational Autoencoder). However, we
also tried a simpler version of the model, without the features
(Variational Autoencoder) to have clearer conclusions of the outputs.

### Outline:

#### 1. Dataset

1)  The [original dataset](Dataset/Images/) used for
    [Training](Dataset/Images/Train_filtered/) and
    [Test](Dataset/Images/Test_filtered/) filtered manually, without
    augmentation.

2)  Features extracted from object detection in two formats for inputs
    in the Conditional Variational Autoencoder:

-   Dummy variables indicating if objects were present or not in the
    image for [Training](Dataset/Features/Features_Dummy_Train/) and
    [Test](Dataset/Features/Features_Dummy_Test/);

-   Counting the number of each object present in the image for
    [Training](Dataset/Features/Features_Class_Train/) and
    [Test](Dataset/Features/Features_Class_Test).

> **Note**: In the example of the paper we used only the first option as
> inputs for the Conditional Variational Autoencoder. However, we make
> available the code and dataset for both options.

#### 2. Codes

We share the [codes in R](Codes/) used for the preprocessing of the
images (image augmentation and object detection) and also to perform the
variational autoenconder (only images as inputs) and the conditional
variational autoencoder (images and features as inputs).

##### Preprocessing

1.  [Image augmentation](Codes/Image_Augmentation.R)

2.  [Object detection](Codes/Object_Detection.R)

##### Generative Deep Modelling

3.  [Variational Autoencoder](Codes/VAE_StreetDesign.R)

4.  [Conditional Variational
    Autoencoder](Codes/ConditionalVAE_StreetDesign.R)

#### 3. Guidelines

We reproduce some results in RMarkdown in order for an easier
understanding of the steps taken in each task.

##### Preprocessing

1.  [Image Augmentation](Image_Augmentation.md)

2.  [Object Detection](Object_Detection.md)

##### Generative Deep Modelling

3.  [Variational Autoencoder](VAE.md)

4.  [Conditional Variational Autoencoder](CVAE.md)
