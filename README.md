Generative deep modelling for street design
================

This page was created to reproduce some of the results and provide the
code in ***R* programming** used in the paper:

> \[Authors Anonymous\]. The road ahead of using AI for visualizations
> in street design: Learning from failures. Paper submitted to the
> International Journal of Information Management.

If any of the material is used, **please cite the paper above**.

> **Note**: The results from the generative deep modelling were not good
> due to many reasons explained in the paper.

#### Motivation:

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
one solution can be applied to the street. Consequently, public managers
may choose one solution for reallocating the space permanently, or more
than one solution, that could be used for changing the space in
different times of the day, week, or season.

### Methodology:

The methodology consists of three main tasks:

-   **Image collection and preprocessing**:

We extracted street images from the Mapillary database. Then performed a
manual filtering process, removing images of low quality. The database
after filtered can be found [here](Dataset/Images/). To avoid
overfitting, we performed image augmentation to the filtered images.

-   **Object detection/ segmentation**:

In this example, we perform object detection to detect vehicles in the
images, and use them as features in the generative deep model. We
suggest that future studies use image segmentation.

-   **Generative deep modelling**:

We tested two types of models. The aim is to generate images conditioned
by specific features (Conditional Variational Autoencoder). However, we
also tried a simpler version of the model, without the features
(Variational Autoencoder) to have clearer conclusions of the outputs.

### Outline:

#### 1. Dataset

1)  Download the original dataset used for
    [Training](https://github.com/valenca13/CVAE_StreetDesign/releases/download/1.0/Train_Images_filtered.zip)
    and
    [Test](https://github.com/valenca13/CVAE_StreetDesign/releases/download/1.0/Test_Images_filtered.zip)
    after manually filtered, without augmentation.

2)  Download the **features** extracted from object detection in two
    formats for inputs in the Conditional Variational Autoencoder:

-   Dummy variables indicating if objects were present or not in the
    image for
    [Training](https://github.com/valenca13/CVAE_StreetDesign/releases/download/1.0/Features_Dummy_Train.zip)
    and
    [Test](https://github.com/valenca13/CVAE_StreetDesign/releases/download/1.0/Features_Dummy_Test.zip);

-   Counting the number of each object present in the image for
    [Training](Dataset/Features/Features_Class_Train/) and
    [Test](Dataset/Features/Features_Class_Test).

> **Note**: In the example of the paper we used only the first option as
> inputs for the Conditional Variational Autoencoder. However, we share
> the code and dataset of both options.

#### 2. Codes

We share the [codes in *R*](Codes/) used for the preprocessing of the
images (image augmentation and object detection) and also to perform the
variational autoencoder (only images as inputs) and the conditional
variational autoencoder (images and features as inputs).

##### Preprocessing

-   [Image augmentation](Codes/Image_Augmentation.R)

-   [Object detection](Codes/Object_Detection.R)

##### Generative Deep Modelling

-   [Variational Autoencoder](Codes/VAE_StreetDesign.R)

-   [Conditional Variational Autoencoder](Codes/CVAE_StreetDesign.R)

#### 3. Guidelines

We reproduce some of the results in RMarkdown in order for an easier
understanding of the steps taken in each task.

> **Note**: The image augmentation and object detection produced
> meaningful and reliable results because we used a pre-trained models.
> The variational autoencoder and conditional variational autoencoder
> produced less reliable results.

##### Preprocessing

-   [Image Augmentation](Image_Augmentation.md)

-   [Object Detection](Object_Detection.md)

##### Generative Deep Modelling

-   [Variational Autoencoder](VAE.md)

-   [Conditional Variational Autoencoder](CVAE.md)

> **Note**: The parameters and architecture of the model are presented
> in these guidelines only as examples. Please check Table 1 of the
> paper for more detail of the experiments performed.
