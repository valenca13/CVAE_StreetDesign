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
for different uses, or using technology to manage the space.Generating
various scenarios of a specific street is important because more than
one solution could be applied to the street. Consequently, public
managers may choose one solution for reallocating the space permanently,
or more than one solution, that could be used for changing the space in
different times of the day, week, or season.

### Methodology:

The methodology consists of three main tasks:

-   Image collection and preprocessing:

We extracted street images from the Mapillary database. Then performed a
manual filtering process, removing images of low quality. The database
after filtered can be found [here](%22Database/Images/%22). To avoid
overfitting, we performed image augmentation.

-   object detection/ segmentation:

We perform object detection in this example, to use the vehicles as
features in the generative deep model. We suggest that future studies
use image segmentation.

-   Generative deep modelling:

We tested two types of models. The aim is to generate images conditioned
by specific features (Conditional Variational Autoencoder). However, we
also tried a simpler version of the model, without the features
(Variational Autoencoder) to have clearer conclusions of the outputs.

Outline:

### Guidelines

##### Preprocessing

[1. Image Augmentation](Image_Augmentation.md)

[2. Object Detection](Object_Detection.md)

##### Generative Deep Modelling

[3. Variational Autoencoder](VAE.md)

4.  Conditional Variational Autoencoder
