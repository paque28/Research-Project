# Research-Project
Reduction of Bias with SVM

## These experiments are part of a research project to reduce biases in Machine Learning using Support Vector Machines
## A neural network was also created to add an additional label to a large data set of images.
## The dataset contains biases in gender as well as within age groups
## The distribution of gender is approximately 60/40 with more samples of males than females.


1. Neural network for hair label:

This is a Keras implementation of ResNet-152 with ImageNet pre-trained weights.
The network trains on a new dataset of images from IMDb website and is able to classify those images with a
new tag pertaining to hair length (long, short).

Original paper where ResNet-152 was introduced can be found at https://arxiv.org/abs/1512.03385

Datasets used for this experiment are provided in the repository. Additional question can be sent to paq.1083@mail.com

ResNet-152 is a state-of-the-art model that was trained on a dataset named ImageNet. ImageNet contains 1,000 different classes.

In this experiment, the model is adjusted to handle only two classes (gender,age).

The process with take the following steps:
a. Create base model
b. Make transfer learning
c. Data generators
d. Train new model

The superior layer that does the classifation must be removed from the base model and a new layer must be added
for the classification of the current problem. Once the classification layer is removed, a GlobalAveragePooling can
be used to ensure that the last convolutional layer has the same number of channels as the number of classes for the classifation.
Each channel is averaged into a single value.


Inferior layers that contain information about generic characteristics of the images can be reused. I decided to freeze 100 layers 
but the nubmer of layer to freeze is up to the researcher and it might require some trial and error. Those layers that do not get 
frozen as well as the newly added layer need to be optimized.


Each network from keras contains a functon called preprocess_input which is a function that must be
used before training the model to ensure that the network requirements are met.

By using Dense, it removes a certain percentage of the neurons during each training step. This can lower the variability of the output
of the NN.

I used relu (Rectified Linear Unit) as the activation fuction.

Dropout is a layer that allows me to do regularization by removing a certain percentage of the neurons
during each training step. This is done randomly. Prevents overfitting.


In the transfer learning step I am specifying the prerequesits for the input and output for the network


For the ImageGenerator function, I am giving it the training and testing datasets, width, height and the preprocess_input funcion.

class mode is categorical. It returns a vector with a one-hot encoding representation.

Seed helps get predictable, repeatable results every time.
Setting the seed to some value will generate the same random numbers during multiple executions
of the code.

sort the images

Do the same image generator procedure for test. Test gets the preprocess_input but there is no data augmentation here.

in the callback section, I applied an earlystopping over the validation loss. patience parameter indicates the number of iterations it
will wait to evaluate if model is improving. If no improvement after five then it stops

I defined the optimizer mectrics. Optimizer being used is Adam and it takes the corresponding parameters

I am using binary accuracy and AUC (Area under the curve)
The Area Under the Curve (AUC): is the measure of the ability of a classifier to distinguish between classes and is used as a summary of
the ROC curve. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.

restore_best_weights saves the weights of the best model.

Compile the code using finetune_model.compile(). It takes the optimizer, loss function and chosen metrics.

While training the model with finetune_model.fit(), history holds all the info from the training process.

After executing the train function, the results show how accuracy increases and loss decreases. The results demostrate that learning was successful

For the prediction step:
Datasets for train, validation and test were read.

The prediction function takes the test dataset, class names, the model and the pre-processing function.

I saved the predictions to display the test dataset with the new tag added

This Neural Network succesfuly learned and trained to predict labels for the hair length attribute.

# # # # # # # # # # # # # # # # # # # # # # # # #


2. SVM Implementation with 60/40 distribution of gender class

Afer reading the data and transforming it to the proper dimensions required by scikit learn objects, I created
a table to show how the categories where distribute among males and females.

Created a pipeline
SVC() takes the default parameters that the orginal base model used for parameters C, gamma, and kernel.
However, I implemented a randomized search alternative to find the parameters that produce the best
results with the given parameters. Default parameters were used due to the length of time that it takes
to execute the randomized search for the best model parameters.


predictions were made and for measuring accuracy I used Standard measures as well as Mathew's Correlation Coefficient.


# # # # # # # # # # # # # # # # # # # # # # # # #


3. SVM Implementation with 50/50 distribution of gender class


The same procedure is followed from the SVM implementation of the previous 60/40 distribution.

A function was introduced to balance gender for each age group based on the gender with the fewer samples.
Balancing groups by age was not done since the samples would have been reduced to 83 and this would not be
a good size for the set.

A series of graphs and plots were created to have a better visual of the effects of balancing the classes.

The results showed an increase of errors. However, the error are better distributed between the two gender categories.


