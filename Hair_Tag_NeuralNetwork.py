# -*- coding: utf-8 -*-

This is an Keras implementation of ResNet-152 originally trained with ImageNet.

import os
import PIL
import cv2
import shutil
import pathlib
import PIL.Image
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm.notebook as tqdmnb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator



params = {
    'HEIGHT': 255,
    'WIDTH' : 255,
    'num_classes' : 2,
    'architecture' : 'ResNet152',
    'final_activation' : 'sigmoid',
    'x_col_name' : 'Name',
    'y_col_name' : 'Hair',
    'kind' : 'categorical',
    'batch_size' : 8,
    'augmenting_factor' : 1,
    'epochs' : 20,
    'lr' : 0.00001,
    'loss' : 'binary_crossentropy',
    'layers_to_freeze': 100
}

kwargs = {}


# defines the new model
def transfer_learning(base_model, num_classes, final_activation, layers_to_freeze):
    
    # ResNet512 has a total of 515 layers
    # I chose not to modify the first 100 layers
    # fine-tunning to optimize the newly added layers and some of the original model structure layers
    
    # layers to freeze, they will not be trained
    for layer in base_model.layers[:layers_to_freeze]:
        layer.trainable = False
    # layers to train
    for layer in base_model.layers[layers_to_freeze:]:
        layer.trainable = True
    
    # x is the output from the original base model
    x = base_model.output
    # x moves forward to the GlobalAveragePooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x keeps propagating as the feed forward with Dense with 1024 neurons and relu as the activation function
    # this is a hidden layer
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # dropout is a layer that allows me to do regularization by removing a certain percentage of the neurons
    # during each training step. This is done randomly. Prevents overfitting
    x = tf.keras.layers.Dropout(0.5)(x)
    # adding classification layer with two classes, sigmoid activation function because there
    # are only two classes.
    predictions = tf.keras.layers.Dense(num_classes, activation=final_activation)(x)
    #instantiating the model with tf.keras.models.Mode to define the input and output layers
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

       
    
    return model 

# multiple model structures to consider
def make_model(architecture, HEIGHT, WIDTH, kwargs=None):
    if architecture == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        base_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(HEIGHT, WIDTH, 3), **kwargs)
    elif architecture == 'InceptionV4':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        base_model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(HEIGHT, WIDTH, 3), **kwargs)
    elif architecture == 'ResNet152':
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        base_model = ResNet152(weights='imagenet',include_top=False,input_shape=(HEIGHT, WIDTH, 3), **kwargs)  
    elif architecture == 'EfficientNetB3':
        from tf.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        base_model = EfficientNetB3(include_top=False,weights='imagenet',input_shape=(HEIGHT, WIDTH, 3),**kwargs)
    elif architecture == 'MobileNet':
        from tf.keras.applications import MobileNet, preprocess_input
        base_model = MobileNet(include_top=False,weights='imagenet',input_shape=(HEIGHT, WIDTH, 3),**kwargs)
        
    else:
        print('Model name error')
    
    return base_model, preprocess_input


# data augmentation
# Returns a train image generator and a test image generator
def make_generator(df_train, df_test, HEIGHT, WIDTH, kind, preprocess_input):
    # Data train generator images
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 rotation_range=40,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.01,
                                 zoom_range=[0.9, 1.25],
                                 horizontal_flip=True,
                                 vertical_flip=False,
                                 fill_mode='reflect',
                                 #specifies filters location
                                 # means that the images should have shape (samples, height, width, channels)
                                 data_format='channels_last'
                                )
    #returns transfored images
    train_generator = datagen.flow_from_dataframe(
                         dataframe=df_train, 
                         x_col=params['x_col_name'],
                         y_col=params['y_col_name'],
                         target_size=(HEIGHT,WIDTH),
                         class_mode=kind, 
                         batch_size=params['batch_size'],
                         seed=seed,
                         shuffle=True)

    # Data test generator images
    test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator=test_datagen.flow_from_dataframe(
                        dataframe=df_test,
                        x_col=params['x_col_name'],
                        y_col=params['y_col_name'],
                        batch_size=params['batch_size'],
                        seed=seed,
                        shuffle=False,
                        class_mode=kind,
                        target_size=(HEIGHT,WIDTH))
    
    return train_generator, test_generator


def train(df_train, df_test, params, **kwargs):
    
    # Creates the base model
    # pass it the architecture parameters
    print('Creating base model ..')
    base_model, preprocess_input = make_model(architecture = params['architecture'], 
                                              HEIGHT = params['HEIGHT'],
                                              WIDTH = params['WIDTH'],
                                              kwargs = kwargs
                                             )
    
    # transfer learning
    # adjusting parameters for my problem
    print('Creating new architecture and tranfer learning process ...')
    finetune_model = transfer_learning(base_model = base_model, 
                                       num_classes = params['num_classes'], 
                                       final_activation = params['final_activation'],
                                       layers_to_freeze = params['layers_to_freeze']
                                      )
    
    print('Creating data generators and augmenting process ...')
    # Creates data generators
    train_generator, test_generator = make_generator(df_train, df_test, 
                                                     HEIGHT = params['HEIGHT'],
                                                     WIDTH = params['WIDTH'],
                                                     kind = params['kind'], 
                                                     preprocess_input = preprocess_input)

    print('Training ...')
    
    # Define iteration steps
    NUM_EPOCHS = params['epochs']
    num_train_images = len(df_train) * params['augmenting_factor']
    
    STEP_SIZE_TRAIN = num_train_images//train_generator.batch_size
    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

    # This callback will stop the training when there is no improvement in
    # the loss for five consecutive epochs.
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                                             patience=5, verbose=1,restore_best_weights=True)
    
    # Defines solver and metrics
    solver = tf.keras.optimizers.Adam(lr = params['lr'])
    metrics = ['binary_accuracy', tf.keras.metrics.AUC(name='auc', multi_label=False)]

    # Model compilation
    finetune_model.compile(solver, loss=params['loss'], metrics=metrics)
    
    #training the model
    #history holds all the info from the training process
    history = finetune_model.fit(
                train_generator,
                epochs = NUM_EPOCHS, 
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_data = test_generator,
                validation_steps = STEP_SIZE_TEST,
                callbacks=[early])
    
    #evalutes the training and generates a final score
    score_train = finetune_model.evaluate(train_generator,verbose=1, steps = STEP_SIZE_TRAIN)
    score_test = finetune_model.evaluate(test_generator,verbose=1, steps = STEP_SIZE_TEST)
    
    return finetune_model, history, train_generator, test_generator, preprocess_input

#generates everything from np and tf.random with seed
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


# Read labeled data 
data_hair = pd.read_csv('hair_data_100.csv', usecols=['Name', 'Hair'])
data_hair

df_train, df_test = train_test_split(data_hair, test_size=0.2, random_state=seed)

# plot bar graph to verify distribution
figure, axx = plt.subplots(1,2, sharey=True, figsize=(10,5))
df_train.groupby('Hair').count().plot(kind='bar', title = 'Train', ax=axx[0]);
df_test.groupby('Hair').count().plot(kind='bar', title = 'Test', ax=axx[1]);


train_generator, test_generator = make_generator(df_train, df_test, 
                                                 HEIGHT = params['HEIGHT'],
                                                 WIDTH = params['WIDTH'],
                                                 kind = params['kind'], 
                                                 preprocess_input = None)
#outpout from previous section
#Found 80 validated image filenames belonging to 2 classes.
#Found 20 validated image filenames belonging to 2 classes.

#training
finetune_model, history, train_generator, test_generator, preprocess_input  = train(df_train, 
                                                                                    df_test, 
                                                                                    params, 
                                                                                    **kwargs)

# plotting accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Making predictions for the rest of the images
# read data
data_train = pd.read_csv('train_set.csv',  usecols=['name', 'age', 'gender', 'age_group'])
data_valid = pd.read_csv('valid_set.csv',  usecols=['name', 'age', 'gender', 'age_group'])
data_test = pd.read_csv('test_set.csv',  usecols=['name', 'age', 'gender', 'age_group'])

def prediction_df(df, class_names, model, preprocess_input):
    predictions = []
    for ind_im, path_image in enumerate(tqdmnb.tqdm(df.name.values)):
        image = PIL.Image.open(path_image)
        image = image.resize((params['HEIGHT'], params['HEIGHT']))
        x = np.array(image)
        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(preprocess_input(x, data_format='channels_last'))
        class_pred = class_names[np.argmax(prediction)]
        predictions.append(class_pred)
    
    df['hair_length'] = predictions
    
    return df


data_test = prediction_df(df=data_test, class_names=class_names,
                          model=finetune_model, preprocess_input=preprocess_input)
data_test
