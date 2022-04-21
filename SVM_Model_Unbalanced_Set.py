# -*- coding: utf-8 -*-

#This is a Support Vector Machine implementation to predict gender using an unbalance image dataset
#class distribution is 60/40

#The following are libraries and modules needed to run the code

import os
import PIL
import cv2
import shutil
import random
import pathlib
import PIL.Image
import numpy
import pandas
from joblib import dump, load
import tqdm.notebook as tqdmnb
import matplotlib.pyplot as plt

from sklearn import metrics 
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import matthews_corrcoef


seed = 42
random.seed(seed)
np.random.seed(seed)
check_random_state(seed)

params = {
    'HEIGHT': 64,
    'WIDTH' : 64,
    'num_classes' : 2,
    'batch_size' : 8,
}


def read_dataset(df, params):
    images = []
    labels_age = []
    labels_gender = []
    labels_hair = []
    print('Reading ...')
    #iterates dataset by row
    for ind_im, row in df.iterrows():
        try:
            #reads image and transforms to gray scale using convert function, from 3 channels to 1 channel
            image = PIL.Image.open(row['name']).convert('L')
            label_age = row['age_group']
            label_gender = row['gender']
            label_hair = row['hair_length']
            #modifies image dimensions
            image = image.resize((params['HEIGHT'], params['WIDTH']))
            # expands on one dimension
            x = np.expand_dims(np.array(image), axis=0)
            # image is stored as an array, here ravel turns the image into a vector of 4,096 values (64x64)
            # reads everything and will save it in the three data set called images, then save all into arrays 
            # labes_age, label_gender, lable_har
            images.append(x.ravel())
            labels_age.append(label_age)
            labels_gender.append(label_gender)
            labels_hair.append(label_hair)
        except:
            print(row['name'])
            pass
    print('End')
    
    return np.squeeze(np.array(images)), np.array(labels_age), np.array(labels_gender), np.array(labels_hair)


# Data loading
data_train = pd.read_csv('path/data_train_hair_length.csv', usecols=['name', 'age', 'gender', 'age_group', 'hair_lenght'])
data_valid = pd.read_csv('path/data_valid_hair_length.csv',  usecols=['name', 'age', 'gender', 'age_group', 'hair_lenght'])
data_test = pd.read_csv('path/data_test_hair_length.csv',  usecols=['name', 'age', 'gender', 'age_group', 'hair_lenght'])


%%time
images_train, y_age_train, y_gender_train, y_hair_train = read_dataset(data_train, params)

%%time
images_valid, y_age_valid, y_gender_valid, y_hair_valid = read_dataset(data_valid, params)

%%time
images_test, y_age_test, y_gender_test, y_hair_test = read_dataset(data_test, params)

# To verify shape
images_train.shape, y_age_train.shape, y_gender_train.shape, y_hair_train.shape

images_valid.shape, y_age_valid.shape, y_gender_valid.shape, y_hair_valid.shape

images_test.shape, y_age_test.shape, y_gender_test.shape, y_hair_test.shape


# Transforming data

# uses an encoder
# fit is for everything that needs to be preprocessed
# will scan and assign a number to each category
LabelBinarizer_gender = preprocessing.LabelBinarizer()
# if the diminesion is only one vector, has to be reshaped
LabelBinarizer_gender.fit(y_gender_train.reshape(-1,1))


print('Categories:')
print(LabelBinarizer_gender.classes_)


y_gender_train_ohc = LabelBinarizer_gender.transform(y_gender_train.reshape(-1,1))
y_gender_valid_ohc = LabelBinarizer_gender.transform(y_gender_valid.reshape(-1,1))
y_gender_test_ohc = LabelBinarizer_gender.transform(y_gender_test.reshape(-1,1))

#takes gender and age label for each gender, counts number of male and females per age group then show table
def create_table_data(y_gender, y_age):
    data_zip = pd.DataFrame(list(zip(y_gender.tolist(), y_age.tolist())), columns=['Gender', 'Age_group'])
    data_zip['count'] = 1
    data = data_zip.pivot_table(index="Gender", columns="Age_group", values="count", aggfunc=np.sum)
    data.columns = [col for col in data.columns]
    data = data.reset_index()
    
    return data_zip, data

total_zip, train_pivot = create_table_data(y_gender_train, y_age_train)

train_pivot

# plotting distribution
train_pivot.set_index('Gender').plot.bar(figsize=(16,8), title='Age group distribution in train');

train_pivot.set_index('Gender').sum(axis=1) # 1.6 male to 1 female: aprox 40% females and 60% males


## ML pipeline

# creates a pipeline:
# 1: Standarization and 2. SVM classifier
# StandardScaler turns the scale from 0-255 to a smaller scale
pipeline = make_pipeline(StandardScaler(), SVC())


# Parameters by default
pipeline.fit(images_train, y_gender_train_ohc.ravel())

# now that the model has been trained, save the model
# dump is part of the joblib library
# Takes the pipeline (what we are saving) and the path to save it
dump(pipeline, 'gender_model_balanced.joblib')


# makes predictions and saves them in y_valid_gender_predicted
# generate a classification report using original and predicted values
# these are the metrics for validation
%%time
y_valid_gender_predicted = pipeline.predict(images_valid)
print(
    f"Classification report for classifier {pipeline}:\n"
    f"{metrics.classification_report(y_gender_valid_ohc.ravel(), y_valid_gender_predicted)}\n"
)

# MCC metric
matthews_corrcoef(y_gender_valid_ohc.ravel(), 
                          y_valid_gender_predicted)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_gender_valid_ohc.ravel(), y_valid_gender_predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


# Test predictions
%%time
y_test_gender_predicted = pipeline.predict(images_test)
print(
    f"Classification report for classifier {pipeline}:\n"
    f"{metrics.classification_report(y_gender_test_ohc.ravel(), y_test_gender_predicted)}\n"
)

# MCC metric
matthews_corrcoef(y_gender_test_ohc.ravel(), 
                          y_test_gender_predicted)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_gender_test_ohc.ravel(), y_test_gender_predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


## Error Analysis

# compares predicted and real tags and returns the errors
# I have errors in female as male and male as female
# if real tag is 0 then it means the predicted tag was 1 (females as male)
# if real tag is 1, then the predicted tag was 0 (males as female)
def error_analysis(y_gender_ohc, y_gender_predicted, y_age):
    df_prediction = pd.DataFrame(list(zip(y_gender_ohc.ravel(), 
                          y_gender_predicted, 
                          y_age)), columns=['true_labels', 
                                            'predictions', 
                                            'Age_group'])
    errors = df_prediction.loc[df_prediction.true_labels!=df_prediction.predictions]
    fem_mas = errors.loc[(errors.true_labels==0)]
    mas_fem = errors.loc[(errors.true_labels==1)]
    
    return errors, fem_mas, mas_fem


# validation errors
errors, fem_mas, mas_fem = error_analysis(y_gender_valid_ohc, y_valid_gender_predicted, y_age_valid)
 
len(errors), len(fem_mas), len(mas_fem)
 
errors_fem_mas_valid = fem_mas.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_fem_mas_valid

# bar plot
errors_fem_mas_valid.plot.bar(figsize=(10,5), title='Female samples predicted as male');

errors_male_fem_valid = mas_fem.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_male_fem_valid

errors_male_fem_valid.plot.bar(figsize=(10,5),title='Male samples predicted as female');

# Test

errors_test, fem_mas_test, mas_fem_test = error_analysis(y_gender_test_ohc, y_test_gender_predicted, y_age_test)

len(errors_test), len(fem_mas_test), len(mas_fem_test)

errors_fem_mas_test = fem_mas_test.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_fem_mas_test

errors_fem_mas_test.plot.bar(figsize=(10,5),title='Female samples predicted as male in test');

errors_male_fem_test = mas_fem_test.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_male_fem_test

errors_male_fem_test.plot.bar(figsize=(10,5),title='Male samples predicted as female in test');

images_train.shape, y_age_train_ohc.toarray().shape



/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / 

## To find the best parameters using a randomized search, the following code can be used
## This is an alternative for optimization. Execution time is excessive


# Parameters for optimization
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}


## Select the best metrics for the problem
scoring = {"f1-score":"f1", "AUC": "roc_auc", "accuracy-score":"accuracy"}

# For optimization process, these are the parameters
# that would be used several times to obtain the best model
optimization_gender = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    scoring=scoring,
    refit="f1-score",
    return_train_score=True,
    #Controls the number of jobs that get dispatched during execution
    n_jobs=os.cpu_count()-2,
    # cross validation
    cv=2,
    # determines how much progress information is displayed
    verbose=10,
    #based on the parameters and folds if there is an error in the score
    # gives penalty but continues the process
    error_score = np.nan
)

optimization_gender.fit(X_train, y_train)

# extracts best score
print('Best avg f1-score: {}'.format(optimization_gender.best_score_))

# extracts best parameters
print('Best params: {}'.format(optimization_gender.best_params_))

# get the best estimator, best model
pipeline = optimization_gender.best_estimator_

# Use the best estimator to predict with test set
y_pred = best_estimator.predict(X_test)


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()