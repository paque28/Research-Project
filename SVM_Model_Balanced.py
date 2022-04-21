# -*- coding: utf-8 -*-

import os
import PIL
import cv2
import shutil
import random
import pathlib
import PIL.Image
import numpy as np
import pandas as pd
from joblib import dump, load
import tqdm.notebook as tqdmnb
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score


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
    for ind_im, row in df.iterrows():
        try:
            image = PIL.Image.open(row['name']).convert('L')
            label_age = row['age_group']
            label_gender = row['gender']
            label_hair = row['hair_length']
            image = image.resize((params['HEIGHT'], params['HEIGHT']))
            x = np.expand_dims(np.array(image), axis=0)
            images.append(x.ravel())
            labels_age.append(label_age)
            labels_gender.append(label_gender)
            labels_hair.append(label_hair)
        except:
            print(row['name'])
            pass
    print('End')
    
    return np.squeeze(np.array(images)), np.array(labels_age), np.array(labels_gender), np.array(labels_hair)


data_train = pd.read_csv('path/data_train_hair_length.csv', usecols=['name', 'age', 'gender', 'age_group', 'hair_lenght'])
data_valid = pd.read_csv('path/data_valid_hair_length.csv',  usecols=['name', 'age', 'gender', 'age_group', 'hair_lenght'])
data_test = pd.read_csv('path/data_test_hair_length.csv',  usecols=['name', 'age', 'gender', 'age_group', 'hair_lenght']


%%time
images_train, y_age_train, y_gender_train, y_hair_train = read_dataset(data_train, params)


%%time
images_valid, y_age_valid, y_gender_valid, y_hair_valid = read_dataset(data_valid, params)

%%time
images_test, y_age_test, y_gender_test, y_hair_test = read_dataset(data_test, params)


images_train.shape, y_age_train.shape, y_gender_train.shape, y_hair_train.shape


images_valid.shape, y_age_valid.shape, y_gender_valid.shape, y_hair_valid.shape

images_test.shape, y_age_test.shape, y_gender_test.shape, y_hair_test.shape


# Similar process for gender sets using a binary encoder
LabelBinarizer_gender = preprocessing.LabelBinarizer()
LabelBinarizer_gender.fit(y_gender_train.reshape(-1,1))


print('Categories:')
print(LabelBinarizer_gender.classes_)


y_gender_train_ohc = LabelBinarizer_gender.transform(y_gender_train.reshape(-1,1))
y_gender_valid_ohc = LabelBinarizer_gender.transform(y_gender_valid.reshape(-1,1))
y_gender_test_ohc = LabelBinarizer_gender.transform(y_gender_test.reshape(-1,1))


# Make the pipeline: 1: Standarization and 2. SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC())

 # Balance data
 def create_table_data(y_gender, y_age):
    data_zip = pd.DataFrame(list(zip(y_gender.tolist(), y_age.tolist())), columns=['Gender', 'Age_group'])
    data_zip['count'] = 1
    data = data_zip.pivot_table(index="Gender", columns="Age_group", values="count", aggfunc=np.sum)
    data.columns = [col for col in data.columns]
    data = data.reset_index()
    
    return data_zip, data


# balances the gender class by identifying which gender has fewer number of samples within each group, and then
# makes that the new count for both male and female within that group
# uses undersampling technique to lower the number of the class with the highest number of samples
def balancing_age_groups(groups, pivot, total_zip):
    total_indx = []
    for age_group in groups:
        gender_list = ['female', 'male']
        number_samples = pivot[['Gender', age_group]]
        min_value = number_samples[age_group].min()
        gender = gender_list[np.argmin(number_samples[age_group].values)]

        print('Undersampling {} - No {} to {}'.format(age_group, gender, min_value))
        sample_df = total_zip.loc[total_zip.Age_group==age_group]

        df_min_class_group = sample_df.loc[sample_df.Gender==gender]
        gender_list.pop(gender_list.index(gender))
        df_max_class_group_undersamplig = sample_df.loc[sample_df.Gender==gender_list[0]].sample(n=min_value, 
                                                                                                 random_state=seed)
        indexes_2_select = pd.concat([df_min_class_group,
                                      df_max_class_group_undersamplig]).index.tolist()
        total_indx = total_indx + indexes_2_select
        
    return total_indx


total_zip, train_pivot = create_table_data(y_gender_train, y_age_train)

groups = train_pivot.columns.tolist()[1:]

train_pivot


# check the undersampling process
_, train_pivot_undersampling = create_table_data(y_gender_train[total_indx], 
                                                 y_age_train[total_indx])
train_pivot_undersampling

# bar plot
train_pivot_undersampling.set_index('Gender').plot.bar(figsize=(16,8),
                                                      title='Age group distribution in train');
             
                                    images_train[total_indx].shape, y_gender_train[total_indx].shape, y_age_train[total_indx].shape
                                    
                                    
%%time
# Parameters by default
pipeline.fit(images_train[total_indx], y_gender_train_ohc.ravel()[total_indx])


dump(pipeline, 'gender_model_balanced.joblib')


pipeline = load('path/gender_model_balanced.joblib')
                                    

%%time
y_valid_gender_predicted = pipeline.predict(images_valid)
print(
    f"Classification report for classifier {pipeline}:\n"
    f"{metrics.classification_report(y_gender_valid_ohc.ravel(), y_valid_gender_predicted)}\n"
)


# MCC
matthews_corrcoef(y_gender_valid_ohc.ravel(), 
                          y_valid_gender_predicted)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_gender_valid_ohc.ravel(), y_valid_gender_predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


%%time
y_valid_gender_predicted_bal = pipeline.predict(images_valid[valid_indx])
print(
    f"Classification report for classifier {pipeline}:\n"
    f"{metrics.classification_report(y_gender_valid_ohc.ravel()[valid_indx], y_valid_gender_predicted_bal)}\n"
)

# MCC
matthews_corrcoef(y_gender_valid_ohc.ravel()[valid_indx], 
                          y_valid_gender_predicted_bal)


total_zip_valid, valid_pivot = create_table_data(y_gender_valid, y_age_valid)
groups = valid_pivot.columns.tolist()[1:]

valid_pivot


valid_indx = balancing_age_groups(groups, valid_pivot, total_zip_valid)


%%time
y_valid_gender_predicted_bal = pipeline.predict(images_valid[valid_indx])
print(
    f"Classification report for classifier {pipeline}:\n"
    f"{metrics.classification_report(y_gender_valid_ohc.ravel()[valid_indx], y_valid_gender_predicted_bal)}\n"
)

matthews_corrcoef(y_gender_valid_ohc.ravel()[valid_indx], 
                          y_valid_gender_predicted_bal)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_gender_valid_ohc.ravel()[valid_indx], 
                                                       y_valid_gender_predicted_bal)
disp.figure_.suptitle("Confusion Matrix (Balanced train and valid)")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


# Test predictions
%%time
y_test_gender_predicted = pipeline.predict(images_test)
print(
    f"Classification report for classifier {pipeline}:\n"
    f"{metrics.classification_report(y_gender_test_ohc.ravel(), y_test_gender_predicted)}\n"
)

matthews_corrcoef(y_gender_test_ohc.ravel(), 
                          y_test_gender_predicted)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_gender_test_ohc.ravel(), y_test_gender_predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()


## Error Analysis
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


errors, fem_mas, mas_fem = error_analysis(y_gender_valid_ohc, y_valid_gender_predicted, y_age_valid)

len(errors), len(fem_mas), len(mas_fem)

errors_fem_mas_valid = fem_mas.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_fem_mas_valid


errors_fem_mas_valid.plot.bar(figsize=(10,5), title='Female samples predicted as male');


errors_male_fem_valid = mas_fem.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_male_fem_valid

errors_male_fem_valid.plot.bar(figsize=(10,5),title='Male samples predicted as female');

errors, fem_mas, mas_fem = error_analysis(y_gender_valid_ohc[valid_indx], y_valid_gender_predicted_bal, y_age_valid[valid_indx])

fem_mas.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'}).plot.bar(figsize=(10,5),
                                                                                                                          title='Female samples predicted as male (Balanced valid)');

mas_fem.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'}).plot.bar(figsize=(10,5),
                                                                                                        title='Male samples predicted as female (Balanced valid)');

errors_test, fem_mas_test, mas_fem_test = error_analysis(y_gender_test_ohc, y_test_gender_predicted, y_age_test)

len(errors_test), len(fem_mas_test), len(mas_fem_test)

errors_fem_mas_test = fem_mas_test.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_fem_mas_test

errors_fem_mas_test.plot.bar(figsize=(10,5),title='Female samples predicted as male in test');

errors_male_fem_test = mas_fem_test.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'})
errors_male_fem_test

# errors are more distributed.
errors_male_fem_test.plot.bar(figsize=(10,5),title='Male samples predicted as female in test');

errors_test, fem_mas_test, mas_fem_test = error_analysis(y_gender_test_ohc[test_indx], y_test_gender_predicted_bal, y_age_test[test_indx])

len(errors_test), len(fem_mas_test), len(mas_fem_test)

fem_mas_test.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'}).plot.bar(figsize=(10,5),
                                                                                                                    title='Female samples predicted as male in test (Balanced test)');
    
mas_fem_test.groupby('Age_group').agg({'true_labels': 'count'}).rename(columns={'true_labels':'# errors'}).plot.bar(figsize=(10,5),
                                                                                                                    title='Male samples predicted as female in test (Balanced test)');
    


















