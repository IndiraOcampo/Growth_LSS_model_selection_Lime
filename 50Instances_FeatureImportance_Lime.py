# Local Feature Importance using Lime: https://christophm.github.io/interpretable-ml-book/lime.html
# Perturbations are performed around 50 data instances

import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, callbacks
from matplotlib import pyplot as plt
import lime
import lime.lime_tabular

#Data
dataframe = pd.read_csv('./data/fs8_fR0_Training_5000samples_Lime.csv')
dataframe.shape

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

#Neural Networks training and testing
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):
    #Normalization layer for our features
    normalizer = Normalization()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    #Normalize input features
    encoded_feature = normalizer(feature)
    return encoded_feature

all_inputs = []
all_features_encoded = []

#Inputs and encode numerical features
for i in range(1, int((len(dataframe.columns)))):
    fs_input = keras.Input(shape=(1,), name=f"fs8_{i}")
    fs_encoded = encode_numerical_feature(fs_input, f"fs8_{i}", train_ds)
    all_inputs.extend([fs_input])
    all_features_encoded.extend([fs_encoded])

#Concatenate all encoded features
all_features = layers.concatenate(all_features_encoded)

#Layers
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation="sigmoid")(x)

#The model
model = keras.Model(all_inputs, output)

model.compile(optimizer='nadam', loss="binary_crossentropy", metrics=["accuracy"])

NCC_1701_D=model.fit(train_ds, epochs=1200, validation_data=val_ds)

theory = ['LCDM','HS']

with open('./data/fs8_fR0_Test_5000samples_Lime.csv', 'r') as f:
    dict_reader = csv.DictReader(f,quoting=csv.QUOTE_NONNUMERIC)
    test_models = list(dict_reader)

def norm(p):
  if 0<p<0.5:
    rr=1-p
  else:
    rr=p
  return rr

true_model=[]
pred_model=[]
prob_pred=[]
for i in range(len(test_models)):
  sample = test_models[i]
  true_model.append(round(test_models[i]['target']))
  sample.pop('target', None)
  input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()} 
  predictions = model.predict(input_dict)
  pred_model.append(round(predictions[0][0]))
  prob_pred.append(round(100*norm(predictions[0][0]),3)) #Probability of belonging to class 1: HS

#Aplication of LIME for Local Interpretability: https://christophm.github.io/interpretable-ml-book/lime.html
#------------------------------------------------------------------

from lime.lime_tabular import LimeTabularExplainer
from lime import lime_tabular

# Define the feature names
feature_names = [f"fs8_{i}" for i in range(1, int((len(dataframe.columns))))]

#Convert TensorFlow datasets to NumPy arrays or pandas DataFrame
test_df = pd.read_csv('./data/fs8_fR0_Test_5000_Lime.csv')

X_train = train_dataframe.drop(columns=['target']).values
y_train = train_dataframe['target'].values
print(X_train)

#Define a custom predict_proba function for LIME
def predict_proba(x):
    feature_names = [f'fs8_{i}' for i in range(1, 17)]
    predict_pr = []
    for i in range(len(x)):
        input_test = {name: tf.convert_to_tensor([value]) for name, value in zip(feature_names, x[i])}
        predict_pr_HS = model.predict(input_test)[0][0]
        predict_pr_LCDM = 1 - model.predict(input_test)[0][0]
        predict_pr.append([predict_pr_LCDM,predict_pr_HS])
    return np.array(predict_pr)

intercepts = []
local_predictions = []
Prob_of_pred = []
explanations_data = []
targets = []

#Create a LimeTabularExplainer object
class_names = theory

explainer = lime_tabular.LimeTabularExplainer(X_train,
                                              feature_names=feature_names, class_names=theory, mode="classification", verbose=True)

for i in range(0,50):
    instance_idx = i  #Index of the instance to explain (we chose 50 to mitigate the neighbourhood problem)
    print(test_df.iloc[instance_idx]['target'])
    targets.append(test_df.iloc[instance_idx]['target'])
    instance = test_df.iloc[instance_idx].drop('target').values #instance without target

    #Explain the prediction for the first test data point
    explanation = explainer.explain_instance(instance, predict_proba, num_samples= 2000, num_features=16)

    #explanation.show_in_notebook() #this option shows the explanation as a html figure
    explanation.save_to_file('./figures/Limeplotest_%d.html' %i)
    explanations_data.append(explanation.as_list())
    intercepts.append(explanation.intercept) #intercept
    local_predictions.append(explanation.local_pred[0]) #local prediction
    Prob_of_pred.append(explanation.predict_proba) #probability of belonging to class LCDM or HS

#Save the LIME data:
import csv
header=["Exp_fs8_1", "Exp_fs8_2", "Exp_fs8_3", "Exp_fs8_4", "Exp_fs8_5", "Exp_fs8_6", "Exp_fs8_7", "Exp_fs8_8", "Exp_fs8_9", "Exp_fs8_10", "Exp_fs8_11", "Exp_fs8_12", "Exp_fs8_13", "Exp_fs8_14", "Exp_fs8_15", "Exp_fs8_16", "Class"]
with open('./data/50_instances_LimeExplanations.csv', 'w', newline= '') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)
    csv_writer.writerow(Prob_of_pred)
    csv_writer.writerow(explanations_data)
    csv_writer.writerow(targets)

csvfile.close()