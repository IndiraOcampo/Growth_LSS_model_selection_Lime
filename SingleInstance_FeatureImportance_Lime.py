# Local Feature Importance using Lime: https://christophm.github.io/interpretable-ml-book/lime.html
# perturbations are performed around one data instance (an example realization)

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

feature_names = [f"fs8_{i}" for i in range(1, int((len(dataframe.columns))))]

test_df = pd.read_csv('./data/fs8_continue_b_Test_5000.csv')

#Convert TensorFlow datasets to NumPy arrays or pandas DataFrame
X_train = train_dataframe.drop(columns=['target']).values
y_train = train_dataframe['target'].values
print(X_train)

#Study instance for LIME: feature importance
instance_idx = 0  #Index of the instance to explain
instance = test_df.iloc[instance_idx].drop('target').values

#Define a custom predict_proba function for LIME: outputs prob(class=1) & prob(class=0)
def predict_proba(x):
    feature_names = [f'fs8_{i}' for i in range(1, 17)]
    predict_pr = []
    for i in range(len(x)):
        input_test = {name: tf.convert_to_tensor([value]) for name, value in zip(feature_names, x[i])}
        predict_pr_HS = model.predict(input_test)[0][0]
        predict_pr_LCDM = 1 - model.predict(input_test)[0][0]
        predict_pr.append([predict_pr_LCDM,predict_pr_HS])
    return np.array(predict_pr)

#Create a LimeTabularExplainer object
class_names = theory

explainer = lime_tabular.LimeTabularExplainer(X_train,
                                              feature_names=feature_names, class_names=theory, mode="classification", verbose=True)

#Explain the prediction for the first test data point
explanation = explainer.explain_instance(instance, predict_proba, num_features=16)

#Print the explanation
explanation.show_in_notebook()
explanation.save_to_file('lime_1.html')

# Generate the plot using Lime's as_pyplot_figure()
lime_plot = explanation.as_pyplot_figure()

#Save the plot as a pdf
lime_plot.savefig('lime_plot_1.pdf')

true_LCDM = []
true_MoG = []
false_LCDM = []
false_MoG = []
for i in range(len(test_models)):
  if(true_model[i]==pred_model[i] and pred_model[i]==0):
    true_LCDM.append(1)
  if(true_model[i]==pred_model[i] and pred_model[i]==1):
    true_MoG.append(1)
  if(true_model[i]!=pred_model[i] and pred_model[i]==0):
    false_LCDM.append(1)
  if(true_model[i]!=pred_model[i] and pred_model[i]==1):
    false_MoG.append(1)  
lens= [len(true_LCDM),len(true_MoG),len(false_LCDM),len(false_MoG)]
print(lens)
print(sum(lens))

print('     ','LCDM','f(R)')
print('True ',len(true_LCDM)/sum(lens),len(true_MoG)/sum(lens))
print('False',len(false_LCDM)/sum(lens),len(false_MoG)/sum(lens))
print('--------------')
print("Correct prediction: ",(len(true_LCDM)+len(true_MoG))/sum(lens))
print("Wrong prediction  : ",(len(false_LCDM)+len(false_MoG))/sum(lens))



