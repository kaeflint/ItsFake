# Prepare Model Evaluation Script

#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
from joblib import dump, load
from utils.helpers import *
from typing import *
import pickle as pk
class PredictionModel:
  def __makePrediction(self,data,model):
    if type(data) is str:
      data = [data]
    prob    = model.predict_proba(data)
    label_pred = prob.argmax(-1)
    label_pred = [s.upper() for s in list(self.labelEncoderinverse_transform(label_pred))]
    if len(label_pred)==1:
      label_pred = label_pred[-1]
    return  {'probability_scores':prob,'labels': label_pred}
  def __predictfunct(self,data,task_key,return_all_models=False):
    # This function will be called to make prediction on any given text data
    if not self.load_all:
      return_all_models= False
    if not return_all_models:
        # Make prediction with only the model specified by self.main_model
        prediction = self.__makePrediction(title,self.main_model[task_key])
        prediction['model']= self.main_model['name']
        return prediction
    else:
        # return the prediction from all available classifiers
        all_predictions=[]
        for model in self.all_models:
          prediction = self.__makePrediction(title,model[task_key])
          prediction['model']= model['name']
          all_predictions.append(prediction)
    return all_predictions
  def __predictWithTitle(self,title,return_all_models=False):
    return self.__predictfunct(data=title,task_key='title_model',return_all_models=return_all_models)
      
  def __predictWithText(self,text,return_all_models=False):
    return self.__predictfunct(data=text,task_key='text_model',return_all_models=return_all_models)
  def __predictWithTextTitle(self,content,return_all_models=False):
    return self.__predictfunct(data=content,task_key='title_text_model',return_all_models=return_all_models)
  
  def __init__(self, load_all:bool, model_type='mlp'):
    self.model_type = model_type
    self.labels=['Fake','Real']
    self.labelEncoder = pk.load(open('trainedModels/label_encoder.tp','wb'))
    self.load_all = load_all

    if not load_all:
      assert self.model_type.lower() in ['rf','pa','nb','mlp','randomforrest','passiveaggressive','multinormalnb',], 'Specified model type not available'
  
  def build():
    if not self.load_all:
      self.__load_model()
    else:
      self.passive = \
                {'name':'PassiveAggressive',
                    'text_model': load('trainedModels/pa_model_text.joblib'
                 ),
                 'title_model': load('trainedModels/pa_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/pa_model_title.joblib'
                 )}
      self.rf = \
                {'text_model': load('trainedModels/rf_model_text.joblib'
                 ),
                 'name':'RandomForrest',
                 'title_model': load('trainedModels/rf_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/rf_model_title.joblib'
                 )}
      self.nb = \
                {'text_model': load('trainedModels/nb_model_text.joblib'
                 ),
                 'name':'MultinormalNB',
                 'title_model': load('trainedModels/nb_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/nb_model_title.joblib'
                 )}
      self.mlp = \
                {'text_model': load('trainedModels/mlp_model_text.joblib'
                 ),
                 'name':'MLP',
                 'title_model': load('trainedModels/mlp_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/mlp_model_title.joblib'
                 )}
      self.all_models=[self.mlp,self.nb,self.rf,self.passive]
  
  def __load_model():
    if self.model_type.lower() in ['rf', 'randomforrest']:
      self.main_model = self.rf = \
                {'text_model': load('trainedModels/rf_model_text.joblib'
                 ),
                  'name':'RandomForrest',
                 'title_model': load('trainedModels/rf_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/rf_model_title.joblib'
                 )}
    elif self.model_type.lower() in ['pa', 'passive']:
      self.main_model = self.passive = \
                {'text_model': load('trainedModels/pa_model_text.joblib'
                 ),
                 'name':'PassiveAggressive',
                 'title_model': load('trainedModels/pa_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/pa_model_title.joblib'
                 )}
    elif self.model_type.lower() in ['mlp']:
      self.main_model = self.mlp = \
                {'text_model': load('trainedModels/mlp_model_text.joblib'
                 ),
                 'name':'MLP',
                 'title_model': load('trainedModels/mlp_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/mlp_model_title.joblib'
                 )}
    elif self.model_type.lower() in ['nb', 'multinormalnb']:
      self.main_model = self.nb = \
                {'text_model': load('trainedModels/nb_model_text.joblib'
                 ),
                 'name':'MultinormalNB',
                 'title_model': load('trainedModels/nb_model_title.joblib'
                 ),
                 'title_text_model': load('trainedModels/nb_model_title.joblib'
                 )}
  
  def predict(self,data,text_type='text',
              return_all_models=False):
      # data: item to be classified
      # text_type: either title, text or article
      # return_all_models: if true then all available calssifiers will be used to predict
      #                     however if self.load_all is false only the indicated by self.model_type is used

      if text_type=='title':
        return self.__predictWithTitle(data,return_all_models=return_all_models)
      elif text_type=='text':
        return self.__predictWithText(data,return_all_models=return_all_models)
      else:
        return self.__predictWithTextTitle(data,return_all_models=return_all_models)