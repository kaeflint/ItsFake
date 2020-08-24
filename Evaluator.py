from __future__ import print_function, division
import os
from joblib import dump, load
from utils.helpers import *
from typing import *
import pickle as pk


class PredictionModel:
    def __makePrediction(self, data, model):
        if type(data) is str:
            data = [data]
        try:
            # Some of the available model dont have the predict_proba method
            prob = model.predict_proba(data)
            label_pred = prob.argmax(-1)
        except:
            prob = ''
            label_pred = model.predict(data)
        label_pred = [s.upper() for s in list(self.labelEncoder.inverse_transform(label_pred))]
        if len(label_pred) == 1:
            label_pred = label_pred[-1]
        return {'probability_scores': prob, 'labels': label_pred}

    def __predictfunct(self, data, task_key, return_all_models=False):
        # This function will be called to make prediction on any given text data
        if not self.load_all:
            return_all_models = False
        if not return_all_models:
            # Make prediction with only the model specified by self.main_model
            prediction = self.__makePrediction(data, self.main_model[task_key])
            prediction['model'] = self.main_model['name']
            return prediction
        else:
            # return the prediction from all available classifiers
            all_predictions = []
            for model in self.all_models:
                prediction = self.__makePrediction(data, model[task_key])
                prediction['model'] = model['name']
                all_predictions.append(prediction)
        return all_predictions

    def __predictWithTitle(self, title, return_all_models=False):
        return self.__predictfunct(data=title, task_key='title_model', return_all_models=return_all_models)

    def __predictWithText(self, text, return_all_models=False):
        return self.__predictfunct(data=text, task_key='text_model', return_all_models=return_all_models)

    def __predictWithTextTitle(self, content, return_all_models=False):
        return self.__predictfunct(data=content, task_key='title_text_model', return_all_models=return_all_models)

    def __init__(self, load_all: bool = False, model_type='nb'):
        self.model_type = model_type
        self.labels = ['Fake', 'Real']
        self.labelEncoder = pk.load(open('pretrained/label_encoder.tp', 'rb'))
        self.load_all = load_all

        self.passive = None
        self.rf = None
        self.nb = None
        self.logistic = None

        if not load_all:
            assert self.model_type.lower() in ['rf', 'pa', 'nb', 'logistic', 'randomforrest', 'passiveaggressive',
                                               'multinormalnb', ], 'Specified model type not available'
        self.__build()

    def __build(self):
        if not self.load_all:
            self.__load_model()
        else:

            self.passive = \
                {'name': 'PassiveAggressive',
                 'text_model': load('pretrained/textclassifiers/pa.joblib'
                                    ),
                 'title_model': load('pretrained/titleclassifiers/pa.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/pa.joblib'
                                          )}
            self.rf = \
                {'text_model': load('pretrained/textclassifiers/rf.joblib'
                                    ),
                 'name': 'RandomForrest',
                 'title_model': load('pretrained/titleclassifiers/rf.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/rf.joblib'
                                          )}
            self.nb = \
                {'text_model': load('pretrained/textclassifiers/nb.joblib'
                                    ),
                 'name': 'MultinormalNB',
                 'title_model': load('pretrained/titleclassifiers/nb.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/nb.joblib'
                                          )}
            self.logistic = \
                {'text_model': load('pretrained/textclassifiers/logistic.joblib'
                                    ),
                 'name': 'Logistic',
                 'title_model': load('pretrained/titleclassifiers/logistic.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/logistic.joblib'
                                          )}
            self.__load_model()
            self.all_models = [self.logistic, self.nb, self.rf, self.passive]
            print('Model(s) loaded')

    def __load_model(self):
        if self.model_type.lower() in ['rf', 'randomforrest']:
            if self.rf:
                self.main_model = self.rf
            else:
                self.main_model = self.rf = \
                {'text_model': load('pretrained/textclassifiers/rf.joblib'
                                    ),
                 'name': 'RandomForrest',
                 'title_model': load('pretrained/titleclassifiers/rf.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/rf.joblib'
                                          )}
        elif self.model_type.lower() in ['pa', 'passive']:
            if self.passive:
                self.main_model = self.passive
            else:
                self.main_model = self.passive = \
                {'name': 'PassiveAggressive',
                 'text_model': load('pretrained/textclassifiers/pa.joblib'
                                    ),
                 'title_model': load('pretrained/titleclassifiers/pa.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/pa.joblib'
                                          )}
        elif self.model_type.lower() in ['logistic']:
            if self.logistic:
                self.main_model = self.logistic
            else:
                self.main_model = self.logistic = \
                {'text_model': load('pretrained/textclassifiers/logistic.joblib'
                                    ),
                 'name': 'Logistic',
                 'title_model': load('pretrained/titleclassifiers/logistic.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/logistic.joblib'
                                          )}
        elif self.model_type.lower() in ['nb', 'multinormalnb']:
            if self.nb:
                self.main_model = self.nb
            else:
                self.main_model = self.nb = \
                {'text_model': load('pretrained/textclassifiers/nb.joblib'
                                    ),
                 'name': 'MultinormalNB',
                 'title_model': load('pretrained/titleclassifiers/nb.joblib'
                                     ),
                 'title_text_model': load('pretrained/documentclassifiers/nb.joblib'
                                          )}

    def predict(self, data, text_type='text',
                return_all_models=False):
        # data: item to be classified
        # text_type: either title, text or article
        # return_all_models: if true then all available calssifiers will be used to predict
        #                     however if self.load_all is false only the indicated by self.model_type is used

        if text_type == 'title':
            return self.__predictWithTitle(data, return_all_models=return_all_models)
        elif text_type == 'text':
            return self.__predictWithText(data, return_all_models=return_all_models)
        else:
            return self.__predictWithTextTitle(data, return_all_models=return_all_models)