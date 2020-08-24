import os
import sys
import pickle
import numpy as np
from Evaluator import *
import argparse
import json



# Create the parser
my_parser = argparse.ArgumentParser(description='Fake New Detection System')

# Add the arguments
my_parser.add_argument('--news_content',
                       '-nc',
                       type=str,
                       required=True,
                       help='''Enter the news content in the format {'title':news_title,
                       'content': news_content,
                       }
                       ''')
my_parser.add_argument('--model_type',
                       '-mt',
                       type=str,
                       default='nb',
                       help='''Select the primary model from the option:
                       rf for randomforrest model,pa for passiveAggressive model,
                       nb for the MultinormalNB model, lg for the Logistic Regression
                       ''')
my_parser.add_argument('--load_all',
                       '-la',default=False,
                        action='store_true',
                       help='Flag  to indicate if all available models should be loaded')
my_parser.add_argument('--return_all',
                       '-ra',
                       action='store_true',
                       default=False,
                       help='Flag  to indicate if predictions should be made from all available models ')
args = my_parser.parse_args()

model_type = args.model_type
load_all = args.load_all
return_all = args.return_all

news = json.loads(args.news_content)

assert list(news.keys())==['title','content'], '''Error improper format of input Enter the news content in the format {'title':news_title,'content': news_content,} '''


def detecting_fake_news(title,text):    
   #retrieving the best model for prediction call
    #pred_model= PredictionModel(load_all,model_type)
    #call the predict_function
    #prob = load_model.predict_proba([var])
    pred_model= PredictionModel(load_all,model_type)
    if title.strip()!='':
        title_prediction= pred_model.predict(title,text_type='title',return_all_models=return_all)
        
    # Make Prediction Based on only the text
    if text.strip()!='':
        text_prediction= pred_model.predict(text,text_type='text',return_all_models=return_all)
        
    #Make prediction based on the text and title
    if title.strip()!='' and text.strip()!='':
        content_prediction = pred_model.predict(title+' '+text,text_type='content',return_all_models=return_all)
        
    label,prob=getJointLabels([content_prediction,
                                   title_prediction,text_prediction],
                                  content_type='news content')

    return label,prob


if __name__ == '__main__':
    detecting_fake_news(news['title'],news['content'])