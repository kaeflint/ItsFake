# Creator: kaeflint 
# Date: 21/08/2020
import re
from typing import *
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder

#Imports to Build the classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# Create a basic preprocessing module to clean our data
class BasicPreprocessor(BaseEstimator, TransformerMixin):
  def __init__(self,lower:bool=True)->None:
    self.lower=lower
    #,remove_stopwords:bool =False,
    #           custom_stopwords:list=[]
    #self.remove_stopwords = remove_stopwords
    #if self.remove_stopwords:
    #  # if there are no custom_stopwords, them use the default nltk english stopwords
    #  self.stopwords= custom_stopwords if len(custom_stopwords)>0 else nltk.corpus.stopwords.words('english')
  
  def __cleanSentence(self,sentence:str)->str:
    sentence= str(sentence)
    if self.lower:
      sentence= sentence.lower()
    # Remove the line-breaks
    sentence=sentence.replace('\n',' ')

    sentence = re.sub(r'\W',' ',sentence)
    #Remove multiple white spaces
    sentence = re.sub('\s+',' ',sentence,flags=re.I).strip()
    return sentence
  def __call__(self,data)->[str,list]:
    # if the type(data) is str then call the __cleanSentence Method else __cleanMultipleSentences
    if type(data)==str:
      return self.__cleanSentence(data)
    else:
      return [self.__cleanSentence(s) for s in data]
  #Just Return self as there is nothing else to do here    
  def fit( self, X, y = None ):
        return self 
  def transform( self, X, y = None ):
    #print(X[:2])
    if type(X)==str:
      return self.__cleanSentence(X)
    else:
      return [self.__cleanSentence(s) for s in X]

# Method to extract entries from the confusion_matrix
def ExtractConfusionMatrixEntries(conf_matrix,model_name):
  tn, fp, fn, tp=conf_matrix.ravel()
  sf = {'Model':model_name,'True_Negative':tn,'False_Positive':fp,
        'False_Negative':fn,'True_Positive':tp}
  return sf
# Function to plot the Confusion matrix
def plotConfusionMatrix(cm,title='Confusion Matrix Plot',axis=None):
      cmap = sns.cubehelix_palette(light=1, as_cmap=True)
      sns.heatmap(cm,ax=axis, cmap=plt.cm.Blues,annot=True,fmt='',annot_kws={"fontsize":18}
      ,xticklabels=['fake','real'],yticklabels=['fake','real'])
      if axis is None:
        plt.title(title)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label',)
      else:
        axis.set_title(title)
        axis.set_ylabel('Actual label')
        axis.set_xlabel('Predicted label',)
      plt.tight_layout(pad=3.0)
      plt.xticks(rotation=90)
      plt.yticks(rotation=90)
      
# Now lets create a single method that can be called with any classifier and vectorizer
def fitEvalModel(X_train,y_train,vectorizer,
                 classifier,preprocessor=None,
                 classifier_id='classifier_x',
                 X_test=[],y_test=[],
                 labelEncoder=None,
                 plot_cm=True):
  '''
  [X_train,y_train]:  the training data
  vectorizer: converts the tokens within the input data to numerical
  classifier: is the classifier employed to train the model
  preprocessor: data preprocessing module. when None, no preprocessing is performed
  classifier_id: will be added to an output dictionary as the value for the model key
  [X_test,y_test]: the test dataset to evaluate the performance of the model
  plot_cm: boolean indicating to control the ploting of the confusion matrix
  '''
  #BasicPreprocessor()
  if preprocessor is not None:
    print(preprocessor)
    pipe_classifier = Pipeline([('data_prep',preprocessor),
                              ('sent_trans',vectorizer), 
                         ('classifier',classifier)])
  else:
    pipe_classifier = Pipeline([
                              ('sent_trans',vectorizer), 
                         ('classifier',classifier)])
  pipe_classifier.fit(X_train,y_train)

  # Check if 
  if len(X_test)>0:
    # Perform the evaluation
    y_pred = pipe_classifier.predict(X_test)
    print(classification_report(y_test,y_pred))

    # Get the Confusion Matrix
    cm= confusion_matrix(labelEncoder.inverse_transform(y_test),
                          labelEncoder.inverse_transform(y_pred),
                        )
    cmatrix=ExtractConfusionMatrixEntries(cm,classifier_id)
    if plot_cm:
      plotConfusionMatrix(cm,title='Confusion Matrix Plot',axis=None)
    return pipe_classifier,y_pred,cmatrix,cm
  else:
    return pipe_classifier

def evalModel(X_test,y_test,pipe_classifier):
    y_pred = pipe_classifier.predict(X_test)
    print(classification_report(y_test,y_pred))

    # Get the Confusion Matrix
    cm= confusion_matrix(labelEncoder.inverse_transform(y_test),
                          labelEncoder.inverse_transform(y_pred),
                        )
    cmatrix=ExtractConfusionMatrixEntries(cm,'')
    plotConfusionMatrix(cm,title='Confusion Matrix Plot',axis=None)

# Alternative preprocessor 
# FunctionTransformer
def cleanSentence(sentence:str,lower:bool =True)->str:
    sentence= str(sentence)
    if lower:
      sentence= sentence.lower()
    # Remove the line-breaks
    sentence=sentence.replace('\n',' ')

    sentence = re.sub(r'\W',' ',sentence)
    #Remove multiple white spaces
    sentence = re.sub('\s+',' ',sentence,flags=re.I).strip()
    return sentence

# Alternative preprocessing Transformer
from sklearn.preprocessing import FunctionTransformer
def preprocessPipeLine(funct,active=True,lower=True):
  def applyOperator(X,active=True,lower=True):
    if active:
      if type(X)==str:
        return funct(X,lower)
      else:
        return [funct(s,lower) for s in X]
    else:
      return X
  return FunctionTransformer(applyOperator,validate=False, kw_args={'lower':lower,'active':active})
  
      #return []
preprocessor = preprocessPipeLine(cleanSentence,)


# Take a look under the hood to identify the words influcing the decision with respect to each class
def InspectClassifierCoef_(classifier,feature_names,n=20):
  try:
    tokens_with_weights = sorted(zip( classifier.coef_[0],feature_names,))
  except:
    try:
      
      tokens_with_weights = sorted(zip( classifier.feature_importances_,feature_names,))
    except:
      return
    pass

  print(f'Top {n} tokens  Influence the classifier decision to predict the fake label: ')
  print(tokens_with_weights[:n])
  print(f'Top {n} tokens  Influence the classifier decision to predict the real label: ')
  print(tokens_with_weights[-n:][::-1])



#Process the labels returned by all available models
# We will use majority voting here
# 
def getJointLabels(predictions,content_type='news content'):
    labels = []
    probs=[]
    if type(predictions)!=list:
        predictions = [predictions]
    for pred in predictions:
        if pred is None:
            continue
        if type(pred) == list:
            for p in pred:
                labels.append(p['labels'])
                if type(p['probability_scores'])!=str:
                    probs.append(p['probability_scores'])
        else:
            labels.append(pred['labels'])
            if type(pred['probability_scores'])!=str:
                    probs.append(pred['probability_scores'])
    joint_label = Counter(labels).most_common(1)[0][0]
    
    print(f"The given {content_type} is: {joint_label}")
    try:
        probs= np.array(probs)
        probs=np.mean(probs,0)
        print(f'Chances of being fake is: {probs[0][0]}')
    except:
        pass
    return joint_label,probs
