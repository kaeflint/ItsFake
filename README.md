# ItsFake
![Test Image 1](wordclouds.png)
This project employs different machine learning algorithms and nlp techniques to predict the authenticity of any given news articles as either fake or real. 
For now only interactions using the command line is allowed, however the model can be deployed using frameworks including flask (later) etc. 
* The command prompt will request for the news title and news content, makes the prediction as a combination of the predictions based on only the title, only the content and both the title and news content. This allows the systems to employ different classifiers to achieve a better performance. Currently, the overall prediction is based on the majority voting aggregated by a function. 


# Required
1. python 3.6

You install the following  python packages using the command: pip install {the_pack_name}

2. Sklearn (scikit-learn) 
3. numpy 
4. pandas
5. wordcloud 
6. voila 
7. jupyter


# Project Structure
### Project_drawingboardFinal.ipynb
The Jupyter Notebook containing all the steps employed in the development along with discussions at each development step
### pretrained/
Contains all the saved models for each sub-task (i.e title classification, news body classification, and  title + body classification).
### utils/helpers.py
This file contains some utility functions and classes employed. These includes, the BasicPreprocessor class and preprocessPipeLine are the document preprocessing unit and plotConfusionMatrix for plotting the confusion matrix. 
### run.py
This file contains the routine for running the fake dectection in an interactive mode.
### systemcmd.py
Accepts the news title and content on a command line, then returns the predicted labels.
### ApplicationUI.ipynb
Sample user interface consisting of two text field where the user can enter the title and news content. Pressing a button runs the prediction algorithm

# Steps to install and use the program for prediction
1. Clone this repository: git clone https://github.com/kaeflint/ItsFake.git
2. Use the command **cd ItsFake/** Change the directory to the folder containing the program

#### For User Interface Mode:
Voila is leveraged to render the jupyter notebook {ApplicationUI.ipynb} as a webpage. 

Run the command **voila ApplicationUI.ipynb --debug**

####  For Interactive Mode:
Run **python run.py** with the following options
* Make prediction based on all available models:  ***python run.py --load_all  --return_all**
* Make prediction based with a specific model: python run.py --model_type {model_namel}
Select the primary model from the option: rf for randomforest model,pa for passiveAggressive model,nb for the MultinormalNB model, lg for the logisticRegression
for example running ***python run.py --model_type rf** sets the primary model as the random forest model.

####  For Command Mode:
The Command mode accepts one news article at a time. Run **python systemcmd.py --load_all  --return_all --news_content '{'title':put_news_title_here,
'content':put_news_content_here}'***
* Example: python systemcmd.py --load_all --return_all -news_content '{"title":"Eminem Terrified As Daughter Begins Dating Man Raised On His Music",
"content":"ROCHESTER, MI—Hip-hop artist Marshall Mathers, a.k.a. Eminem, said he was left wholly terrified today after meeting his daughter Hailie’s new boyfriend Justin Denham, an 18-year-old who was reportedly raised on the rapper’s music.
Saying he could barely fathom the thought of Hailie, 17, with a man who ever enjoyed listening to, or was inspired by, his often misogynistic and violent lyrics, Eminem, 40, claimed he was disturbed from the second Denham said he was “a huge fan” of all of the rapper’s seven albums.
"}'
