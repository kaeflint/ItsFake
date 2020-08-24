# ItsFake
![Test Image 1](wordclouds.png)
This project employs different machine learning algorithms and nlp techniques to predict the authenticity of any given news articles as either fake or real. 

# Required
1. python 3.6

You install the following  python packages using the command: pip install {the_pack_name}

2. Sklearn (scikit-learn) version 0.23.1
3. numpy 
4. pandas
5. wordcloud 



# Project Structure
## Project_drawingboardFinal.ipynb
The Jupyter Notebook containing all the steps employed in the development along with discussions at each development step
## trainedModels/
Contains all the saved models
## utils/helpers.py
This file contains some utility functions and classes employed. These includes, the BasicPreprocessor class and preprocessPipeLine are the document preprocessing unit and plotConfusionMatrix for plotting the confusion matrix. 
## run.py
This file contains the routine for running the fake dectection in an interactive mode.
## systemcmd.py
acccepts the news title and content in a command line, then returns the predicted labels.

# Steps to install and use the program for prediction
1. Clone this repository: git clone 
