# Self_assigned_project_language

## This is the repository for my self assigned project in Language Analytics.

### Project Description

This project performs sentiment analysis using two different methods; Textblob and Vader.  After the sentiment analysis has been made and I have labelled my data, I use a deep learning model to perform text classification. The aim is to achieve the greatest precision in predictions on my testdata; to predict whether reviews of Disneyland is of positive or negative sentiment. This will allow me to easily prepare the precision for both of the sentiment analysis methods, to see on which of these the deep learning model performs better text classification for this specific task.  

### Repository Structure

The repository includes three folders:

- in: this folder should contain the data that the code is run on
- out: this folder will contain the results after the code has been run
- src: this folder contains the script of code that has to be run to achieve the results

### Method

For this project I have made one script containing both sentiment analysis methods and the model to train and predict labels with. The script performs the following tasks:

- I first load the data using pandas
- I then process the data using three helping functions: strip_html_tags, remove_accented_chars and pre_process_corpus
- Then, based on the user input from the command line, one of two functions are being called: one for Textblob sentiment analysis or one for Vader sentiment analysis. These functions both perform sentiment analysis on the reviews, and uses the previously defined getAnalysis function to determine whether the sentiment is positive or negative, and labels the reviews. I chose to only work with negative and positive (a binary problem) rather than negative, positive and neutral (a multiclass problem), as there were hardly no neutral reviews (a sentiment score of 0).
- The data is then splitted into negative and positive reviews and balanced, since the dataset otherwise would be extremely unbalanced (with 36422 positive vs 3600 negative for Textblob and 36029 positive vs. 3993 negative for Vader). I balanced it by using the scikit-learn resample function in order to downsample the positive reviews, and then merged them with the negative reviews in a balanced dataframe. 
- I then split my data into train/test and initialise the tensorflow keras tokenizer and use it to the documents
- After this a padding value is set in order to ensure that the documents have the same length. If a document is shorter, we just pad by adding 0's to the end of the document
- I then convert my texts to sequences, set the maximum number of tokens in each document to 1000 and add the padding sequences.
- I then use my label_binarizer on my labels, in order to get 0 and 1 to work with again
- I then clear the session for previous models, define the parameters for the new model, create the model (Sequential) and add layers.
- I then fit/train the model on the training data and evaluate the model
- I then make predictions on the test data, including that I want a 0.5 decission boundary for my predictions
- Lastly I assign the desired labels (positive/negative) and print my classification report to the "out" folder.
    

### Usage

In order to reproduce the results I have gotten (and which can be found in the "out" folder), a few steps has to be followed:

- Install the relevant packages - relevant packages for both scripts can be found in the "requirements.txt" file.
- Make sure to place the script in the "src" folder and the data in the "in" folder. The data used for this project can be accessed on the following page: https://www.kaggle.com/code/ahmedterry/disneyland-reviews-nlp-sentiment-analysis/notebook 
- Run the script from the terminal and remember to pass the required arguments: -ds (dataset) and -sm (sentiment_method) -> Make sure to navigate to the main folder before executing the script - then you just have to type the following in the terminal:

"python src/sentiment_and_class.py -ds {name_of_the_dataset} -sm {sentiment_method}" 

This should give you the same results as I have gotten in the "out" folder.

### Results

As the aim of this project was to compare the two sentiment analysis methods (Textblob and Vader) in order for the best possible text classification, it is interesting to look into the results of my classification reports. Both of the classification reports shows decent results - however it is clear that the text classification based on the labelling using the Vader sentiment analysis has a higher accuracy. With a 93% precision on positive sentiment it performs 7% better than the text classification based on the Textblob sentiment analysis, that has a maximum precision on 86% and a minimum score on 62% accuracy. Compared to this, the lowest accuracy for Vader is 77% which is still pretty good. From this I can conclude that on this specific task the deep learning classification performs better when the sentiment of the data is being analysed using Vader - however it would be interesting to compare this to other datasets, as to see if there's a pattern or not. This could easily be done using this script, as it has a relatively high reproducability, granted that the data used is in the same format and with the same column names.

### Credits 
Inspiration for code was found (amongst multiple other pages) on the following site: 
https://www.kaggle.com/code/ahmedterry/disneyland-reviews-nlp-sentiment-analysis/data 
