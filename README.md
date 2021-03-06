
### Table of Contents

1. [Installation](#installation)
2. [Project Description](#description)
3. [Files in the Repository](#files)
4. [Instructions](#Instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The list of commands required to install necessary libraries are given in requirements.txt file.  The code should run with no issues using Python versions 3.*. 

## Project Description<a name="description"></a>

The project is about developing an NLP model to classify disaster response messages into 36 categories using a supervised learning.

1. The cleaned data is saved in sqllite database and then retrieved using sqlalchemy library for training model

2. NLP model is created involving all important steps (Tokenizing, Lemmatizing, Normalizing) of transfroming messages. CountVectorizer, TFIDF Transformer and a custom transformer (Spacy) making use of word embeddings have been used for transformation which is then finally trained using a RandomForestClassifier.

3. A simple Dashboard is made using Flask where the user will enter the message which will then use the NLP model to classify it into listed categories. Along with it Distribution of Messages is created using plotly which is displayed on Dashboard.

4. The Application has a great utility in the sense that it can classify messages in real time using the application which can further be directed to the appropriate team.

## Files in the Repository<a name="files"></a>

    • app
	| - template
	| |- master.html # main page of web app
	| |- go.html # classification result page of web app
	|- run.py # Flask file that runs app
	|- data_visualiation.py #file that builds and saves plotly visualizations
	|- custom_transformer.py #file that has custom transformer (word embeddings) that further used in feature union while building pipeline
    • data
	|- disaster_categories.csv # data to process
	|- disaster_messages.csv # data to process
	|- process_data.py # file that builds ETL pipeline that reads and cleans and saves data into database
	|- DisasterResponse.db # database to save clean data to
    • models
	|- train_classifier.py #file that builds Machine Learning pipeline to train model
	|- classifier.pkl # saved model
	|- custom_transformer.py #file that has custom transformer (word embeddings) that further used in feature union while building pipeline
    • requirements.txt #file that contains list of pre-requisite libraries to be installed
    • README.md


## Instructions <a name="Instructions"></a>

1. Install all the necessary libraries and framework using requirements.txt file
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/ 

## Results<a name="results"></a>

The Precision of the NLP model stands at 0.86, Recall at 0.44 and F1 score of 0.52. It can be improved a lot by incorporating GridSearchCV library for parameter tuning. However, owing to the huge amount of time that it take due ot resource limitation, I have not used it in my model.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The entire project in partial fulfilment towards completing Udacity's Data Science Nanodegree Program.

The videos and lectures guided me to sucessfully build ETL and NLP pipelines.

The idea of custom transformer that uses word embedding as a Feature is taken from below link
https://towardsdatascience.com/the-triune-pipeline-for-three-major-transformers-in-nlp-18c14e20530



