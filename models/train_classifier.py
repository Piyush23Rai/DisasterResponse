import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import spacy 
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
from custom_transformer import SpacyVectorTransformer

def load_data(database_filepath):
    
    """
    Returns pandas dataframe read from table
    
    Parameters:
    -----------
        database_filepath : path to database file create in ETL Part
    
    Returns:
    --------
        df : dataframe read from table.
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    
    #reading table created in ETL part into pandas dataframe
    df = pd.read_sql_table('DisasterResponse',engine)
    
    #setting target variables X and Y
    X = df['message'].astype(str)
    Y = df.drop(['id', 'message', 'original','genre'],axis=1)
    
    return X,Y,list(Y.columns)


def tokenize(text):
    
    """
    Function convert sentences into tokens, normalizes case,, strip trailing and preceding
    whitespaces and removes puntuations and stop words
    
    Parameters:
    -----------
        text : message passed to tokenize into individual words
    
    Returns:
    --------
        clean_tokens : clean array of words .
    """
    
    #regular expression to avoid pucntuations or any special character
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    
    #tokenizing text
    tokens = tokenizer.tokenize(text)
    
    #initiating lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iteratating through each token
    clean_tokens = []
    for tok in tokens:
        
        #stop words are irrelevant in this context of classifying response
        if (tok.lower() not in stopwords.words("english")):
            
            # lemmatizing, normalizing case, and removing leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    
    """
    Function builds a pipeline using CountVectorizer and Tfidf transformer and then performs 
    a feature union with a word embedding pipeline created using space library.
     
    TruncateSVD library is used to reduce dimensions created by word embeddings
    
    Returns:
    --------
        pipeline : final machine learning model ready to be trained.
    """
    
    #english trained optimized pipeline for word embedding
    nlp = spacy.load("en_core_web_md")  # this model will give you 300D
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            
            ('embeddings_pipeline', Pipeline([
                ('vect_trans',SpacyVectorTransformer(nlp)),
                ('reduce_dim', TruncatedSVD(50)),
            ])),
            
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__embeddings_pipeline__reduce_dim__n_components':(50,60,70,100,120,130,150)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters,cv=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function prints the Precision, recall and F1 score for the trained model by predicting
    test data values.
    
    Parameters:
    --------
        model : trained model.
        X_test : test data on which prediction has to be made
        Y_test : Actual categories corresponding to X_test data
        category_names : different categories into which disaster response message is classified
    """
  
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Function prints the Precision, recall and F1 score for the trained model by predicting
    test data values.
    
    Parameters:
    --------
        model : trained model.
        model_filepath : pickle file into which model needs to be saved 
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()