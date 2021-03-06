import json
import plotly
import pandas as pd
import custom_transformer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
from data_visualization import return_figures


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #importing figures by calling return_figures function
    graphs = return_figures(df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()