import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    food_water_count = len(df[(df['water'] == 1) | (df['food'] == 1)])
    electricity_shelter_count = len(df[(df['electricity'] == 1) | (df['shelter'] == 1)])
    food_shelt_counts = [food_water_count ,electricity_shelter_count]
    food_shelt_names = ['Food and Water', 'Electricity and Shelter']
    
    search_rescue_count = len(df[(df['search_and_rescue'] == 1)])
    medical_help_count = len(df[(df['medical_help'] == 1)])
    search_medical_counts = [search_rescue_count, medical_help_count]
    search_medical_names = ['Search and Rescue', ' Medical Help']
    

    graphs = [
                {
            'data': [
                Bar(
                    x=food_shelt_names,
                    y=food_shelt_counts
                )
            ],

            'layout': {
                'title': 'People Needing Food and Water or Electricity and Shelter',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=search_medical_names,
                    y=search_medical_counts
                )
            ],

            'layout': {
                'title': 'People Needing Search and Rescue or Medical Help',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
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
