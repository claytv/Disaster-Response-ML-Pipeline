import sys

import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score


def load_data(database_filepath):
    '''loads data from the designated filepath and 
        transform it variables X(messages) and y(category indicator variables)
       
        parameters 
            * database_filepath - path to database where message data is stored 
        returns 
            * X - message content
            * y - class indicator variables
            * category_names - name of each category of message
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = list(y)
    
    return X, y, category_names


def tokenize(text):
    '''tokenizes by word and lemmatizes
    
       parameters
        * text - text to be tokenized and lemmatized
       returns 
        * cleaned tokens for each root word
    '''
                           
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in word_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    '''creates pipeline based off of the model built
        in the exploratory part of the project
        
        returns
            * pipeline - model to be trained
    '''                   
                           
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(learning_rate=.7, n_estimators=100)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates the model based on accuracy, weighted precision_score 
           and weighted recall_score
    
        parameters
            * model - model to be evaluated
            * X_test - test predicting data
            * Y_test - test response data to compare to predictions
            * catgegory_names - names of each category being classified
    '''
    Y_pred = model.predict(X_test)
    
    for i in range(36):
        category = category_names[i]
        accuracy = accuracy_score(Y_test.iloc[:,i], Y_pred[:,i])
        precision = precision_score(Y_test.iloc[:,i], Y_pred[:,i], average='weighted')
        recall = recall_score(Y_test.iloc[:,i], Y_pred[:,i], average='weighted')
        print(category)
        print("\tAccuracy: %.4f\tPrecision: %.4f\t Recall: %.4f\n" % (accuracy, precision, recall))
    
 


def save_model(model, model_filepath):
    '''saves the model so that it may be reused

        parameters 
            * model - model to be saved
            * model_filepath - destination for file to be saved to
    '''
    
    filename = 'classifier.pkl'
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