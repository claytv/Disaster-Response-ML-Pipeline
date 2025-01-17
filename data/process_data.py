import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' reads in messages and categories data then merges 
        into one dataframe on the column 'id'
        
        parameters 
            * messages_filepath - location of the messages data
            * categories_filepath - location of the categories data
        
        returns 
            * df - messages and categories data merged
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on=['id'])
    
    return df


def clean_data(df):
    ''' cleans the data so that it can be used for analysis 
        -transforms categories column into 36 indicator columns
        -drops duplicate data
        
        parameters 
            * df - dataframe to be cleaned
        returns 
            * clean_df - cleaned dataframe 
    
    '''
    
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.iloc[0]
    
    category_colnames = row.map(lambda x: x[:-2])   
    
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.get(-1)  #grabs indicator value
        categories[column] = categories[column].astype(str).astype(int)
        
    df.drop(['categories'], axis=1,inplace=True)
    
    df = pd.concat([df,categories], axis=1)   #merges original data with indicator columns      
    
    clean_df = df.drop_duplicates() 

    return clean_df


def save_data(df, database_filename):
    '''saves the data to desired database
        
        parameters
            * df - dataframe to be saved
            * database_filename - name of database   
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False)
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()