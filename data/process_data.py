import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Returns a merged dataframe of messages and categories dataset created from csv file
    
    Parameters:
    -----------
        message_filepath : path to csv file containing messages dataset
        categories_filepath : path to csv file containing categories dataset
    
    Returns:
    --------
        df : Merged messages and categories dataset.
    """
    
    ## loading messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # loading categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merging datasets
    df = messages.merge(categories, on='id', how='left')
    
    return df

def clean_data(df):
    
    """
    Returns a clean dataframe discarding all duplicates and categories column separated into individual columns
    
    Parameters:
    -----------
        df : raw dataframe read and merged from csv files 
    
    Returns:
    --------
        df : clean dataset with categories column separated and duplicates dropped
     
    """
    
    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # selecting the first row of the categories dataframe
    row = categories.iloc[0]

    # using this row to extract a list of new column names for categories by using a lambda function
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # renaming the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]

        # converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
     
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # dropping the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)
    
    #dropping columns which only have only 1 unique value
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(subset=df.columns, inplace=True)
    
    #replace value 2 with 1 for 'related' column (related column had 3 distinct values (0,1,2))
    df.loc[(df.related==2)]=1
    
    return df
        

def save_data(df, database_filename):
    
    """
    function saves the dataframe into a sqllite database
    
    Parameters:
    -----------
        df : dataframe to be read into a sqllite table
        database_filename : Filename of the database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace') 


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