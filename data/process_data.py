import sys
import pandas as pd
import sqlalchemy as db


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df
    
def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand=True)
    category_columns = categories.iloc[:1].apply(lambda x :  x[0][:-2])
    categories.columns = category_columns
    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x : x[-1])
        categories[column] = categories[column].astype(int)
    df = pd.concat([df, categories], axis=1)
    df['related'].replace(2, 1, inplace=True)
    df.drop('categories', inplace=True, axis=1)
    df.drop_duplicates(inplace=True)
    return df
                                                      
def save_data(df, database_filename):
    engine = db.create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')

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