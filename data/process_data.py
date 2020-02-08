import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('../data/disaster_categories.csv')
    df = pd.merge(messages, categories, how = 'right', on = 'id')

    return df


def clean_data(df):

    categories = df.categories.str.split(pat = ';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda str : str[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda str: str[-1])
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)
    print("Data was saved to {} in the DisasterResponse table".format(database_filename))

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
