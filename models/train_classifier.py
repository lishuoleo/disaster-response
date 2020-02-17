# import libraries

import sys
import pandas as pd
from sqlalchemy import create_engine

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV


def load_data(database_filepath, table_name):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: File path of sql database
        table_name: Target table name
    Output:
        X: Message data (features)
        y: Categories (target)
        category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, y, category_names


def tokenize(text):
    '''
    INPUT
        text: Raw text messages
    OUTPUT
        Return the text in lower-cased, space-free, lemmatised and stemmed from, also free from stopwords.
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos = 'v') for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(LinearSVC()))
                        ])

    # parameters = {'clf__estimator__n_estimators': [50, 100],
    #               'clf__estimator__min_samples_split': [2, 3, 4],
    #               'clf__estimator__criterion': ['entropy', 'gini']
    #              }
    #cv = GridSearchCV(pipeline, param_grid=parameters)

    #return cv
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate model performance using test data
    Input:
        model: Model to be evaluated
        X_test: Test data (features)
        y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    y_pred = model.predict(X_test)

    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))



def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    Input:
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
