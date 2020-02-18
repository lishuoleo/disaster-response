import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report

import pickle

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load data from database as dataframe

    Input:
        database_filepath: File path of sql database

    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and clean text

    Input:
        text: message text

    Output:
        clean_tokens: cleaned(url/stopwords removed, lemmatised and stemmed) tokenised text
    """
#    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#    detected_urls = re.findall(url_regex, text)
#    for url in detected_urls:
#        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = lemmatizer.lemmatize(clean_tok, pos = 'v')
        clean_tok = stemmer.stem(clean_tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    Extract the starting word and return True if it is a verb.
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build a text classification pipeline.

    This pipeline transform the raw text into a cleaned and tokenised dataframe;
    then add a binary feature to indcate if the starting word for the text is a verb;
    then use AdaBoost Classifier to train the model for classification.
    """

    model = Pipeline([
    ('vectoriser', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

#    model = Pipeline([
#        ('features', FeatureUnion([
#
#            ('text_pipeline', Pipeline([
#                ('vect', CountVectorizer(tokenizer=tokenize)),
#                ('tfidf', TfidfTransformer())
#            ])),
#
#            ('starting_verb', StartingVerbExtractor())
#        ])),
#        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
#    ])

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function applies text classification pipeline and output the
    performance of the model.

    Input:
        model <- trained sklearn model
        X_test <- test feautres
        Y_test <- test targets
        category_names <- category labels

    Output:
        Accuracy score and Classfication Report.
    """
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n",
        classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i],
        accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    """
    This function saves trained sklearn models as Pickle file.

    Input:
        model <- trained sklearn model.
        model_filepath <- file path for the Pickle file.

    Output:

        model.pickle file

    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    """
    Train Classifier Main function

    - Load data from database
    - Natural Language Processing (NLP) with classification pipeline
    - Evaluating the model and output the classification report
    - Save the model as a pickle file

    """
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
