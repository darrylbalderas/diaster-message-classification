import sys
import sqlalchemy as db
import pandas as pd
import numpy as np
import re
import nltk
from nltk import pos_tag
nltk.download(['stopwords', 'punkt','wordnet', 'averaged_perceptron_tagger','maxent_ne_chunker', 'words'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
stop_words = set(stopwords.words('english'))

def load_data(database_filepath):
    """Loads data from database and returns label and  features dataframe"""
    engine = db.create_engine('sqlite:///{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table('messages', con=conn)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, list(Y.columns)

def tokenize(text):
    """Cleans text by lowercase letters, remove any special characters, lemmatize token words and remove and stop words"""
    text = re.sub('[^a-zA-Z0-9]',' ', text.lower().strip())
    tokens = [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
    return tokens


def build_model():
    """Builds a model pipeline that uses grid search best estimator parameters"""
    pipeline = Pipeline(steps=[
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=1.0)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints out classification report for each category name"""
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names, dtype=int)
    for column in Y_test.columns:
        print("-------{}-------".format(column))
        print(classification_report(Y_test[column], y_pred[column]))


def save_model(model, model_filepath):
    """Saves model into a pickle file"""
    joblib.dump(model, model_filepath)

    
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