# disaster-message-classification

### Motivation

The project showcases my knowledge of building ETL and machine learning pipeline to create a python machine learning classifier. This classifier is then used in a web app that takes a message input and classifies it into possible 36 labels.

### Installation

1. Create a python3 virtual environment

    `python3 -m venv ./venv`

2. Start virtual environment

    ` source venv/bin/activate`

3. Install project dependencies

    > (virtual env must be started to install dependencies)

    `pip install -r requirements.txt`

4. Stop virtual environment

    `deactivate`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Project

#### ETL

Cleans and process csv files in to useable dataframes

This dataframe is then stored into a SQLite database

### Machine learning pipeline
Read data from SQLite database and split the data into a training set and a test set. 

Create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV

In the pipeline, we use CountVectorizer with a pre-built tokenizer and TfidfTransformer to process our text data

We utilize GridSearchCV to find our best parameters to use in our classifier, which was a RandomForestClassifier

Our feature column was disaster text messages

Our labels or predictors were 36 message categories

After training we serialized our model into a pickle file for later reuse in our web app

### Webapp

We used Flask to built a web app that takes in a message input to pass into our classifier so that it can be classified into possible 36 categories. 

In the frontend uses google fonts, bootstrap, and plotly

