# diaster-message-classification

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

