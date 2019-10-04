# Disaster Response Pipeline Project
The goal of this project was to get experience using natural lanugage processing and machine learning to classify tweets from disaster victims. The model was deployed to a web application which allowed the user to type in some text and the message classifications are displayed so that the user has a better idea how the model works. This application also displays some visualizations based on aggregations on the training data. 

### Web app 
Right when you arrive to the web app you will see a place to type in a message. Typing in a message and clicking the classify message box will return classification results for that message
![webapp1](webapp1.png)  

The home page of the web app also contains some visualizations of the data  
![webapp3](webapp3.png)  
![webapp4](webapp4.png)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
