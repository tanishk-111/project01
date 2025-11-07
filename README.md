# HOUSEWORT
housewort 
HOUSEWORT - Housing Price Prediction Project
Project Overview
The housewort project is a machine learning-based web application that predicts housing prices based on various house features such as area, number of bedrooms, bathrooms, furnishing status, and more. This application uses historical housing data to train machine learning models, and then provides an easy-to-use web interface where users can input house details and get an estimated price.

The project demonstrates the full workflow of a machine learning solution, including data preprocessing, model training, evaluation, saving the model, and deploying it with a Flask web application.

Explanation of Each File and Folder
1. model_build.py
Purpose: This file is responsible for preparing the data, training the machine learning models, evaluating their performance, and saving the best model.

What it does:

Loads and cleans the housing dataset (Dataset/Housing.csv).

Removes outliers using statistical techniques (Interquartile Range).

Converts categorical variables (like yes/no and furnishing status) into numerical values.

Splits the data into training and testing sets.

Trains two models: Linear Regression and Random Forest Regression.

Evaluates models using Mean Squared Error and R-squared metrics.

Selects and saves the best performing model and data scaler for later use.

Skills Used:

Machine Learning with scikit-learn (model training, evaluation, and saving)

Data preprocessing with pandas

Handling outliers using statistical methods

Model serialization with joblib

2. app.py
Purpose: This is the main Flask web application that interacts with the user.

What it does:

Loads the saved machine learning model and scaler.

Defines web routes to display a form for inputting house details and to show prediction results.

Preprocesses user inputs (mapping categorical values and scaling numerical values).

Uses the trained model to predict house prices based on the input.

Skills Used:

Web development using Flask

Preprocessing input data for prediction with pandas and scikit-learn scaler

Loading models with joblib

3. requirements.txt
Purpose: Lists all Python libraries and their versions needed to run this project.

Why it's important: Allows others to install all dependencies easily using pip.

4. README.md
Currently contains only the project title but acts as a placeholder for project documentation.

5. LICENSE
Contains licensing information about how the project can be used or shared.

6. .gitignore
Lists files and folders (like virtual environments or system files) to be excluded from version control (Git).

Expected Folders
ML_Models/: Stores saved machine learning models (model.joblib) and scalers (scaler.joblib).

Dataset/: Contains the dataset CSV file (Housing.csv) used for training.

Why We Use Machine Learning (ML) in This Project
Predicting house prices is a complex problem influenced by many variables interacting in non-obvious ways. Machine learning allows us to:

Learn complex relationships in historical data to predict prices more accurately than simple formulas.

Generalize predictions to new houses with features not seen before.

Automatically handle multiple features and their interdependencies.

Improve model accuracy by comparing different algorithms and choosing the best.

This project uses Linear Regression and Random Forest Regression to capture linear and nonlinear patterns in the data.

Where Specific Libraries and Skills Are Used
Library/Skill	File	Description
Flask	app.py	Web server setup, routing, rendering templates, handling HTTP requests and responses.
Joblib	model_build.py, app.py	Saving trained models and scalers to disk; loading them back for prediction.
pandas	model_build.py, app.py	Data loading, cleaning, transformation, and converting user inputs to DataFrame.
scikit-learn	model_build.py, app.py	Data scaling (MinMaxScaler), model training (Linear & Random Forest Regression), model evaluation, and prediction.
Machine Learning Concepts	model_build.py	Data preprocessing (outlier removal, encoding), training/testing split, selecting best model, evaluation with MSE and R².
Summary
This project showcases how to build a full machine learning pipeline and deploy it as a web application. It combines data science, backend web development, and software engineering fundamentals to provide a practical housing price prediction tool.

Feel free to explore the code in model_build.py to understand how the model is built and trained, and check app.py to see how the prediction web app is created and how the trained models are used to predict prices based on user input.

Project Overview
The housewort project is a machine learning-based web application that predicts housing prices based on various house features such as area, number of bedrooms, bathrooms, furnishing status, and more. This application uses historical housing data to train machine learning models, and then provides an easy-to-use web interface where users can input house details and get an estimated price.

Setup Instructions
To run this project on your local machine, follow these steps:

1. Clone the Repository
bash
git clone <repository-url>
cd housewort
2. Create a Virtual Environment (Optional but Recommended)
bash
python -m venv venv
# Activate the environment:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
3. Install Required Packages
Use the requirements.txt file to install all dependencies:

bash
pip install -r requirements.txt
4. Ensure Dataset and Models Are Available
Make sure the housing dataset Housing.csv is placed in the Dataset/ folder.

Ensure the ML_Models/ folder contains the saved model files (model.joblib and scaler.joblib). If not, run model_build.py to train and save the models:

bash
python model_build.py
5. Run the Web Application
Start the Flask server to launch the web app:

bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/ to see the input form and use the price predictor.

Example Usage
Open the web app in your browser.

Fill out the form with house details such as area, number of bedrooms, bathrooms, and features like "main road", "furnishing status".

Submit the form.

View the predicted price for the house, generated by the machine learning model based on your inputs.

Explanation of Key Files
model_build.py: Preprocesses data, trains ML models, evaluates them, and saves the best model and scaler.

app.py: Runs the Flask app that takes user input, preprocesses it, and predicts house prices with the saved model.

requirements.txt: Lists Python packages needed for running the project.

README.md: This file, explaining project details, setup, and usage.

LICENSE: The legal license under which the code is provided.

