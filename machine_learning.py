import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def ML_pipeline(estimator, train_set, train_label, test_set, test_label):
    """
    1. Machine learning pipeline
        TFIDF Vectorisation
        Choose classifier
    2. Train the classifier
    3. Use the classifier to make predictions and return the result (which can be stored as a variable)
    4. Print out classification report for evaluation
    5. Save the pipeline and estimator
    """
    # Step 1
    pipeline = Pipeline([
            ('classifier',estimator)
            ])
    # Step 2
    pipeline.fit(train_set, train_label)
    # Step 3
    pred = pipeline.predict(test_set)
    # Step 4
    print(classification_report(test_label,pred))
    # Step 5
    filename = str(estimator).split('(')[0] + '10.sav'
    pickle.dump(pipeline, open(filename,'wb'))

    return pred

def train_ML_models(models, X_train, y_train, X_test, y_test):
    """
    Train and save machine learning models
    """
    for model in models:
        print('%s' % model)
        ML_pipeline(model, X_train, y_train, X_test, y_test)

def find_ML_models(directory):
    """
    Locate all the .sav files and append them into an array
    """
    ML_models = []
    for file in os.listdir(directory):
        if file.endswith(".sav"):
            ML_models.append(file)
    
    return ML_models

def ML_load(model):
    """
    Load trained machine learning model
    """
    loaded_model = pickle.load(open(model,'rb'))
    
    return loaded_model

def load_ML_report(models, test_set,test_label):
    """
    Load all trained machine learning models and print out respective classification report
    """
    for model in models:
        loaded_model = pickle.load(open(model,'rb'))
        result = loaded_model.predict(test_set)
        print(model)
        print('---------------------------------------')
        print(classification_report(test_label, result))

# =============================================================================
# knn = ml.ML_pipeline(KNeighborsClassifier(n_neighbors=3), msg_train, label_train, msg_test, label_test)
# sgd = ml.ML_pipeline(SGDClassifier(), msg_train, label_train, msg_test, label_test)
# svc
# Maximum Entropy Model - GIS, IIS, MEGAM, TADM
# I am currently using a voted model between Naive Bayes and Lexicon based approach.
# =============================================================================