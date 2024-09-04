#Do a fully automatize pipeline or an option to chose everythong you want

#Essayer de mettre le GridsearchCV dans la boucle de tous les model de classification à tester:
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

def standariztion_features(df):

# standardization des colonnes de list_to_norm
    for col in df.columns[:-1]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def split_dataset(df,test_size=0.2):
#split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
def balancing_train(X_train,y_train):

    smote = SMOTE()
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote
def auto_ML_selection(X_train,y_train,X_test,y_test):
# Define the parameter grid to search over
    param_grid = {'LogisticRegression':{
        'max_iter':[10000, 1000, 100, 10], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
        'DecisionTreeClassifier':{
        'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'RandomForestClassifier':{
        'n_estimators': [100,5000,10000], 'max_depth': [None, 20, 30]},
        'KNeighborsClassifier':{
        'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'SVC':{
        'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }
# models to test: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), Naive Bayes, K-Nearest Neighbors (KNN)
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
        KNeighborsClassifier()
    ]



    for model in models:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model.__class__.__name__], cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best parameters and the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Create a new figure for the confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix of {model.__class__.__name__}")

        # Show the plot in Streamlit
        st.pyplot(fig)  # Now we pass the created figure object to Streamlit
        report_dict = classification_report(y_test, y_pred,output_dict=True)  # Set output_dict=True to return a dictionary
        df_classification_report = pd.DataFrame(report_dict)
        st.write(f"The model {model.__class__.__name__} has the following performance:")
        st.write(df_classification_report)
        st.write(f"The best parameters for the {model.__class__.__name__} are {best_params}")
        #save the model as a binary avec scikit learn
        # save the model to disk
        #filename = f'sample_data/{model.__class__.__name__}.sav'
        #pickle.dump(model, open(filename, 'wb'))