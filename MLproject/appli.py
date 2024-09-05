import streamlit as st
from sections.classification.Analyse_df import colinearities, explicative_columns, load_and_encode,histogram_plot, pairplots,correlation_table
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page
from sections.classification.ML_model import standariztion_features, split_dataset, auto_ML_selection,balancing_train
import ast

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

if type_data == "Regression":
    regression_page()
elif type_data == "Classification":
    df = load_and_encode()
    histogram_plot(df)
#Identify columns of interest based on correlation with the target
    liste_col = list(df.columns[:-1])

    correlation_table(df,list_items=liste_col)

    expl = explicative_columns(df)

    st.write(f"Features with an absolute correlation >0.5 with the target are:")
    st.write(expl)

    list_choice = st.multiselect("Define your features of interest", options=liste_col, default=liste_col)
    #first_check = st.button("Afficher les corrélations")
    #first_check = st.text_input("Afficher les corrélations")
    if st.checkbox("Afficher les corrélations"):
        pairplots(df,list_choice)

        colin = colinearities(df, list_choice)
        st.write("Calculation of the Variance inflation factor, It is a measure for multicollinearity of the design matrix ")
        st.write(colin)
        list_choice2 = st.multiselect("Refine your features", options=liste_col, default=list_choice)
        autoML = st.text_input("if you want a fully automated analyses write YES","NO")
        if autoML == "YES":
            df2 = standariztion_features(df,list_choice2)
            X_train, X_test, y_train, y_test = split_dataset(df2)
            #X_train, y_train = balancing_train(X_train, y_train)
            auto_ML_selection(X_train,y_train,X_test,y_test)
        else:
            st.write("The autoanalysis will not be done")


elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")