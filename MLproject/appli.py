import streamlit as st
from sections.classification.Analyse_df import colinearities, explicative_columns, load_and_encode,histogram_plot, pairplots,correlation_table
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page
from sections.classification.ML_model import standariztion_features, split_dataset, auto_ML_selection,balancing_train

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
    pairplots(df)
    correlation_table(df)
    expl = explicative_columns(df)
    colin = colinearities(df)
    st.write(expl,colin)
    autoML = st.text_input("if you want a fully automated analyses write YES","NO")
    if autoML == "YES":
        standariztion_features(df)
        X_train, X_test, y_train, y_test = split_dataset(df)
        #X_train, y_train = balancing_train(X_train, y_train)
        auto_ML_selection(X_train,y_train,X_test,y_test)
    else:
        st.write("The autoanalysis will not be done")


elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")