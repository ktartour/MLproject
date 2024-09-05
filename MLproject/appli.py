import streamlit as st
from sections.classification.Analyse_df import colinearities, explicative_columns, load_and_encode,histogram_plot, pairplots,correlation_table
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page
from sections.classification.ML_model import standardization_features, auto_ML_selection
from sections.classification.ml_model_choice import choice_split_dataset,choice_cut_dataframe,choice_standardization_features,choice_Balancing,choice_auto_ML_selection


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
    tab_visit = "None"
    tab1, tab2, tab3 = st.tabs(["Preparation", "Analyse", "play with parameters"])
    with tab1:
        st.header("Preparation")
        st.write("test")

    with tab2:
        st.header("Analyse")
        df = load_and_encode()
        histogram_plot(df)
    #Identify columns of interest based on correlation with the target
        liste_col = list(df.columns[:-1])

        correlation_table(df,list_items=liste_col)  #

        explicative_columns(df)     #Return and print the correlation between target and feature with the thershold asked to the user

        list_choice = st.multiselect("Define your features of interest", options=liste_col, default=liste_col)
        #first_check = st.button("Afficher les corrélations")
        #first_check = st.text_input("Afficher les corrélations")
        if st.checkbox("Print correlations"):
            pairplots(df,list_choice)
            colinearities(df, list_choice)

            list_choice2 = st.multiselect("Refine your features", options=liste_col, default=list_choice)
            df2 = standardization_features(df, list_choice2)
            tab_visit="tab2"
            autoML = st.text_input("if you want a fully automated analyses write YES","NO")

            if autoML == "YES":
                auto_ML_selection(df2)
            else:
                st.write("The autoanalysis will not be done")
    with tab3:
        st.header("play with parameters")
        if tab_visit=="tab2":
            list_choice3 = st.multiselect("Define the features of interest for your custom analyse", options=liste_col, default=list_choice2)
            df3 = choice_cut_dataframe(df, list_choice3)
            df3 = choice_standardization_features(df3, list_choice3)
            X_train, X_test, y_train, y_test = choice_split_dataset(df3)
            X_train, y_train = choice_Balancing(X_train, y_train)
            choice_auto_ML_selection(X_train, X_test, y_train, y_test)


        else:
            st.write(f"Start by running the analyse tab")


elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")