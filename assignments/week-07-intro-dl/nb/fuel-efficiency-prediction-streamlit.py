import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Add and resize an image to the top of the app
img_fuel = Image.open("../img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
train_df = pd.read_csv("../dat/train.csv.gz", compression="gzip")
model_results_df = pd.read_csv("../dat/model_results.csv")

# Images for SHAP
img_shap_tpot = Image.open("../img/tpot.png")
img_shap_dnn = Image.open("../img/dnn.png")

# SHAP dict
shap_dict = {'TPOT': img_shap_tpot,
             'DNN': img_shap_dnn}

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Available models for selection

    models = ["DNN", "TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    
    model1_results = model_results_df[model_results_df['model'] == model1_select]

    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)


    with col1:
        st.header(model1_select)
        st.text(model1_results)


    with col2:
        model2_results = model_results_df[model_results_df['model'] == model2_select]
        st.header(model2_select)
        st.text(model2_results)


with tab3: 
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header(model1_select)
        res1 = shap_dict.get(model1_select)
        st.image(res1, width=350)
    
    with col2:
        st.header(model2_select)
        res2 = shap_dict.get(model2_select)
        st.image(res2, width=350)
    