import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import models

st.title("Sensor Data Valdiation (GTRE)")

uploaded_file = st.file_uploader("Upload the sensor data", type=["csv"])


def clean_dataframe(df):
    # drop the non-numeric columns directly
    df = df.select_dtypes(include=["float", "int"])

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop missing values
    df = df.dropna()

    # Drop columns with all zeroes
    df = df.loc[:, (df != 0).any(axis=0)]

    # Reset the index
    df = df.reset_index(drop=True)

    return df


def normalized_data(df):
    num_pipeline = Pipeline(
        [
            ("std_scaler", StandardScaler()),
        ]
    )
    df_std = pd.DataFrame(
        num_pipeline.fit_transform(df.values), columns=df.columns, index=df.index
    )
    return df_std


def add_parameter_ui(model_name):
    params = dict()

    if model_name == "Autoencoder Neural Network":
        epochs_aenn = st.sidebar.slider("epochs", 10, 200, value=100)
        params["epochs_aenn"] = epochs_aenn
        batch_size_aenn = st.sidebar.slider("batch size", 16, 128, step=16, value=32)
        params["batch_size_aenn"] = batch_size_aenn
    
    elif model_name == "Variational Autoencoder":
        epochs_vae = st.sidebar.slider("epochs", 10, 200, value=100)
        params["epochs_vae"] = epochs_vae
        batch_size_vae = st.sidebar.slider("batch size", 16, 128, step=16, value=32)
        params["batch_size_vae"] = batch_size_vae

    elif model_name == "Long Short-Term Memory Autoencoder":
        epochs_lstm = st.sidebar.slider("epochs", 10, 200, value=100)
        params["epochs_lstm"] = epochs_lstm
        batch_size_lstm = st.sidebar.slider("batch size", 16, 128, step=16, value=32)
        params["batch_size_lstm"] = batch_size_lstm

    elif model_name == "Gaussian Mixture Variational Autoencoder":
        epochs_gmvae = st.sidebar.slider("epochs", 10, 200, value=100)
        params["epochs_gmvae"] = epochs_gmvae
        batch_size_gmvae = st.sidebar.slider("batch size", 16, 128, step=16, value=32)
        params["batch_size_gmvae"] = batch_size_gmvae

    elif model_name == "Recurrent Auto Encoder":
        epochs_rae = st.sidebar.slider("epochs", 10, 200, value=100)
        params["epochs_rae"] = epochs_rae
        batch_size_rae = st.sidebar.slider("batch size", 16, 128, step=16, value=32)
        params["batch_size_rae"] = batch_size_rae

    elif model_name == "Isolation Forest":
        contamination_if = st.sidebar.slider("contamination", 0.00001, 0.25, value=0.05)
        params["contamination_if"] = contamination_if
    
    elif model_name == "Gaussian Mixture Model":
        n_components_gmm = st.sidebar.slider("components", 1, 15, value=5)
        params["n_components_gmm"] = n_components_gmm

    elif model_name == "One-Class SVM":
        fraction_osvm = st.sidebar.slider("Fraction of outliers", 0.00001, 0.25, value=0.05)
        params["fraction_osvm"] = fraction_osvm
 
    return params

def get_anomalies(model_name , params , sensor_data_normalized , sensor_data_normalized_test):
    anomalies = pd.DataFrame()

    if model_name == "Autoencoder Neural Network":
        st.write(f"Training the data on {model_name} ....")
        anomalies = models.AENN(sensor_data_normalized ,sensor_data_normalized_test, params['epochs_aenn'] , params['batch_size_aenn'])
        st.success("Completed running on the model")
    
    elif model_name == "Variational Autoencoder":
        st.write(f"Training the data on {model_name} ....")
        anomalies = models.VAE(sensor_data_normalized, sensor_data_normalized_test , params['epochs_vae'], params['batch_size_vae'])
        st.success("completed running on the model")

    elif model_name == "Isolation Forest":
        st.write(f"Training the data on {model_name} ....")
        anomalies = models.IF(sensor_data, sensor_data_test , params['contamination_if'])
        st.success("completed running on the model")

    return anomalies


if uploaded_file is not None:
    sensor_data = pd.read_csv(uploaded_file)
    sensor_data = clean_dataframe(sensor_data)
    sensor_data_normalized = normalized_data(sensor_data)


    model_name = st.sidebar.selectbox(
        "Select model",
        (
            "Autoencoder Neural Network",
            "Depth Based Isolation Forest Feature Importance",
            "Gaussian Mixture Model",
            "Gaussian Mixture Variational Autoencoder",
            "Isolation Forest",
            "Long Short-Term Memory Autoencoder",
            "One-Class SVM",
            "Recurrent Auto Encoder",
            "SHapely Additive exPlantations",
            "Variational Autoencoder",
        ),
    )
    
    params = add_parameter_ui(model_name)

    uploaded_file_test = st.file_uploader("Upload the sensor data to be tested", type=["csv"])

    if uploaded_file_test is not None:
        sensor_data_test = pd.read_csv(uploaded_file_test)
        sensor_data_test = clean_dataframe(sensor_data_test)
        sensor_data_normalized_test = normalized_data(sensor_data_test)
        
        if sensor_data.shape == sensor_data_test.shape:
            
            anomalies = get_anomalies(model_name, params, sensor_data_normalized , sensor_data_normalized_test)

            st.write(anomalies.head())


            temp_org = sensor_data_normalized_test['T2'].to_numpy()
            temp_ano = anomalies['T2'].to_numpy()
            time_org = sensor_data_normalized_test['Time'].to_numpy()
            time_ano = anomalies['Time'].to_numpy()
            
            fig = plt.figure()
            plt.scatter(time_org, temp_org , label = 'original')
            plt.scatter(time_ano, temp_ano , label = 'anoamly')
            plt.title(f"{model_name}")
            plt.legend()
            plt.grid()
            plt.show()
            # plt.show()
            st.pyplot(fig)

            st.write(f"""
                    ### Number of anomaly points = {anomalies.shape[0]}
                    """)
        else:
            st.write("""
                     ### The shape of the test data dosent match the shape of the training data
                     """)
