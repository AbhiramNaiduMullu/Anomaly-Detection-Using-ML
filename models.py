import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import tensorflow_probability as tfp
import xgboost as xgb
import shap
import streamlit as st

# Autoencoder_Neural_Network


def AENN(sensor_data, sensor_data_test, epochs=100, batch_size=32):
    data = sensor_data.values
    data_test = sensor_data_test
    # Define the autoencoder model
    input_dim = data.shape[1]  # Number of features
    encoding_dim = 64  # Number of neurons in the hidden layer

    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoder = layers.Dense(input_dim, activation="sigmoid")(encoder)

    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")

    
    # Train the model
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    # Use the trained autoencoder to detect anomalies
    reconstructed_data = autoencoder.predict(data_test)
    mse = np.mean(np.power(data_test - reconstructed_data, 2), axis=1)
    threshold = np.percentile(mse, 95)  # Set a threshold for anomaly detection

    # Identify anomalies in the data
    anomalies = sensor_data_test[mse > threshold]
    
    return anomalies


# Isolation_Forest


def IF(sensor_data_normalized , sensor_data_normalized_test ,contamination=0.05):
    model = IsolationForest(contamination=contamination)
    model.fit(sensor_data_normalized)
    predictions = model.predict(sensor_data_normalized_test)
    sensor_data_normalized_test["anomaly"] = predictions
    anomalies = sensor_data_normalized_test[sensor_data_normalized_test["anomaly"] == -1]
    return anomalies


# Variational_Auto_Encoder


def VAE(sensor_data, sensor_data_test , epochs=100, batch_size=32):
    # Assuming you have already handled missing values, scaling, etc.
    # Convert the DataFrame to a numpy array
    data = sensor_data.values
    data_test = sensor_data_test.values

    # Define the Variational Autoencoder model
    input_dim = data.shape[1]  # Number of features
    latent_dim = 2  # Dimensionality of the latent space

    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(encoder_inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(
            shape=(tf.keras.backend.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0
        )
        return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation="relu")(decoder_inputs)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    # Define the VAE model
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # Connect the encoder and decoder
    z_inputs = encoder(encoder_inputs)[2]
    reconstructed_outputs = decoder(z_inputs)

    # Define the VAE model
    vae = tf.keras.Model(encoder_inputs, reconstructed_outputs, name="vae")

    # Loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(
        encoder_inputs, reconstructed_outputs
    )
    reconstruction_loss *= input_dim
    kl_loss = (
        1
        + z_log_var
        - tf.keras.backend.square(z_mean)
        - tf.keras.backend.exp(z_log_var)
    )
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")

    # Train the VAE
    vae.fit(data, epochs=epochs, batch_size=batch_size, shuffle=True)

    # Use the trained VAE to detect anomalies
    reconstructed_data = vae.predict(data)
    mse = np.mean(np.power(data - reconstructed_data, 2), axis=1)
    threshold = np.percentile(mse, 95)  # Set a threshold for anomaly detection

    # Identify anomalies in the data
    anomalies = sensor_data[mse > threshold]
    return anomalies


# One_Class_SVM


def OSVM(sensor_data, fraction_of_outliers=0.05):
    data = sensor_data.values

    # Define the One-Class SVM model
    nu = fraction_of_outliers  # Expected fraction of outliers
    one_class_svm = OneClassSVM(nu=nu)
    one_class_svm.fit(data)

    # Predict the anomaly score for each data point
    anomaly_scores = one_class_svm.decision_function(data)

    # Set a threshold for anomaly detection
    threshold = anomaly_scores.mean() - 2 * anomaly_scores.std()

    # Identify anomalies in the data
    anomalies = sensor_data[anomaly_scores < threshold]

    return anomalies


# Gaussian_Mixture_Model


def GMM(sensor_data, n_components=5):
    # Load your sensor data into a DataFrame
    # Preprocess the data
    # Assuming you have already handled missing values, scaling, etc.
    # Convert the DataFrame to a numpy array
    data = sensor_data.values

    # Define the Gaussian Mixture Model
    n_components = n_components  # Number of Gaussian components
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)

    # Calculate the anomaly scores for each data point
    anomaly_scores = gmm.score_samples(data)

    # Set a threshold for anomaly detection
    threshold = anomaly_scores.mean() - 2 * anomaly_scores.std()

    # Identify anomalies in the data
    anomalies = sensor_data[anomaly_scores < threshold]

    return anomalies


# Long_Short-Term_Memory


def LSTM(sensor_data, epochs=100, batch_size=32):
    # Preprocess the data
    # Assuming you have already handled missing values, scaling, etc.
    # Convert the DataFrame to a numpy array
    data = sensor_data.values

    # Normalize the data
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)

    # Reshape the data for LSTM input (assuming it's sequential data)
    n_samples, n_features = normalized_data.shape
    reshaped_data = normalized_data.reshape((n_samples, n_features, 1))

    # Define the LSTM Autoencoder model
    input_shape = (n_features, 1)

    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    encoder_lstm = layers.LSTM(64, return_sequences=True)(encoder_inputs)
    encoder_lstm = layers.LSTM(32, return_sequences=False)(encoder_lstm)
    encoded = layers.RepeatVector(n_features)(encoder_lstm)

    # Decoder
    decoder_lstm = layers.LSTM(32, return_sequences=True)(encoded)
    decoder_lstm = layers.LSTM(64, return_sequences=True)(decoder_lstm)
    decoder_outputs = layers.TimeDistributed(layers.Dense(1))(decoder_lstm)

    # Define the Autoencoder model
    autoencoder = tf.keras.Model(encoder_inputs, decoder_outputs)

    # Compile the model
    autoencoder.compile(optimizer="adam", loss="mse")

    # Train the Autoencoder
    autoencoder.fit(
        reshaped_data, reshaped_data, epochs=epochs, batch_size=batch_size, shuffle=True
    )

    # Use the trained Autoencoder to detect anomalies
    reconstructed_data = autoencoder.predict(reshaped_data)
    mse = np.mean(np.power(reshaped_data - reconstructed_data, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)  # Set a threshold for anomaly detection

    # Identify anomalies in the data
    anomalies = sensor_data[mse > threshold]

    return anomalies


# Gaussian_Mixture_Variational_Autoencoder (GMVAE)


def GMVAE(sensor_data, epochs=100, batch_size=32):
    # Normalize the data
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)

    # Define the Gaussian Mixture Variational Autoencoder model
    input_dim = normalized_data.shape[1]  # Number of features
    latent_dim = 2  # Latent space dimension
    n_components = 5  # Number of Gaussian components

    # Custom layer for KL Divergence calculation
    class KLDivergenceLayer(layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            prior_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros_like(mean), scale_diag=tf.ones_like(log_var)
            )
            posterior_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(0.5 * log_var)
            )
            kl_loss = tf.reduce_mean(
                tfp.distributions.kl_divergence(
                    posterior_distribution, prior_distribution
                )
            )
            self.add_loss(kl_loss)
            return inputs

    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(encoder_inputs)
    x = layers.Dense(32, activation="relu")(x)
    mean = layers.Dense(latent_dim)(x)
    log_var = layers.Dense(latent_dim)(x)
    kl_divergence = KLDivergenceLayer()([mean, log_var])

    # Latent space sampling
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
    z = mean + tf.exp(0.5 * log_var) * epsilon

    # Decoder
    x = layers.Dense(32, activation="relu")(z)
    x = layers.Dense(64, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim)(x)

    # Define the Autoencoder model
    autoencoder = tf.keras.Model(encoder_inputs, decoder_outputs)

    # Compile the model
    autoencoder.compile(optimizer="adam", loss="mse")

    # Train the Autoencoder
    autoencoder.fit(
        normalized_data,
        normalized_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
    )

    # Use the trained Autoencoder to detect anomalies
    reconstructed_data = autoencoder.predict(normalized_data)
    mse = np.mean(np.power(normalized_data - reconstructed_data, 2), axis=1)
    threshold = np.percentile(mse, 95)  # Set a threshold for anomaly detection

    # Identify anomalies in the data
    anomalies = sensor_data[mse > threshold]

    return anomalies


# SHapely_Additive_exPlantations


def SHAP(sensor_data):
    sensor_data[["anomalies"]] = 0
    # Preprocess the data
    # Assuming you have already handled missing values, scaling, etc.

    # Split the data into training and testing sets
    train_size = int(0.8 * len(sensor_data))
    train_data = sensor_data[:train_size]
    test_data = sensor_data[train_size:]

    # Separate the features and target variable
    X_train = train_data.drop(columns=["anomalies"]).values
    y_train = train_data["anomalies"].values
    X_test = test_data.drop(columns=["anomalies"]).values
    y_test = test_data["anomalies"].values

    # Train the XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Summarize the SHAP values
    # shap.summary_plot(shap_values, X_test)

    # Set a threshold for anomaly detection based on SHAP values
    threshold = np.percentile(np.abs(shap_values.values), 95)

    # Identify anomalies in the data
    anomalies = test_data[np.abs(shap_values.values) > threshold]

    return anomalies


# Depth_Based_Isolation_Forest_Feature_Importance


def DBIFFI(sensor_data):
    # Preprocess the data
    # Assuming you have already handled missing values, scaling, etc.

    # Separate the features and target variable
    X = sensor_data.drop(columns=["T2"]).values
    y = sensor_data["T2"].values

    # Train the Isolation Forest model
    model = IsolationForest(contamination="auto")
    model.fit(X)

    # Calculate anomaly scores
    anomaly_scores = model.decision_function(X)

    # Calculate feature importance
    feature_importance = np.sum(
        model.score_samples(X)[:, None] * (X - np.mean(X, axis=0)) ** 2, axis=0
    )

    # Sort and print feature importance
    # sorted_features = sorted(zip(sensor_data.columns[:-1], feature_importance), key=lambda x: x[1], reverse=True)
    # for feature, importance in sorted_features:
    #   print(f"Feature: {feature}, Importance: {importance}")

    # Set a threshold for anomaly detection based on anomaly scores
    threshold = pd.Series(anomaly_scores).quantile(0.95)

    # Identify anomalies in the data
    anomalies = sensor_data[anomaly_scores > threshold]

    return anomalies


# Recurrent Auto Encoder


def RAE(sensor_data, epochs=10, batch_size=32):
    # Define the architecture of the autoencoder model
    input_dim = sensor_data.shape[1]  # Dimensionality of input data
    hidden_dim = 64  # Dimensionality of the hidden layer
    sequence_length = sensor_data.shape[0]  # Length of input sequence

    # Encoder
    encoder_inputs = tf.keras.Input(shape=(sequence_length, input_dim))
    encoder = layers.LSTM(hidden_dim, return_sequences=True)(encoder_inputs)
    encoder = layers.LSTM(hidden_dim, return_sequences=False)(encoder)

    # Decoder
    decoder = layers.RepeatVector(sequence_length)(encoder)
    decoder = layers.LSTM(hidden_dim, return_sequences=True)(decoder)
    decoder_outputs = layers.TimeDistributed(layers.Dense(input_dim))(decoder)

    # Autoencoder model
    autoencoder = tf.keras.Model(encoder_inputs, decoder_outputs)

    # Compile the model
    autoencoder.compile(optimizer="adam", loss="mse")

    # Assuming you have your own sensor data in the variable 'sensor_data'

    # Extract the values from the DataFrame and reshape them
    sensor_data_values = sensor_data.values
    sensor_data_reshaped = sensor_data_values.reshape(-1, sequence_length, input_dim)

    # Train the autoencoder
    autoencoder.fit(
        sensor_data_reshaped, sensor_data_reshaped, epochs=epochs, batch_size=batch_size
    )

    # Detect anomalies
    reconstructions = autoencoder.predict(sensor_data_reshaped)
    mse = np.mean(np.square(sensor_data_reshaped - reconstructions), axis=(1, 2))
    threshold = np.mean(mse) + 2 * np.std(
        mse
    )  # Define a threshold for anomaly detection

    anomalies = sensor_data_reshaped[mse > threshold]

    # Reshape the anomalies to match the expected shape
    num_anomalies = anomalies.shape[0]
    anomalies_reshaped = anomalies.reshape(num_anomalies * sequence_length, input_dim)

    # Create a DataFrame with the original column names
    anomalies_df = pd.DataFrame(anomalies_reshaped, columns=sensor_data.columns)

    return anomalies_df
