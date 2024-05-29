
import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras import layers, Model, Input, applications
from keras.applications import MobileNetV2
from keras.applications import ResNet50
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Pneumonia Detection Image Classifier")
st.text("Upload a Chest X-ray Image for Pneumonia Detection")


def load_model_xception():
    # Define the model architecture
    xcep_base = applications.Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    xcep_base.trainable = False

    inputs = Input(shape=(150, 150, 3))
    x = xcep_base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='sigmoid')(x)
    xcep_model = Model(inputs, outputs)

    # Compile the model with the same configuration
    xcep_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    # Load the model weights
    xcep_model.load_weights('xcep_model_weights.h5')
    return xcep_model


def load_model_mobilenet():
  mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
  mobilenet_base.trainable = False

  inputs = Input(shape=(150, 150, 3))

  x = mobilenet_base(inputs, training=False)
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  outputs = Dense(2, activation='sigmoid')(x)
  mobilenet_model = Model(inputs, outputs)

  mobilenet_model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy']
  )
  mobilenet_model.load_weights('mobilenet_model_weights.h5')
  return mobilenet_model


def load_model_resnet():
  resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
  resnet_base.trainable = False

  inputs = Input(shape=(150, 150, 3))

  x = resnet_base(inputs, training=False)
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(0.5)(x)
  outputs = Dense(2, activation='sigmoid')(x)
  resnet_model = Model(inputs, outputs)

  resnet_model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy']
  )
  resnet_model.load_weights('resnet_model_weights.h5')
  return resnet_model


with st.spinner('Loading Models Into Memory....'):
    model_xception = load_model_xception()
    model_mobilenet = load_model_mobilenet()
    model_resnet = load_model_resnet()

st.write("Models Loaded Successfully!")

classes = ['Normal', 'Pneumonia']

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = tf.image.resize(img, [150, 150])
    img = img / 255.0  # Normalizing the image
    return np.expand_dims(img, axis=0)

uploaded_file = st.file_uploader("Choose a Chest X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Sidebar model selection
        st.sidebar.header('Model Selection')
        model_options = st.sidebar.multiselect(
            'Select the models to use for prediction:',
            ('Xception', 'MobileNetV2', 'ResNet50')
        )

        models = {
            'Xception': model_xception,
            'MobileNetV2': model_mobilenet,
            'ResNet50': model_resnet
        }

        selected_models = [models[model_name] for model_name in model_options]

        if not selected_models:
            st.error("Please select at least one model.")
        else:
            st.write("Predicted Class and Probabilities:")
            with st.spinner('Classifying...'):
                image_array = decode_img(uploaded_file.getvalue())
                all_predictions = [model.predict(image_array) for model in selected_models]

                mean_probabilities = np.mean(all_predictions, axis=0)[0]
                label = np.argmax(mean_probabilities)

            st.write(f"Predicted: {classes[label]}")
            
            # Aggregate probabilities
            aggregate_data = {
                'Class': classes,
                'Probability': mean_probabilities
            }
            aggregate_df = pd.DataFrame(aggregate_data)
            st.write("Aggregate Probabilities:")
            st.table(aggregate_df)

            # Individual model predictions
            individual_data = {'Class': classes}
            for model_name, prediction in zip(model_options, all_predictions):
                individual_data[model_name] = prediction[0]

            individual_df = pd.DataFrame(individual_data)
            st.write("Individual Model Predictions:")
            st.table(individual_df)

    except Exception as e:
        st.error(f"Error: {e}")
