import streamlit as st
import os
import pickle
from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image

# Load embeddings and filenames
features_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    # ✅ FIX: use img_path instead of hardcoded sample
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, batch_size=32).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(feature, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([feature])
    return indices

# Streamlit UI
st.title('Fashion-Recommender-System')
uploaded_file = st.file_uploader('choose an image')

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_img = Image.open(uploaded_file)
        st.image(display_img)

        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)
        indices = recommend(features, features_list)

        cols = st.columns(6)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]])
    else:
        st.header('some error occur')
