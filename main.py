import streamlit as st
import os
import pickle
from PIL import Image
import tensorflow
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image

features_list=np.array(pickle.load(open('embedding.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))


model=ResNet50( weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img('sample/Alia_Bhatt.jpg', target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, batch_size=32).flatten()
    normalized_result = result / norm(result)
    return normalized_result
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([normalized_result])
    print(indices)
def recommend(feature,features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distances, indices = neighbors.kneighbors([features])
    return indices

st.title('Fashion-Recommender-System')
uploaded_file=st.file_uploader('choose an image')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
       display_img=Image.open(uploaded_file)
       st.image(display_img)

       features=feature_extraction(os.path.join('uploads',uploaded_file.name),model)
    # st.text(features)
       indices=recommend(features,features_list)
       col1,col2,col3,col4,col5,col6=st.columns(6)
       with col1:
        st.image(filenames[indices[0][0]])
       with col2:
        st.image(filenames[indices[0][1]])
       with col3:
        st.image(filenames[indices[0][2]])
       with col4:
        st.image(filenames[indices[0][3]])
       with col5:
        st.image(filenames[indices[0][4]])
       with col6:
        st.image(filenames[indices[0][5]])

    else:
      st.header('some error occur')
