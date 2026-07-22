import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor




model=ResNet50( weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# model.summary()
def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result= model.predict(preprocessed_img,batch_size=32).flatten()
    normalized_result=result/norm(result)
    return normalized_result


filenames=[]

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

with ThreadPoolExecutor() as executor:
    features_list = list(
        tqdm(
            executor.map(lambda f: extract_features(f, model),filenames),
            total=len(filenames)
        )
    )



#
# features_list=[]
# for file in  tqdm(filenames):
#     features_list.append(extract_features(file,model))
# print(np.array(features_list).shape)
#
# pickle.dump(features_list,open('embedding.pkl','wb'))
#
# pickle.dump(features_list,open('filenames.pkl','wb'))

# with open('embedding.pkl','wb') as f:
#     pickle.dump(features_list, f)
#
# with open('filenames.pkl','wb') as f:
#     pickle.dump(filenames, f)



import os
import pickle
import gdown
import numpy as np

# --- embedding.pkl ---
embedding_path = "embedding.pkl"
if not os.path.exists(embedding_path):
    file_id = "1X22IuWE2GKIHZiaGa7P7acpHPvX7GjKZ"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading embedding.pkl from Google Drive...")
    gdown.download(url, embedding_path, quiet=False)
features_list = np.array(pickle.load(open(embedding_path, "rb")))

# --- filenames.pkl ---
filenames_path = "filenames.pkl"
if not os.path.exists(filenames_path):
    file_id = "1_lkgAANr8IurB2vKq05Vu4t-FLpP_hzz"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Downloading filenames.pkl from Google Drive...")
    gdown.download(url, filenames_path, quiet=False)
filenames = pickle.load(open(filenames_path, "rb"))




