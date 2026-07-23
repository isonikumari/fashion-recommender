markdown
# 👗 Fashion Recommender System

A deep learning–based fashion recommendation app built with **TensorFlow (ResNet50)**, **scikit-learn**, and **Streamlit**.  
Upload an image, and the system will suggest visually similar fashion items from the dataset.

---

## 🚀 Features
- Extracts image embeddings using **ResNet50 pretrained on ImageNet**.
- Uses **Nearest Neighbors (Euclidean distance)** to find similar items.
- Interactive **Streamlit web app** for uploading and viewing recommendations.
- Automatic download of required pickle files (`embedding.pkl`, `filenames.pkl`) from Google Drive using **gdown**.

---

## 📂 Project Structure
├── app.py                 # Streamlit app
├── feature_extraction.py  # Script to build embeddings
├── test_opencv.py         # Local OpenCV test script
├── requirements.txt       # Dependencies
├── uploads/               # Uploaded images
├── images/                # Dataset images
├── embedding.pkl          # Precomputed embeddings
├── filenames.pkl          # Image filenames

Code

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-recommender.git
   cd fashion-recommender
Install dependencies:

bash
pip install -r requirements.txt
Make sure you have Python 3.9+ and TensorFlow GPU (optional) installed.

▶️ Usage
Run the Streamlit app:

bash
streamlit run app.py
Upload an image (e.g., a fashion photo).

The app will display 6 similar fashion items side by side.

📦 Dependencies
tensorflow

scikit-learn

numpy

opencv-python

streamlit

pillow

gdown

🗂️ Dataset
Images are stored in the images/ folder.

Embeddings (embedding.pkl) and filenames (filenames.pkl) are auto-downloaded from Google Drive if missing.

🔮 Future Improvements
Add support for larger datasets.

Improve similarity search with FAISS or Annoy.

Deploy on Heroku / Streamlit Cloud for easy access.

👨‍💻 Author
Developed by Soni  
Fashion + Deep Learning Enthusiast ✨


