
# Anime Recommender System

This project is a Streamlit-based web application that recommends anime titles to users using two main techniques:
1. **Content-Based Filtering**
2. **Collaborative Filtering (SVD model)**

---

## 📁 Project Structure

```
.
├── app.py                 # Streamlit app
├── anime.csv              # Anime metadata
├── train.csv              # User-anime rating dataset
├── svd_model              # Saved SVD model (Surprise library)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🚀 How It Works

### 🔍 Content-Based Filtering
- Uses cosine similarity between anime descriptions.
- You input an anime title, and the app recommends similar titles.

### 🧠 Collaborative Filtering
- Based on user-item interactions using the **SVD (Singular Value Decomposition)** algorithm.
- Predicts ratings for unseen anime and recommends top-rated ones.

---

## 🛠️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/anime-recommender.git
cd anime-recommender
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (If Not Provided)
```python
from surprise import SVD, Dataset, Reader
from surprise import dump
import pandas as pd

train_df = pd.read_csv("train.csv")
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(train_df[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)
dump.dump('svd_model', algo=model)
```

---

## 🖥️ Run the Streamlit App

```bash
streamlit run app.py
```

If you're using Google Colab:
```python
!pip install streamlit pyngrok
from pyngrok import ngrok
!streamlit run app.py & 
public_url = ngrok.connect(port="8501")
print(public_url)
```

---

## 🧪 Example Usage

### Content-Based
- Input: `Naruto`
- Output: Similar titles like `Naruto Shippuden`, `Bleach`, etc.

### Collaborative-Based
- Input: `User ID = 5`
- Output: Recommended anime based on previous ratings.

---

## 🔐 Ngrok Token (if using Colab)
Set up your auth token (from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)):

```python
ngrok.set_auth_token("your_token_here")
```

---

## 📦 Dependencies

```txt
streamlit
pyngrok
pandas
numpy
scikit-learn
scipy
surprise
```

---

## 🤝 Contributing

Feel free to fork this repo and contribute via PRs. All contributions are welcome!

---

## 📄 License

This project is licensed under the MIT License.
