# 🛒 E-Commerce Recommendation System

A machine learning-powered product recommendation web app built with Flask and scikit-learn. Users can search for any product and get personalized recommendations based on content similarity.

---

## 🧠 How It Works

1. Product data is loaded from a cleaned Walmart dataset
2. A **TF-IDF Vectorizer** converts product tags into numerical vectors
3. **Cosine Similarity** finds the most similar products
4. Results are displayed in a clean responsive UI built with Bootstrap

---

## 🗂️ Project Structure

design-project-rec-system/
├── app.py                  → Flask backend and recommendation logic
├── templates/
│   └── index.html          → Frontend UI
├── static/
│   └── img_1-8.png         → Product images
├── models/
│   └── clean_data.csv      → Cleaned product dataset
├── trending_products.csv   → Trending products for homepage
├── requirements.txt        → Python dependencies
└── README.md               → Project documentation

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/kn-keerthana/design-project-rec-system.git
cd design-project-rec-system
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
http://127.0.0.1:5001

---

## 🔍 How to Use

1. Type a full product name in the search box
2. Enter the number of recommendations you want
3. Click **Search**
4. View personalized recommendations below

### Example search:
OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn, TF-IDF, Cosine Similarity |
| Data | pandas, Walmart product dataset |
| Frontend | HTML, Bootstrap 4, Font Awesome |

---

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

Key libraries:
- `flask` — web framework
- `pandas` — data manipulation
- `scikit-learn` — TF-IDF vectorizer and cosine similarity

---

## 🙌 Contributors

- [@jyothikumargoud](https://github.com/jyothikumargoud)
- [@kn-keerthana](https://github.com/kn-keerthana)

---

## 📄 License

This project is for educational purposes.