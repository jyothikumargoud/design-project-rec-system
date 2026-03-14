from flask import Flask, render_template, request
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
data = pd.read_csv("models/clean_data.csv")
trending_products = pd.read_csv("trending_products.csv")

# images
images = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png"
]

# truncate function
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    return text


# recommendation
def recommend_products(product_name, n=5):

    if product_name not in data['Name'].values:
        return pd.DataFrame()

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(data["Tags"])

    similarity = cosine_similarity(matrix)

    index = data[data["Name"] == product_name].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    product_indices = [i[0] for i in scores]

    return data.iloc[product_indices][["Name","Brand","Rating"]]


@app.route("/", methods=["GET","POST"])
def home():

    recommendations = None
    message = None

    if request.method == "POST":

        product = request.form["product"]
        n = int(request.form["number"])

        results = recommend_products(product, n)

        if results.empty:
            message = "Product not found"
        else:
            recommendations = results

    random_images = [random.choice(images) for _ in range(len(trending_products))]

    return render_template(
        "index.html",
        trending_products=trending_products.head(8),
        random_product_image_urls=random_images,
        truncate=truncate,
        products=recommendations,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)