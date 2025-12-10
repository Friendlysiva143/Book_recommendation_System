import streamlit as st
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------
# Load Model Files
# ----------------------------------------------------
W = np.load("W.npy")
H = np.load("H.npy")
user_categories = np.load("user_categories.npy", allow_pickle=True)
book_categories = np.load("book_categories.npy", allow_pickle=True)
books_df = pd.read_csv("books.csv")

# Load sparse matrix safely (fix for 0-D issue)
sparse_matrix = np.load("sparse_matrix.npy", allow_pickle=True)
if sparse_matrix.ndim == 0:
    sparse_matrix = sparse_matrix.item()  # extract actual sparse matrix object

# ----------------------------------------------------
# Mapping dictionaries
# ----------------------------------------------------
isbn_to_title = dict(zip(books_df["ISBN"], books_df["Book-Title"]))
isbn_to_image = dict(zip(books_df["ISBN"], books_df["Image-URL-M"])) \
                 if "Image-URL-M" in books_df.columns else {}

# Popular books fallback
top_books = books_df.head(10)

# Placeholder image
PLACEHOLDER_PATH = "placeholder.jpg"

# ----------------------------------------------------
# Recommendation Function
# ----------------------------------------------------
def recommend_nmf(user_id, top_n=5):

    # Cold user fallback
    if user_id not in user_categories:
        df = top_books.head(top_n).reset_index(drop=True)
        return df.to_dict(orient="records"), "User not found. Showing popular books."
    
    # Find user index
    try:
        user_idx = np.where(user_categories == user_id)[0][0]
    except:
        return top_books.head(top_n).to_dict(orient="records"), "User not found."

    user_vec = W[user_idx]
    scores = np.dot(user_vec, H)

    # Remove already-rated items
    try:
        rated_books = sparse_matrix[user_idx].nonzero()[1]
        scores[rated_books] = -np.inf
    except Exception as e:
        return None, f"Sparse matrix error: {e}"

    # Top-N indices
    top_idx = np.argsort(scores)[::-1][:top_n]

    results = []
    for idx in top_idx:
        isbn = book_categories[idx]
        title = isbn_to_title.get(isbn, "Unknown Title")
        image = isbn_to_image.get(isbn, None)
        score = round(float(scores[idx]), 3)

        results.append({
            "ISBN": isbn,
            "Book-Title": title,
            "Image": image,
            "Score": score
        })

    return results, None

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("Book Recommendation System (NMF Model)")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
top_n = st.number_input("Number of Recommendations", min_value=1, max_value=50, value=5)

if st.button("Recommend"):
    results, error = recommend_nmf(user_id, top_n)

    if error:
        st.warning(error)

    if results:
        st.subheader("Recommended Books")

        # Display in rows of 3
        for i in range(0, len(results), 3):
            cols = st.columns(3)
            for j, book in enumerate(results[i:i+3]):
                with cols[j]:
                    img_url = book.get("Image")
                    if img_url:
                        st.image(img_url, width=150)
                    else:
                        st.image(PLACEHOLDER_PATH, width=150)

                    st.write(f"**{book['Book-Title']}**")
                    st.write(f"ISBN: {book['ISBN']}")
