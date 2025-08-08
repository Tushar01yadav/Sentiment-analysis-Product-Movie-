import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Download dataset if not already
nltk.download('movie_reviews')

# Load the data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle data
random.shuffle(documents)

# Join words to get the review text and label
texts = [" ".join(doc) for doc, label in documents]
labels = [label for doc, label in documents]

# Vectorize text (convert text to numbers)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Predict sentiment of a new review
def predict_sentiment(review):
    review_vec = vectorizer.transform([review])
    pred = model.predict(review_vec)[0]
    return pred

# Test the function
new_review = "This movie was fantastic! I really enjoyed it."
print(f"Sentiment: {predict_sentiment(new_review)}")
# Save the trained model and vectorizer to disk
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump((model, vectorizer), model_file)

        