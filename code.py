import streamlit as st
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)

st.title("Fake News Detector")
@st.cache_data
def load_data():
    df = pd.read_excel("indiafakenews.xlsx")
    return df[['Text', 'Label']]

df = load_data()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % string.punctuation, '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Text'] = df['Text'].apply(clean_text)

X = df['Text']
y = df['Label']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = PassiveAggressiveClassifier(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)


st.subheader("Check News")

news = st.text_area("Paste news text here:")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        cleaned_news = clean_text(news)
        vec = vectorizer.transform([cleaned_news])
        prediction = model.predict(vec)[0]

        if prediction == 0:
            st.error("FAKE NEWS")
        else:
            st.success("REAL NEWS")

