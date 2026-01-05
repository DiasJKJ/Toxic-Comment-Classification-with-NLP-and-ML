import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

categories = ['toxic', 'severe_toxic', 'threat', 'obscene', 'insult', 'identity_hate']
models = {}
vectorizers = {}
thresholds = {
    'toxic': 0.58,
    'severe_toxic': 0.5,
    'threat': 0.45,
    'obscene': 0.55,
    'insult': 0.55,
    'identity_hate': 0.5
}

for cat in categories:
    with open(f'pickel/{cat}_model.pkl', 'rb') as f:
        models[cat] = pickle.load(f)
    with open(f'pickel/{cat}_vect.pkl', 'rb') as f:
        vectorizers[cat] = pickle.load(f)

def toxic_classify(comment):
    probs = {}
    classified = []

    for cat in categories:
        vect = vectorizers[cat].transform([comment])
        prob = models[cat].predict_proba(vect)[:,1][0] 
        probs[cat] = prob
        if prob >= thresholds[cat]:
            classified.append(cat.capitalize())
    
    return probs, classified

st.title("Toxic Comment Classifier with Thresholds")
comment = st.text_area("Enter your comment:")

if comment:
    probs, classified = toxic_classify(comment)


    if classified:
        st.subheader("Detected Toxic Categories")
        st.write(", ".join(classified))
    else:
        st.subheader("No category passes the threshold — comment is likely non-toxic")

    fig, ax = plt.subplots()
    ax.bar(probs.keys(), [p*100 for p in probs.values()], color='salmon')
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
