# Toxic-Comment-Classification-with-NLP-and-ML
**From Kaggle:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
<hr>

## Libraries and Data Collection 
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
```
```python
df_train = pd.read_csv('train.csv', na_values=[' ?'])
df_test = pd.read_csv('test.csv')
df_train['comment_text'].fillna(' ')
df_test['comment_text'].fillna(' ')
```
<img width="520" height="401" alt="image" src="https://github.com/user-attachments/assets/67b9ab57-2d88-41ce-988d-2ae324e307c8" />


<hr>

<hr>

## Data Pre-Procesing - Text Pre-Processing Using Regular Expressions

- Removing \n characters 
- Removing Aplha-Numeric Characters
- Removing Punctuations
- Removing Non Ascii Characters

```python
import re
import string

remove_n = lambda x: re.sub("\n", "", x)

remove_alpha_num = lambda x: re.sub("\w*\d\w*", '', x)

remove_pun = lambda x: re.sub(r"([^\w\s]|_)", '', x.lower())

remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

df_train['comment_text'] = df_train['comment_text'].map(remove_n).map(remove_alpha_num).map(remove_pun).map(remove_non_ascii)
df_test['comment_text'] = df_test['comment_text'].map(remove_n).map(remove_alpha_num).map(remove_pun).map(remove_non_ascii)
```
<hr>

## EDA - Performaing Data analysis to Discover Potential Issues and trend of the Data

<img width="594" height="453" alt="image" src="https://github.com/user-attachments/assets/05fa2630-3908-4ee3-87cd-468f1152e17d" />


Through Bar charts of Each Category: <b>Prob</b> = Class Imbalance. <b>Solution</b> = Making Frequency of 0s equal to Frequency of 1s by Making Different Dataset of each Category [ id, comment_text, category]. Helps to solve the Issue of Class Imbalance and Helps in Binary Classification of Each Category

- id, comment_text, toxic
- id, comment_text, severe_toxic
- id, comment_text, obscene
- id, comment_text, threat
- id, comment_text, insult
- id, comment_text, identity_hate

```python
df_toxic = df_train[['id', 'comment_text', 'toxic']]
df_severe_toxic = df_train[['id', 'comment_text', 'severe_toxic']]
df_obscene = df_train[['id', 'comment_text', 'obscene']]
df_threat = df_train[['id', 'comment_text', 'threat']]
df_insult = df_train[['id', 'comment_text', 'insult']]
df_identity_hate = df_train[['id', 'comment_text', 'identity_hate']]
```

**Making toxic category balance**

15294 = toxic [ from graph ] 15294 = non-toxic [ to balance out ]
```python
df_toxic_1 = df_toxic[df_toxic['toxic'] == 1]
df_toxic_0 = df_toxic[df_toxic['toxic'] == 0].iloc[:15294]
df_toxic_bal = pd.concat([df_toxic_1, df_toxic_0], axis=0)
```

**Making severe_toxic category balance**

1595 = severe toxic [ from graph ] 1595 = non severe toxic [ to balance out ]
```python
df_severe_toxic_1 = df_severe_toxic[df_severe_toxic['severe_toxic'] == 1]
df_severe_toxic_0 = df_severe_toxic[df_severe_toxic['severe_toxic'] == 0].iloc[:1595]
df_severe_toxic_bal = pd.concat([df_severe_toxic_1, df_severe_toxic_0], axis=0)
```
**Making obscene category balance**

8449 = obscene [ from graph ] 8449 = non obscene [ to balance out ]
```python
df_obscene_1 = df_obscene[df_obscene['obscene'] == 1]
df_obscene_0 = df_obscene[df_obscene['obscene'] == 0].iloc[:8449]
df_obscene_bal = pd.concat([df_obscene_1, df_obscene_0], axis=0)
```
**Making threat category balance**

478 = threat [ from graph ] 478 = non threat [ to balance out ]|
```python
df_threat_1 = df_threat[df_threat['threat'] == 1]
df_threat_0 = df_threat[df_threat['threat'] == 0].iloc[:700]
df_threat_bal = pd.concat([df_threat_1, df_threat_0], axis=0)
```
**Making insult category balance**

7877 = insult [ from graph ] 7877 = non insult [ to balance out ]
```python
df_insult_1 = df_insult[df_insult['insult'] == 1]
df_insult_0 = df_insult[df_insult['insult'] == 0].iloc[:7877]
df_insult_bal = pd.concat([df_insult_1, df_insult_0], axis=0)
```
**Making identity_hate category balance**

1405 = identity hate [ from graph ] 1405 = non identity hate [ to balance out ]
```python
df_identity_hate_1 = df_identity_hate[df_identity_hate['identity_hate'] == 1]
df_identity_hate_0 = df_identity_hate[df_identity_hate['identity_hate'] == 0].iloc[:1405]
df_identity_hate_bal = pd.concat([df_identity_hate_1, df_identity_hate_0], axis=0)
```
### Analysing most frequent words using wordcharts
```python
def frequent_words(dataset, category):
    stopwords = STOPWORDS
    wc = WordCloud(width = 600, height = 600, random_state=42, background_color='black', colormap='rainbow', collocations=False, stopwords = stopwords)
    filter = dataset[dataset[category] == 1]
    text = filter.comment_text.values
    wc.generate(' '.join(text))
    wc.to_file(f"Frequent words in balanced classes/Frequent words in {category} category.png")
```
**Threat category**

<img width="636" height="659" alt="image" src="https://github.com/user-attachments/assets/2b61ea46-9252-4b65-b449-d1d1260dc953" />


<hr>
## Model Building

### VECTORIZATION: Using TF-IDF and Unigram Approach

**Function to perform Vectorization and model building**

Model Used For Each Category: KNN, Logistic Regression, SVM, CNB, BNB, DT, RF, XGBoost

```python
def vector_model(df, category, vectorizer, ngram):
    X = df['comment_text'].fillna(' ')
    Y = df[category]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    vector = vectorizer(ngram_range=(ngram), stop_words='english')

    X_train_scal = vector.fit_transform(X_train)
    X_test_scal = vector.transform(X_test)
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scal, Y_train)
    Y_pred_knn = knn.predict(X_test_scal)
    print(f"Knn done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_knn)} ")
    print("\n----------------------------------------------------------------------")

    #logistic regression
    lr = LogisticRegression()
    lr.fit(X_train_scal, Y_train)
    Y_pred_lr = lr.predict(X_test_scal)
    print(f"\nLr done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_lr)} ")
    print("\n----------------------------------------------------------------------\n")

    #Support Vector Machine
    svm = SVC(kernel='rbf')
    svm.fit(X_train_scal, Y_train)
    Y_pred_svm = svm.predict(X_test_scal)
    print(f"\nsvm done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_svm)} ")
    print("\n----------------------------------------------------------------------\n")

    #Naive Bayes
    cnb = ComplementNB()
    cnb.fit(X_train_scal, Y_train)
    Y_pred_cnb = cnb.predict(X_test_scal)
    print(f"\ncnb done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_cnb)} ")
    print("\n----------------------------------------------------------------------\n")

    bnb = BernoulliNB()
    bnb.fit(X_train_scal, Y_train)
    Y_pred_bnb = bnb.predict(X_test_scal)
    print(f"\nbnb done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_bnb)} ")
    print("\n----------------------------------------------------------------------\n")

    #Decision Tree Classifier
    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, random_state=42)
    dt.fit(X_train_scal, Y_train)
    Y_pred_dt = dt.predict(X_test_scal)
    print(f"\nDT done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_dt)} ")
    print("\n----------------------------------------------------------------------\n")

    #Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=105, min_samples_split=2, random_state=42)
    rf.fit(X_train_scal, Y_train)
    Y_pred_rf = rf.predict(X_test_scal)
    print(f"\nRF done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_rf)} ")
    print("\n----------------------------------------------------------------------\n")
    
    # XGBoost Classifier
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    xgb.fit(X_train_scal, Y_train)
    Y_pred_xgb = xgb.predict(X_test_scal)
    print(f"\nXGBoost done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_xgb)} ")
    print("\n----------------------------------------------------------------------\n")

    f1_scores = [
        round(f1_score(Y_test, Y_pred_knn), 2),
        round(f1_score(Y_test, Y_pred_lr), 2),
        round(f1_score(Y_test, Y_pred_svm), 2),
        round(f1_score(Y_test, Y_pred_cnb), 2),
        round(f1_score(Y_test, Y_pred_bnb), 2),
        round(f1_score(Y_test, Y_pred_dt), 2),
        round(f1_score(Y_test, Y_pred_rf), 2),
        round(f1_score(Y_test, Y_pred_xgb), 2)
    ]

    print(f"F1_scores for {category} category are calculated")

    Scores = {f'F1_Score - {category}': f1_scores}
    Scores_df = pd.DataFrame(
        Scores,
        index=['KNN', 'Logistic Regression', 'SVM', 'Complement NB', 'Bernoulli NB', 'Decision Tree', 'Random Forest', 'XGBoost']
    )
    
    return Scores_df
