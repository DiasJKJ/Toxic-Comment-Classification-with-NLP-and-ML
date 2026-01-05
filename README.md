# Toxic-Comment-Classification-with-NLP-and-ML
**From Kaggle:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
<hr>

## Libraries and Data Collection 
 https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

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
![image](https://github.com/user-attachments/assets/c4e85c56-7f33-4318-b9e4-0b3455d395e4)

<hr>
