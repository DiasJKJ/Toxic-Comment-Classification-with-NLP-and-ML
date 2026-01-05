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
![image](<img width="520" height="401" alt="image" src="https://github.com/user-attachments/assets/6f58ed22-09f0-4c11-aaba-54f22c5ca7b1" />)

<hr>
