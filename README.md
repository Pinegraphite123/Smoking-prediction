# Using bio-signals to predict smoking status

## Pine Xie 

## Workflow
1.   Exploratory Data Analysis
2.   Classification imbalance
3.   Feature selection
4.   Supervised learning
5.   Semi-supervised learning
6.   Unsupervised learning


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
```

### download data from kaggle at https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals/data?select=train_dataset.csv
### Use train set only since test set does not have label so it cannot be evaluated

## Exploratory data analysis
```python
train_df = pd.read_csv('/content/train_dataset.csv')
print(train_df.head())
```

<div>
  
| age | height(cm) | weight(kg) | waist(cm) | eyesight(left) | eyesight(right) | hearing(left) | hearing(right) | systolic | relaxation | ... | HDL | LDL | hemoglobin | Urine protein | serum creatinine | AST  | ALT  | Gtp | dental caries | smoking |
|-----|------------|------------|-----------|----------------|-----------------|---------------|----------------|----------|------------|-----|-----|-----|------------|---------------|-----------------|------|------|-----|---------------|---------|
|  35 |        170 |         85 |      97.0 |            0.9 |             0.9  |             1 |              1 |      118 |         78 | ... |  70 | 142 |       19.8 |             1 |             1.0  |   61 |  115 | 125 |             1 |       1 |
|  20 |        175 |        110 |     110.0 |            0.7 |             0.9  |             1 |              1 |      119 |         79 | ... |  71 | 114 |       15.9 |             1 |             1.1  |   19 |   25 |  30 |             1 |       0 |
|  45 |        155 |         65 |      86.0 |            0.9 |             0.9  |             1 |              1 |      110 |         80 | ... |  57 | 112 |       13.7 |             3 |             0.6  | 1090 | 1400 | 276 |             0 |       0 |
|  45 |        165 |         80 |      94.0 |            0.8 |             0.7  |             1 |              1 |      158 |         88 | ... |  46 |  91 |       16.9 |             1 |             0.9  |   32 |   36 |  36 |             0 |       0 |
|  20 |        165 |         60 |      81.0 |            1.5 |             0.1  |             1 |              1 |      109 |         64 | ... |  47 |  92 |       14.9 |             1 |             1.2  |   26 |   28 |  15 |             0 |       0 |
<p>5 rows x 23 columns</p>

</div>


### Check if there is any NA or unencoded values
```python
train_df.isnull().sum()
```

## Class imbalance handling
### Check class imbalance
```python
ax = sns.countplot(x='smoking', data=(train_df))
ax.set_xticklabels(['non-smoking', 'smoking'])
```
![Class imbalance]([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true](https://github.com/Pinegraphite123/Smoking-prediction/blob/fd8aa0a4d16530bef762a4d832ed99dc42ba3a23/Graphs/Class%20imbalance.png))








