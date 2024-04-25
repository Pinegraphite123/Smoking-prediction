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

Download data from kaggle at https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals/data?select=train_dataset.csv
Use train set only since test set does not have label so it cannot be evaluated

Exploratory data analysis
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
![Class imbalance](https://github.com/Pinegraphite123/Smoking-prediction/blob/fd8aa0a4d16530bef762a4d832ed99dc42ba3a23/Graphs/Class%20imbalance.png))

There is a slight class imbalance.

Since there is enough number of samples to learn from, downsampling of non-smoking population should be a better choice versus upsampling of smoking population. Unlike oversampling or SMOTE, downsapling does not introduce synthetic data, which reduces the chance of overfitting because it might increase the instances of the minority class based on existing ones.

```python
### df1 is the smoking population, df1 is the non-smoking population

df1 = train_df[train_df.smoking == 1]
df0 = train_df[train_df.smoking == 0]

df0_downsampled = df0.sample(n=len(df1), random_state=123)

# Combine minority class with downsampled majority class
downsampled = pd.concat([df0_downsampled, df1])

# Display new class counts
ax = sns.countplot(x='smoking', data=downsampled)
ax.set_xticklabels(['non-smoking', 'smoking'])
```

![class imbalance handled](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/Class%20imbalance%20handled.png?raw=true)

```python
### downsampled is our new dataset to be used

train_df = downsampled
```

# Feature selection
keep the important and meaningful feature and remove correlated features

```python
### Some basic research tells me height has the least relationship with smoking
train_df = train_df.drop(['height(cm)'], axis=1)
```

Checking if there is multicollnearity on the features. If two features are highly correlated, one of them can be dropped to save some computational resource and reduce confounding variable.

```python
# Using variance inflation factor (VIF) to check multicollinearity, a higher VIF indicates a stronger correlation compared to rest of the features.
# Generally a VIF > 5 is moderate to high in multicollnearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import make_regression

X_train_vif = train_df.copy()

# Adding a constant column for the intercept
X_train_vif['Intercept'] = 1

# Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]

# Removing the intercept row for display
vif_data = vif_data[vif_data["Feature"] != "Intercept"]

plt.figure(figsize=(30, 6))
sns.barplot(x='Feature', y='VIF', data=vif_data)
plt.xticks(rotation=45)
plt.show()
```
![VIF](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/VIF.png?raw=true)

Using pearson correlation together with VIF, one of the correlated features can be dropped
```python
correlation_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```

![CorrMatrix](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/CorrMatrix.png?raw=true)

Features that to be dropped
1. weight: weight might be less representative than waist because waist is more indicative of organ fat, which is directly linked to smoking
2. Cholesterol: basically HDL + LDL, LDL, the bad cholesterol might be more direct than just cholesterol
3. AST: ALT is more indicative of liver injury, which is directly linked to smoking
4. One of the hearing: only a mild correlation. Just drop one of the two, the implication of right or left hearing isnt clear.

```python
train_df = train_df.drop(['weight(kg)', 'Cholesterol', 'AST', 'hearing(left)'], axis=1)
correlation_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```

![CorrMatrixCleaned](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/CorrMatrixCleaned.png?raw=true)













