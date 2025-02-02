# -*- coding: utf-8 -*-
"""employee_turnover.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ojhk_5Is7HYu3KmMVuJTWd1cT1_Q7ziz

### 1. Common Functions ###
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import auc,roc_curve,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,cross_validate,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#some print formatting constants
BOLD_START = '\033[1m'
END = '\033[0m'
UNDERLINE = '\033[4m'
DARKCYAN = '\033[36m'

models = {
    "LR" : LogisticRegression(random_state=123),
    "RF" : RandomForestClassifier(),
    "GB" : GradientBoostingClassifier()
}

params_grid = {

    "LR" : {'model__C' :[x/10 for x in range(1,11)],'model__max_iter' : [10000]},

    "RF" :{'model__n_estimators' : range(50,100,10),'model__bootstrap':[False]},

    "GB" :{'model__n_estimators' : range(50,100,10),'model__learning_rate':[0.1]},
}




def evaluate_performance(model_key,X_train,y_train,X_test,cv=5, model_name=None):

    print('%s%s%s%s%s' %(BOLD_START,UNDERLINE,DARKCYAN,'Evaluating performance for {}'.format(model_name),END))
    gs_result = grid_search_cv(model_key,X_train,y_train,cv=5)
    estimator = gs_result.best_estimator_
    score = cross_val_score(estimator,X_train,y_train,cv=cv)
    cv_results = cross_validate(estimator,X_train,y_train,cv=cv,scoring=['accuracy','recall','roc_auc','precision_macro','recall_macro'])
    cl_report = classification_report(y_test,estimator.predict(X_test))
    print('%s%s%s'%(BOLD_START,'Classification Report\n',END))
    accuracy = accuracy_score(y_test,estimator.predict(X_test))
    roc_auc  = roc_auc_score(y_test,estimator.predict(X_test))
    print('%s%s%s'%(BOLD_START,'Accuracy amd roc_auc scores\n',END))
    print('accuracy = %0.4f, roc_auc = %0.4f'%(accuracy,roc_auc))
    plot_classification_report(estimator,X_test,y_test,model_name=model_name)


def plot_classification_report(estimator,X_test,y_test,model_name=None):
    precision_1 = precision_score(y_test,estimator.predict(X_test))
    recall_1 = recall_score(y_test,estimator.predict(X_test))
    f1_1 = f1_score(y_test,estimator.predict(X_test))

    precision_0 = precision_score(y_test,estimator.predict(X_test),pos_label=0)
    recall_0 = recall_score(y_test,estimator.predict(X_test),pos_label=0)
    f1_0 = f1_score(y_test,estimator.predict(X_test),pos_label=0)

    cl_report_dict = {
        'left' : [0,1],
        'precision':[precision_0,precision_1],
        'recall':[recall_0,recall_1],
        'f1':[f1_0,f1_1]
    }

    df = pd.DataFrame(cl_report_dict)
    sns.heatmap(df,annot=True,cmap='GnBu')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.suptitle('Classification report for {}'.format(model_name))
    plt.show()


def grid_search_cv(model_key,X_train,y_train,cv=5):
    model = models[model_key]
    param_grid = params_grid[model_key]
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    grid_search = GridSearchCV(pipeline,param_grid,scoring='accuracy',cv=cv)
    grid_search.fit(X_train,y_train)
    print('%s%s%s'%(BOLD_START,'Best scores and Best Params\n',END))
    print('best score = {}'.format(grid_search.best_score_))
    print('best params = {}'.format(grid_search.best_params_))
    return grid_search

def plot_roc_curve_and_cm(pipeline,X_test,y_test,model_name=None):
    print('%s%s%s%s'%(BOLD_START,DARKCYAN,'evaluating and visualizing ROC_AUC curve and Confusion Matrix for {}\n'.format(model_name),END))
    y_pred_test = pipeline.predict(X_test)
    y_pred_test_prob = pipeline.predict_proba(X_test)
    # plot confusion matrix
    print('Plotting confusion matrix\n')
    cm = confusion_matrix(y_test,y_pred_test)
    print('confusion matrix\n',cm)
    display_labels = ['stayed','left']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=display_labels)
    disp.plot()
    plt.suptitle('Confusion Matrix for {}'.format(model_name))
    plt.show()

    # plot ROC_AUC curve
    print('Plotting ROC_AUC curve\n')
    # we need the probability of 1s which is the second column(index 1)
    y_positive_probability = y_pred_test_prob[:,1]
    result_df = pd.DataFrame({'Actual_label': y_test, 'Pred_label': y_pred_test, 'Pred_prob': y_positive_probability})
    fpr,tpr,threshold = roc_curve(y_test, y_positive_probability)
    area_under_curve = auc(fpr,tpr)
    print('auc = {}\n'.format(area_under_curve))
    plt.plot(fpr,tpr,label=f'ROC curve , AUC = {area_under_curve:.4f}')
    plt.plot([0,1],[0,1],linestyle='--',color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    title_ = 'ROC curve for various thresholds for {}'.format(model_name)
    plt.title(title_)
    plt.legend(loc = 'lower right')
    plt.show()

def create_pipeline(model,X_train,y_train):
     pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
     pipeline.fit(X_train,y_train)
     return pipeline

"""### 2. EDA ###"""

import pandas as pd

print('%s%s%s%s%s' %(BOLD_START,UNDERLINE,DARKCYAN,'Exploratory Data Analysis',END))
# this is the same data as provided in the assignment which i have committed to github.
#This way it is possible to run from anywhere when the file is not present locally
url = 'https://raw.githubusercontent.com/tksundar/employee_turnover/refs/heads/master/HR_comma_sep.csv'

hr_data = pd.read_csv(url,header=0,skip_blank_lines=True, skipinitialspace=True)
#check for missing values
if hr_data.isna().sum().any():
    print('missing values found')
else:
    print('no missing values found')

#the column sales has to be renamed to dept
hr_data.rename(columns = {'sales':'dept'},inplace = True)
# we willl also fix a typo in the column name
hr_data.rename(columns = {'average_montly_hours':'average_monthly_hours'},inplace = True)
hr_data.dept.unique()

"""Consolidating some dept values..."""

hr_data['dept']=np.where(hr_data['dept'] =='support', 'technical', hr_data['dept'])
hr_data['dept']=np.where(hr_data['dept'] =='IT', 'technical', hr_data['dept'])
hr_data['dept'].unique()

hr_data.describe()

"""**3. Which dept has the maximum turnover?**"""

pd.crosstab(hr_data.dept,hr_data.left).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('dept')
plt.ylabel('Frequency of Turnover')
plt.show()

"""**4. Is salary a factor in employee turnover?**"""

pd.crosstab(hr_data.salary, hr_data.left,normalize='index').plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

"""**5. Does number or projects have an impact on employee turnover?**"""

pd.crosstab(hr_data.number_project, hr_data.left,normalize='index').plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart of number of projects vs Turnover')
plt.xlabel('Number of projects')
plt.ylabel('Proportion of Employees')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

"""**6. Correlations - Heatmap**"""

from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#encode the salary data
categories = hr_data['salary'].unique()
enc = OrdinalEncoder(categories=[categories])
hr_data['salary'] = enc.fit_transform(pd.DataFrame(hr_data['salary']))
hr_data.drop(['dept'],axis=1)
plt.figure(figsize=(10,10))
sns.heatmap(hr_data.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Heatmap of numerical columns')
plt.show()

"""satisfaction_level has a medium negative correlatio with employee leaving.

**7. Distributions - Histograms**
"""

hr_data.drop(['left'],axis=1).loc[:,['satisfaction_level','last_evaluation','average_monthly_hours']].hist(bins=30,figsize = (12,6),layout=(1,3))
plt.suptitle('Histogram of feature columns')
plt.show()

"""The above chart shows that employee turnover is high with too few projects or too many projects

**8. Clustering**

**Initial Scatter plot**
"""

plt.figure(figsize=(12,12))
sns.scatterplot(data=hr_data,x='last_evaluation',y='satisfaction_level',hue='left')
plt.title('scatter plot for key features')
plt.show()

"""Looking at the above plot,we see employees leaving for a wide range of satisfaction_level and last_evaluation. However we can also see 3 concentrated regions of employee leaving, but it is not obviuos how many clusters will be optimum for k means clustering. So we will use the elbow curve to find out

**9. Elbow Curve**
"""

from sklearn.cluster import KMeans

df = hr_data.loc[:,['satisfaction_level','last_evaluation','left']]
sum_squared_errors = []

for k in range(1,11):
  km = KMeans(n_clusters=k)
  km.fit(df)
  sum_squared_errors.append(km.inertia_)
plt.plot(range(1,11),sum_squared_errors)
plt.xticks(range(1,11))
plt.xlabel('number of clusters')
plt.ylabel('sum of squared errors')
plt.suptitle('Elbow curve')
plt.show()

"""The above curve as many  elbows and shows that 2,3,4 and 5 are potential values for n_cluster at which points there slope of the elbow curve changes.

**10. Analysis by cluster numbers**
"""

df = hr_data.loc[:,['satisfaction_level','last_evaluation','left']]
k_values = [2,3,4,5]

fig,axes = plt.subplots(1,len(k_values),figsize=(12,4),layout='constrained')

for i,k in enumerate(k_values):
  km = KMeans(n_clusters=k,random_state=42)
  km.fit(df)
  df['cluster'] = km.predict(df)
  df_cluster_wise = df.groupby('cluster').mean()
  print(df_cluster_wise)
  bars = sns.barplot(data = df_cluster_wise,x='cluster',y='satisfaction_level',hue='left',ax=axes[i])
  bars.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.suptitle('bar plot for key features for different clusters')
plt.show()
print()

fig,axes = plt.subplots(1,len(k_values),figsize=(18,8),layout='constrained')
colors = ['red','green','blue','purple']
for i,k in enumerate(k_values):
  km = KMeans(n_clusters=k,random_state=42)
  km.fit(df)
  df['cluster'] = km.predict(df)
  cluster_centers = km.cluster_centers_
  axe = sns.scatterplot(data=df,x='satisfaction_level',y='last_evaluation',hue='left',ax=axes[i] )
  axe.scatter(cluster_centers[:,0],cluster_centers[:,1],marker='X',s=200,c=colors[i])
  axe.legend(loc='upper left', bbox_to_anchor=(1, 1))
  axe.set_title(f'k = {k}')
plt.suptitle('scatter plot for key features for different k values and cluster centers marked')
plt.show()

km

"""K=5 captures the 3 dense clusters of employee leaving in additon to 2 sparse clusters. At all k values we can see that those with a satisfaction_level around 0.5 or less have left. However with 5 clusters, we can also see that a high satisfaction_level and no recent evaluation( high last_evaluation) have also tended to leave. Those who stayed appear to be those whose satisfaction level was average and whose last_evaluation was also neither too old nor too recent.

<i>The random_state parameter of the <code>KMeans.__init__() </code>method determines the output to a large extent. So the above clustering may not be the same for a differnet random_state</i>

**12. Handle the left Class Imbalance using the SMOTE technique**
"""

import seaborn as sns
import matplotlib.pyplot as plt
class_values = pd.DataFrame(hr_data.value_counts('left')).reset_index()
print(type(class_values))
print(class_values)
sns.barplot(data=class_values,x='left',y='count')
plt.show()

"""#### Data is  unbalanced and biased towards the minority class(retention or 0 outcomes).  ####

**13. Encode Categorical Data**
"""

# we can do all of the above in one line of code as the dataset contains just
# one categorical column, viz., dept. We have already encoded the salary column
# values(low, medium,high)with OrdinalEncoder when generating heatmaps

hr_data = pd.get_dummies(hr_data,dtype=int)
print(hr_data.info())

"""**14.SMOTE resampling**"""

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

sm = SMOTE(random_state=123)
X = hr_data.drop('left',axis=1)
y = hr_data['left']
print(hr_data.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123) # 80/20 split

print("Before OverSampling, counts of label '1' in train data: {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': in train data {}".format(sum(y_train == 0)))
X_train,y_train = sm.fit_resample(X_train,y_train)

print("After OverSampling, counts of label '1' in train data: {}".format(sum(y_train == 1)))
print("After OverSampling, counts of label '0': in train data {}".format(sum(y_train == 0)))

"""SMOTE analysis has increased the minority class training sample size to the same as that of the majority class.

**15. 5 fold cross validation and performance evaluation**
"""

evaluate_performance('LR',X_train,y_train,X_test,model_name='Logistic Regression',cv = 5)
evaluate_performance('RF',X_train,y_train,X_test,model_name='Random Forest Classifier',cv = 5)
evaluate_performance('GB',X_train,y_train,X_test,model_name='Gradient Boosting Classifier',cv = 5)

"""The above output show that the Random Forest Classifier performs best with respect to all the metrics.

**16. ROC_AUC curve and Confusion Matrix**
"""

gs = grid_search_cv('LR',X_train,y_train)
lr_estimator = gs.best_estimator_
pipeline = create_pipeline(lr_estimator,X_train,y_train)
plot_roc_curve_and_cm(pipeline,X_test,y_test,model_name='Logistic Regression')

gs = grid_search_cv('RF',X_train,y_train)
rf_estimator = gs.best_estimator_
pipeline = create_pipeline(rf_estimator,X_train,y_train)
plot_roc_curve_and_cm(pipeline,X_test,y_test,model_name='Random Forest Classifier')

gs = grid_search_cv('GB',X_train,y_train)
gb_estimator = gs.best_estimator_
pipeline = create_pipeline(gb_estimator,X_train,y_train)
plot_roc_curve_and_cm(pipeline,X_test,y_test,model_name='Gradient Boosting Classifier')

"""Confusion matrix gives the  counts of the following

Recall = TP/TP+FN . That is out of all employees who left, how many did the model predict(recall) correctly? This measurement is called "recall" and a quick look at these diagrams can demonstrate that random forest is clearly best for this criteria. Out of all the turnover cases, random forest correctly retrieved 695 out of 709. This translates to a turnover "recall" of about 98% (695/709), far better than logistic regression (71%) and better than Gradient Boosing Classifier (94%).

Precison = TP/TP+FP. When a classifier predicts an employee will leave, how often does that employee actually leave? This measurement is called "precision". Random forest again out preforms the other two at about 95% precision (991 out of 1045) with logistic regression at about 51% (273 out of 540), and support vector machine at about 77% (890 out of 1150).

**17. Safe Zones and Retention Strategies**
"""

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

sm = SMOTE(random_state=123)
X = hr_data.drop('left',axis=1)
y = hr_data['left']
print(hr_data.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123) # 80/20 split
X_train,y_train = sm.fit_resample(X_train,y_train)
pipeline = create_pipeline(rf_estimator,X_train,y_train)
y_pred_test = pipeline.predict(X_test)
y_pred_test_prob = pipeline.predict_proba(X_test)
y_positive_probability = y_pred_test_prob[:,1]
X_test['pos_probability'] = y_positive_probability
X_test['zones'] = pd.cut(X_test['pos_probability'],bins=[0.0,0.2,0.6,0.9,1],labels=['Safe Zone','Low-Risk Zone','Medium-Risk Zone','High-Risk Zone'])
#The above code puts 0 probablity as NaN. We will change it to safe zone
X_test = X_test.apply(lambda x:x.fillna('Safe Zone'))
X_test.head()

key_features_df = X_test[['satisfaction_level','last_evaluation','number_project','average_monthly_hours','zones']]
key_features_df.head()
key_features_df['zones'].value_counts()
grp = key_features_df.groupby('zones',observed=False).mean()
grp

y = ['satisfaction_level','last_evaluation','average_monthly_hours']
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,5),layout='constrained')
for i,y_axis in enumerate(y):
  bars = sns.barplot(data=grp.round(2),x='zones',y=y_axis,ax=axes[i],hue='number_project',palette=['green','red','yellow','orange'] )
  xticks = bars.get_xticklabels()
  bars.set_xticks(bars.get_xticks())
  bars.set_xticklabels(xticks, rotation=45)
  bars.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.xticks(rotation=45)
plt.suptitle('Mean of key features for each zone')
plt.show()

"""### Recommendation based on the above data: ###

*caveat: The above clustering analysis considers certain features suggested in the problem statement. But we can see that features such as salasry, number or projects etc also have a measurable impact on employee turnover.*




**Suggestions for retention** <p>
<code><i>To increase retention , the company should aim to increase the satisfaction level, have frequent evaluations reduce working hours and striking a right balance on number of projects. From the charts in sections 3 , 4 and 5 above, we also see that certain departments have greater turnover(technical for example) compared to others. Also, as is intuitive , salary is a factor affecting employee turnover.</i></code>
"""