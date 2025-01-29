

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, xticks, title
from scipy.stats import normaltest
from seaborn import kdeplot
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

def do_eda(hr_data):
    # check for missing values
    if hr_data.isna().sum().any():
        print('missing values found')
    else:
        print('no missing values found')

    # the column sales has to be renamed to dept
    hr_data.rename(columns={'sales': 'dept'}, inplace=True)
    # encode the salary data
    print(hr_data['salary'].unique())
    enc = OrdinalEncoder(categories=[['low', 'medium', 'high']])
    hr_data['salary'] = enc.fit_transform(pd.DataFrame(hr_data['salary']))
    # we have just one categorical column which is dept
    e_hr_data = pd.get_dummies(hr_data)
    print(e_hr_data.columns)

    # it appears as if dept does not really affect employee turnover. Lets find its correlation
    # Uncomment the following lines to see the heatmap of correlations between dept and employee turnover
    '''
    columns = ['dept_IT', 'dept_RandD',
           'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing',
           'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical','left']
    dep_data = e_hr_data.loc[:,columns]
    plt.figure(figsize = (8,8))
    sns.heatmap(dep_data.corr(),annot=True, cmap='coolwarm')
    plt.show()
    '''
    # There is negligible correlation between dept and employee turnover. We can therefore safely drop this column
    e_hr_data = hr_data.drop(['dept'], axis=1)
    plt.figure(figsize=(10, 6))
    sns.heatmap(e_hr_data.corr(), annot=True, cmap='coolwarm')
    plt.show()

    # lets plot the distributions of feature variables
    e_hr_data.drop(['left'], axis=1).loc[:, ['satisfaction_level', 'last_evaluation', 'average_montly_hours']].hist(
        bins=30, figsize=(10, 6), layout=(1, 3))
    plt.suptitle('Histogram of feature columns')
    plt.show()

    # bar plot of num projects vs employee turnover
    plt.figure(figsize=(10, 10))
    left_num_projects = pd.DataFrame(hr_data.groupby('left').number_project.mean())
    print('mean of number of projects of those who left and those who stayed',left_num_projects)
    sns.barplot(y=hr_data['number_project'], data=hr_data, hue='left')
    plt.show()

# if __name__ == '__main__':
#     do_eda(pd.read_csv('HR_comma_sep.csv',skipinitialspace=True,skip_blank_lines=True))
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import LogisticRegression
#
#     X = e_hr_data.drop('left', axis=1)
#     y = e_hr_data['left']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#     model = LogisticRegression()
#     do_predict(model, X_train, X_test, y_train, y_test)