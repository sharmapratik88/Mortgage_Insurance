###########################################################################################################
## Customized describe function
## Reference: Feature Engineering Technique, link: https://github.com/sharmapratik88/AIML-Projects
import collections, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
def custom_describe(df):
    results = []
    for col in df.select_dtypes(include = ['float64', 'int64']).columns.tolist():
        stats = collections.OrderedDict({'': col, 'Count': df[col].count(), 
                                         'Type': df[col].dtype, 
                                         'Mean': round(df[col].mean(), 2), 
                                         'StandardDeviation': round(df[col].std(), 2), 
                                         #'Variance': round(df[col].var(), 2), 
                                         'Minimum': round(df[col].min(), 2), 
                                         'Q1': round(df[col].quantile(0.25), 2), 
                                         'Median': round(df[col].median(), 2), 
                                         'Q3': round(df[col].quantile(0.75), 2), 
                                         'Maximum': round(df[col].max(), 2),
                                         #'Range': round(df[col].max(), 2)-round(df[col].min(), 2), 
                                         'IQR': round(df[col].quantile(0.75), 2)-round(df[col].quantile(0.25), 2),
                                         #'Kurtosis': round(df[col].kurt(), 2), 
                                         'Skewness': round(df[col].skew(), 2), 
                                         #'MeanAbsoluteDeviation': round(df[col].mad(), 2)
                                        })
        if df[col].skew() < -1:
            if df[col].median() < df[col].mean(): ske = 'Highly Skewed (Right)'
            else: ske = 'Highly Skewed (Left)'
        elif -1 <= df[col].skew() <= -0.5:
            if df[col].median() < df[col].mean(): ske = 'Moderately Skewed (Right)'
            else: ske = 'Moderately Skewed (Left)'
        elif -0.5 < df[col].skew() <= 0:
            if df[col].median() < df[col].mean(): ske = 'Fairly Symmetrical (Right)'
            else: ske = 'Fairly Symmetrical (Left)'
        elif 0 < df[col].skew() <= 0.5:
            if df[col].median() < df[col].mean(): ske = 'Fairly Symmetrical (Right)'
            else: ske = 'Fairly Symmetrical (Left)'
        elif 0.5 < df[col].skew() <= 1:
            if df[col].median() < df[col].mean(): ske = 'Moderately Skewed (Right)'
            else: ske = 'Moderately Skewed (Left)'
        elif df[col].skew() > 1:
            if df[col].median() < df[col].mean(): ske = 'Highly Skewed (Right)'
            else: ske = 'Highly Skewed (Left)'
        else: ske = 'Error'
        stats['SkewnessComment'] = ske
        upper_lim, lower_lim = stats['Q3'] + (1.5 * stats['IQR']), stats['Q1'] - (1.5 * stats['IQR'])
        if len([x for x in df[col] if x < lower_lim or x > upper_lim])>1: out = 'HasOutliers'
        else: out = 'NoOutliers'
        stats['OutliersComment'] = out
        results.append(stats) 
    statistics = pd.DataFrame(results).set_index('')
    return statistics
###########################################################################################################

###########################################################################################################
## Functions that will help us with EDA plot
## Reference: Ensemble Techniques, link: https://github.com/sharmapratik88/AIML-Projects
from scipy import stats; from scipy.stats import zscore, norm, randint
def odp_plots(df, col):
    f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))
    
    # Boxplot to check outliers
    sns.boxplot(x = col, data = df, ax = ax1, orient = 'v', color = 'darkslategrey')
    
    # Distribution plot with outliers
    sns.distplot(df[col], ax = ax2, color = 'teal', fit = norm).set_title('Distribution of\n{}\nwith outliers'.format(col))
    
    # Removing outliers, but in a new dataframe
    upperbound, lowerbound = np.percentile(df[col], [1, 99])
    y = pd.DataFrame(np.clip(df[col], upperbound, lowerbound))
    
    # Distribution plot without outliers
    sns.distplot(y[col], ax = ax3, color = 'tab:orange', fit = norm).set_title('Distribution of\n{}\nwithout outliers'.format(col))
    
    kwargs = {'fontsize':14, 'color':'black'}
    ax1.set_title(col + '\nBoxplot Analysis', **kwargs)
    ax1.set_xlabel('Box', **kwargs)
    ax1.set_ylabel(col + ' Values', **kwargs)
    f.tight_layout()
    return plt.show()
###########################################################################################################

###########################################################################################################
## Functions to plot given column against target values (0 & 1s)
def target_plot(df, col, target):
    fig = plt.figure(figsize = (15, 7.2))
    # Distribution for 'PPI' - doesn't have a PPI product, considering outliers   
    ax = fig.add_subplot(121)
    sns.distplot(df[(df[target] == 0)][col], color = 'c', ax = ax).set_title(f'{col.capitalize()} don\'t a PPI Product')

    # Distribution for 'PPI' - have a PPI product, considering outliers
    ax= fig.add_subplot(122)
    sns.distplot(df[(df[target] == 1)][col], color = 'b', ax = ax).set_title(f'{col.capitalize()} have a PPI Product')
    return plt.show()
###########################################################################################################