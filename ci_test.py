
# coding: utf-8

# # Data science analysis: ttH(bb) dilepton channel
# 
# **Notebook by Christian Contreras-Campana, PhD**

# ## Introduction
# 
# Developing a data analytic report scheme for ttH(bb) multivariate analysis study using machine larning technologies.
# 
# The columns in the file are:
# - mass_tag_tag_min_deltaR: Mass for b-tag jet pair with minimum $\Delta$R
# - median_mass_jet_jet: Median invariant mass of all combinations of jet pairs
# - maxDeltaEta_tag_tag:  The $\Delta\eta$ between the two furthest b-tagged jets
# - mass_higgsLikeDijet:  The invariant mass of a jet pair ordered in closeness to a Higgs mass
# - HT_tags: Scalar sum of transverse momentum for all jets
# - btagDiscriminatorAverage_tagged:  Average CSV b-tag discriminant value for b-tagged jets
# - mass_jet_tag_min_deltaR:  Invariant mass of jet pair (with at least one b-tagged) $\Delta$R
# - mass_jet_jet_min_deltaR:  Invariant mass of jet pair $\Delta$R
# - mass_tag_tag_max_mass:  Mass for b-tagged jet pair with maximum invariant mass combination
# - centrality_jets_leps:  The ratio of the sum of the transverse momentum of all jets and leptons
# - maxDeltaEta_jet_jet:  Invariant mass of jet pair $\Delta$R
# - centrality_tags:  The ratio of the sum of the transverse momentum of all b-tagged jets
# 
# While we have some grasp on the matter, we're not experts, so the following might contain inaccuracies or even outright errors. Feel free to point them out, either in the comments or privately.

# ## Load Libraries
# 
# We load all the necessary python libraries that will permit us to load the data files, pre-process and clean the data, perform data validation, produce statistical summaries, conduct exploratory data analysis, as well as feature transformation, feature ranking, and feature selection. Python libraries will also be needed for model selection, evaluating overfitting, executing standard nested k-fold cross validation for hyper-parameter optimization and model evaluation.  

# In[12]:

## Import common python libraries
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Import from root_numpy library
import root_numpy
from root_numpy import root2array, rec2array

# Import panda library
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix
from pandas.core.index import Index
import pandas.core.common as com

# Import scipy
import scipy
from scipy.stats import ks_2samp
import scipy as sp

# Import itertools
import itertools
from itertools import cycle

# Import Jupyter
from IPython.core.interactiveshell import InteractiveShell

# Import scikit-learn
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RandomizedLasso

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFECV
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                             auc, average_precision_score, precision_score, 
                             brier_score_loss, recall_score, f1_score, log_loss, 
                             classification_report, precision_recall_curve)
from sklearn.dummy import DummyClassifier

from sklearn.externals import joblib
from sklearn import feature_selection

# Import imblearn
import imblearn
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import defaultdict, Counter

import re

# Check the versions of libraries/packages
print("Python version " + sys.version)
print("Sklearn version " + sklearn.__version__)
print("Root_numpy version " + root_numpy.__version__)
print("Numpy version " + np.__version__)
print("Scipy version " + scipy.__version__)
print("Pandas version " + pd.__version__)
print("Matplotlib version " + matplotlib.__version__)
print("Seaborn version " + sns.__version__)
print("Imblance version " +imblearn.__version__)

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Specifying which nodes should be run interactively
InteractiveShell.ast_node_interactivity = "all"
print(__doc__)


# ## Load Data Files
# 
# Most data files contain approximately 15K events. There are a total of 4 files totaling 80K data events. We list the features and response names. We store the data in a Pandas DataFrame for greater ease of data manipulation.
# 
# **Note: To reduce running time of the program we use our most signal-like category which is statistically limited**

# In[13]:

## Define data load function

def load(sig_filename, bkg_filename, category, features):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    sig_filename : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    bkg_filename : array, shape = [n_samples, n_classes]
    category: string
    features: array, shape = [n_features]

    Returns
    -------
    data : pandas.DataFrame
    """

    signal = root2array(sig_filename, category, features)
    signal = rec2array(signal)

    backgr = root2array(bkg_filename, category, features)
    backgr = rec2array(backgr)

    # for sklearn data is usually organised
    # into one 2D array of shape (n_samples x n_features)
    # containing all the data and one array of categories
    # of length n_samples
    X = np.concatenate((signal, backgr))
    y = np.concatenate((np.ones(signal.shape[0]), np.zeros(backgr.shape[0])))

    # convert to numpy ndarray into pandas dataframe
    dataframe_X = pd.DataFrame(data=X, columns=features)
    dataframe_y = pd.DataFrame(data=y, columns=['y'])

    data = pd.concat([dataframe_X, dataframe_y], axis=1)

    return data


# In[14]:

## Load data files

# Feature names
branch_names = """mass_tag_tag_min_deltaR,median_mass_jet_jet,
    maxDeltaEta_tag_tag,mass_higgsLikeDijet,HT_tags,
    btagDiscriminatorAverage_tagged,mass_jet_tag_min_deltaR,
    mass_jet_jet_min_deltaR,mass_tag_tag_max_mass,maxDeltaEta_jet_jet,
    centrality_jets_leps,centrality_tags""".split(",")

features = [c.strip() for c in branch_names]
features = (b.replace(" ", "_") for b in features)
features = list(b.replace("-", "_") for b in features)

wall = time.time()
process = time.clock()

# Load dataset
signal_sample = "combined/signalMC.root"
background_sample = "combined/backgroundMC.root"
tree_category = "event_mvaVariables_step7_cate4"

data = load(signal_sample, background_sample, tree_category, features)

print "Total number of events: {}\nNumber of features: {}".format(data.shape[0], data.shape[1])

# Store a copy for later use
df_archived = data.copy(deep=True)

print "\nWall time to read in file input: ", time.time()-wall
print "Elapsed time to read in file input: ", time.clock()-process


# In[15]:

## Define class label counts and percentages

def class_info(classes):
    # Store the number of signal and background events
    class_count = {}
    counts = Counter(classes)
    total = sum(counts.values())

    for cls in counts.keys():
        class_count[class_label[cls]] = counts[cls]
        print("%6s: % 7d  =  % 5.1f%%" % (class_label[cls], counts[cls], float(counts[cls])/float((total))*100.0))

    return (class_count["signal"], class_count["background"])


# In[16]:

# Determine class label counts and percentages
class_label = {0.0: "background", 1.0: "signal"}
class_info(data.y);


# In[17]:

## Create features dataframe and target array

df_X = data.drop("y", axis=1, inplace=False)
df_y = data["y"]


# ## Statistical Summary
# 
# We give a statistical summary below to make sure the data makes sense and that nothing anomolous is present. As we can see values look promising and have acceptable variances.

# In[18]:

## Print statistical summary of dataset

# To print out all rows and columns to the terminal
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

wall = time.time()
process = time.clock()

print "Head:"
data.head()

print "Describe:"
data.describe()

print "Information:" 
data.info()


print "\nWall time to print statistical summary: ", time.time()-wall
print "Elapsed time to print statistical summary: ", time.clock()-process


# ## Feature Visualization: Basic Exploratory Data Analysis¶
# We conduct a basic exploratory data analyses by producing correlaiton matrices between all variables of interest. In addition, we visually depict the relationship signal and background rate and feature variables.

# In[19]:

## Plot signal and background distributions for some variables

# The first two arguments select what is "signal"
# and what is "background". This means you can
# use it for more general comparisons of two
# subsets as well.

def signal_background(data1, data2, column=None, grid=True,
                      xlabelsize=None, xrot=None, ylabelsize=None,
                      yrot=None, ax=None, sharex=False,
                      sharey=False, figsize=None,
                      layout=None, bins=10, **kwds):
   """Draw histogram of the DataFrame's series comparing the distribution
   in `data1` to `data2`.

   data1: DataFrame
   data2: DataFrame
   column: string or sequence
       If passed, will be used to limit data to a subset of columns
   grid : boolean, default True
       Whether to show axis grid lines
   xlabelsize : int, default None
       If specified changes the x-axis label size
   xrot : float, default None
       rotation of x axis labels
   ylabelsize : int, default None
       If specified changes the y-axis label size
   yrot : float, default None
       rotation of y axis labels
   ax : matplotlib axes object, default None
   sharex : bool, if True, the X axis will be shared amongst all subplots.
   sharey : bool, if True, the Y axis will be shared amongst all subplots.
   figsize : tuple
       The size of the figure to create in inches by default
   layout: (optional) a tuple (rows, columns) for the layout of the histograms
   bins: integer, default 10
       Number of histogram bins to be used
   kwds : other plotting keyword arguments
       To be passed to hist function
   """        

   if "alpha" not in kwds:
       kwds["alpha"] = 0.5

   w, h = (12, 8)
   figsize = (w, h)

   if column is not None:
       if not isinstance(column, (list, np.ndarray, Index)):
           column = [column]
       data1 = data1[column]
       data2 = data2[column]

   data1 = data1._get_numeric_data()
   data2 = data2._get_numeric_data()
   naxes = len(data1.columns)


   fig, axes = plotting._subplots(naxes=naxes, 
                                  ax=ax, 
                                  squeeze=False,
                                  sharex=sharex,
                                  sharey=sharey,
                                  figsize=figsize,
                                  layout=layout)
   xs = plotting._flatten(axes)

   for i, col in enumerate(com._try_sort(data1.columns)):
       ax = xs[i]
       low = min(data1[col].min(), data2[col].min())
       high = max(data1[col].max(), data2[col].max())
       ax.hist(data1[col].dropna().values,
               bins=bins, histtype='stepfilled', range=(low,high), **kwds)
       ax.hist(data2[col].dropna().values,
               bins=bins, histtype='stepfilled', range=(low,high), **kwds)
       ax.set_title(col)
       ax.legend(['background', 'signal'], loc='best')
       ax.set_facecolor('white')
    
       # Customize the major grid
       ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
       ax.set_facecolor('white')
    

   plotting._set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                             ylabelsize=ylabelsize, yrot=yrot)
   fig.subplots_adjust(wspace=0.5, hspace=0.8)

   return plt.show()


# In[20]:

## Plot feature hitograms

wall = time.time()
process = time.clock()

signal_background(data[data["y"] < 0.5], data[data["y"] > 0.5],
                  column=features, bins=40);

print "\nWall time to plot exploratory data features: ", time.time()-wall
print "Elapsed time to plot exploratory data features: ", time.clock()-process


# In[21]:

## Define linear correlation matrix

def correlations(data, **kwds):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    data : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    kwds : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
        
    
    """To calculate pairwise correlation between features.
    
    Extra arguments are passed on to DataFrame.corr()
    """
    
    # Select signal or background label for plot title
    if (data["y"] > 0.5).all(axis=0):
        label = "signal"
    elif (data["y"] < 0.5).all(axis=0):
        label = "background"
    
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    data = data.drop("y", axis=1) 
 
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    labels = data.corr(**kwds).columns.values
    
    fig, ax1 = plt.subplots(ncols=1, figsize=(8,7))
    
    opts = {"annot" : True,
            "ax" : ax1,
            "vmin": 0, "vmax": 1*100,
            "annot_kws" : {"size": 8}, 
            "cmap": plt.get_cmap("Blues", 20),
            }
    
    ax1.set_title("Correlations: " + label)

    sns.heatmap(data.corr(method="spearman").iloc[::-1]*100, **opts) 
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels[::-1], minor=False, ha="right", rotation=70)
        ax.set_yticklabels(np.flipud(labels), minor=False)
        
    plt.tight_layout()
    
    return plt.show()


# In[22]:

## Plot feature correlations (assuming linear correlations)

wall = time.time()
process = time.clock()

# Remove the y column from the correlation matrix
# after using it to select background and signal
sig = data[data["y"] > 0.5]
bg = data[data["y"] < 0.5]

# Correlation Matrix
correlations(sig)
correlations(bg)

print "Wall time to plot correlation matrix: ", time.time()-wall
print "Elapsed time to plot correlation matrix: ", time.clock()-process


# In[23]:

## Scatter Plot
#get_ipython().magic(u'matplotlib inline')

sns.set(style="ticks", color_codes=True)
wall = time.time()
process = time.clock()

random.seed(a=seed)

#_ = sns.pairplot(data.drop(data.y, axis=0), size=2.5, hue="y")
_ = sns.pairplot(data.drop(data.y, axis=0), size=2.5, hue="y", 
                 markers=["o", "s"], plot_kws={ "s":5,"alpha":0.7 })
#sns.plt.show()

print "Wall time to plot scatter distribution: ", time.time()-wall
print "Elapsed time to plot scatter distribution: ", time.clock()-process


# ## Model performance measure
# 
# We investigate several machine learning models in order to establish which algorithm may be the most promising for the discrimination modeling of signal and background processes. Two performance measures wil be used to help select our model, namely, accuracy and the area under the receiver operating characteristic (ROC) curve. Receiver Operating Characteristic (ROC) curve number is equal to the probability that a random positive example will be ranked above a random negative example.

# In[24]:

## Compute ROC curve and area under the curve

def roc_plot(models, X, y, n_folds=3):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    models : dictionary, shape = [n_models]
    X : DataFrame, shape = [n_samples, n_classes]
    y : DataFrame, shape = [n_classes]

    Returns
    -------
    roc : matplotlib plot
    """
     
    # Split data into a development and evaluation set
    X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.33, random_state=42)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, 
                                                        test_size=0.33, random_state=seed)
    
    # contains rates for ML classifiers
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    # Include random by chance 'luck' curve
    plt.plot([1, 0], [0, 1], '--', color=(0.1, 0.1, 0.1), label='Luck')
        
    # Loop through classifiers
    for (name, model) in models.items():
        
        print "\n\x1b[1;31mBuilding model ...\x1b[0m"
        process = time.clock()
        model.fit(X_train, y_train)
        print "\t%s fit time: %.3f"%(name, time.clock()-process)
        
        y_predicted = model.predict(X_test)
        
        process = time.clock()
        print classification_report(y_test, y_predicted, target_names=['signal', 'background'])
        print("\tScore (i.e. accuracy) of test dataset: {:.5f}".format(model.score(X_test, y_test)))
        scores = cross_val_score(model, X_test, y_test, scoring="roc_auc",
                                 n_jobs=1, cv=n_folds) #n_jobs=-1
        print "\tCross-validated AUC ROC accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std())  
        print "\tCross-validation time: ", time.clock()-process
        
        if hasattr(model, "predict_proba"):
            # probability estimates of the positive class(as needed in the roc_curve function)
            decisions = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            decisions = model.decision_function(X_test)
        
        process = time.clock()
        fpr[name], tpr[name], thresholds = roc_curve(y_test, decisions)
        
        roc_auc[name] = auc(fpr[name], tpr[name])
        print "\tAUC ROC score for %s: %.4f"%(name, roc_auc[name])
        print "\tAUC ROC time: ", time.clock()-process
    
    # color choices: https://css-tricks.com/snippets/css/named-colors-and-hex-equivalents/
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 
                    'green', 'yellow', 'SlateBlue', 'DarkSlateGrey',
                    'CadetBlue', 'Chocolate', 'darkred', 'GoldenRod'])
  
    for (name, model), color in zip(models.items(), colors):

        signal_efficiecy = tpr[name] # true positive rate (tpr)
        background_efficiecy = fpr[name] # false positive rate (fpr)
        # NOTE: background rejection rate = 1 - background efficiency (i.e specicity)
        background_rejection_rate = 1 - background_efficiecy
        
        plt.plot(signal_efficiecy, background_rejection_rate, color=color, lw=2,
                 label='%s (AUC = %0.3f)'%(name, roc_auc[name]))                 
    
    # Plot all ROC curves
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Signal Efficiency (True Positive Rate)')
    plt.ylabel('Background Rejection Rate (1- False Positive Rate)')
    plt.title("Receiver operating characteristic ({} events)".format(X.shape[0]))
    leg = plt.legend(loc="lower left", frameon=True, fancybox=True, fontsize=8) # loc='best'
    leg.get_frame().set_edgecolor('w')
    frame = leg.get_frame()
    frame.set_facecolor('White')
    
    return plt.show()


# In[29]:

# Plot AUC for ROC curve for several classifiers out-of-the-box

wall = time.time()
process = time.clock()

# Set feature scaling type
scaler = RobustScaler()

# NOTE: When using scikit-learn's DecisionTreeClassifier, 
# always set min_samples_leaf to something like 5 or 10. 
# Its default value of 1 is useless and is guaranteed to overfit. 
# (This is why every example of DecisionTreeClassifier in their docs shows overfitting.)

# prepare models: create a mapping of ML classifier name to algorithm
pipe_classifiers = {
    'SVM':  make_pipeline(scaler, SVC(class_weight="balanced")),
    'NB' :  make_pipeline(scaler, GaussianNB()), 
    'MLP':  make_pipeline(scaler, MLPClassifier()),
    'LR' :  make_pipeline(scaler, LogisticRegression(class_weight="balanced")),
    'ADA':  make_pipeline(None,   AdaBoostClassifier()),
    'KNN':  make_pipeline(scaler, KNeighborsClassifier()),
    'RFC':  make_pipeline(None,   RandomForestClassifier()),
    'CART': make_pipeline(None,   DecisionTreeClassifier(min_samples_leaf=10,
                                                        class_weight="balanced")),
    'LDA':  make_pipeline(scaler, LinearDiscriminantAnalysis()),
    'GRAD': make_pipeline(None,   GradientBoostingClassifier()),
    'BAGG': make_pipeline(None,   BaggingClassifier())
}

# Assessing a Classifier's Performance
roc_plot(pipe_classifiers, df_X, df_y)

# Generally speaking non-cross-validated AUC ROC version is slight optmistic 
# compared to cv-version by ~1-2% because the data composition between signal and data is balanced.
# Note: Useclass balancing for thos classifier that can apply it, shows about 2% improvement
print "\nWall time to generate ROC plots: ", time.time()-wall
print "Elapsed time to generate ROC plots: ", time.clock()-process


# ## Precision-Recall Plots
# 
# Precision-Recall metric to evaluate classifier output quality.
# 
# Recall is a performance measure of the whole positive part of a dataset, whereas precision is a performance measure of positive predictions.
# 
# In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).
# 
# A system with high recall but low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels. A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels. An ideal system with high precision and high recall will return many results, with all results labeled correctly.
# 
# It is important to note that the precision may not decrease with recall. The definition of precision ($\frac{T_p}{T_p + F_p}$) shows that lowering the threshold of a classifier may increase the denominator, by increasing the number of results returned. If the threshold was previously set too high, the new results may all be true positives, which will increase precision. If the previous threshold was about right or too low, further lowering the threshold will introduce false positives, decreasing precision.
# 
# Recall is defined as $\frac{T_p}{T_p+F_n}$, where $T_p+F_n$ does not depend on the classifier threshold. This means that lowering the classifier threshold may increase recall, by increasing the number of true positive results. It is also possible that lowering the threshold may leave recall unchanged, while the precision fluctuates.
# 
# SOURCE: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
# 
# "it has been shown by Davis & Goadrich that an algorithm that optimizes the area under the ROC curve is not guaranteed to optimize the area under the PR curve."

# In[27]:

## Define precision-recall curve

def plot_PR_curve(classifier, X, y, n_folds=5):
    """
    Plot a basic precision/recall curve.
    """
    
    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')
    
    # Calculate the random luck for PR 
    # (above the constant line is a classifier that is well modeled)
    signal_count, background_count = class_info(y)
    ratio = float(signal_count)/float(signal_count + background_count)
    
    # store average precision calculation
    avg_scores = []
    
    # Loop through classifiers
    for (name, model) in classifier.items():
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for i, (train, test) in enumerate(skf.split(X,y)):
            model.fit(X[train], y[train])
                   
            if hasattr(model, "predict_proba"):
                probas_ = model.predict_proba(X[test])[:, 1]
            else:  # use decision function
                probas_ = model.decision_function(X[test])
            
            # Compute precision recall curve
            precision, recall, thresholds = precision_recall_curve(y[test],
                                                                   probas_, pos_label=1)
            # Area under the precision-recall curve (AUCPR)
            average_precision = average_precision_score(y[test], probas_)
            avg_scores.append(average_precision)
        
        plt.plot(recall, precision, lw=1, 
                 label='{0} (auc = {1:0.2f})'.format(name,np.mean(avg_scores, axis=0)))
    
    plt.plot([ratio,ratio], '--', color=(0.1, 0.1, 0.1), 
             label='Luck (auc = {0:0.2f})'.format(ratio))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower left")

    return plt.show()


# In[30]:

# Plot precision-recall curve for several classifiers out-of-the-box

wall = time.time()
process = time.clock()

plot_PR_curve(pipe_classifiers, df_X.values, df_y.values, n_folds=3)

print "\nWall time to generate Precision-Recall plots: ", time.time()-wall
print "Elapsed time to generate Precision-Recall plots: ", time.clock()-process


# Among the models with best performance on the test set:
# 
# - Random forests
# - Gradient boosting 
# - Boosting decision tree
# 
# We observe that the GradientBoostingClassifier has relatively the best accuracy and area under the ROC curve value. Therefore, we select this predictive model and proceed to evaluate whether it is overfitting the model to the noise (e.g. statistical fluctuation) of the data.

# ## Overfitting Evaluation
# 
# Comparing the ML classifier output distribution for the training and testing set to check for overtraining. By comparing the ML classifier's decision function for each class, as well as overlaying it with the shape of the decision function in the training set.
# 
# 
# Using the default parameters for the ML Classifiers we study whether the model is over-fitting the test data.

# In[31]:

## Defined overfitting plot

def compare_train_test(clf, X, y, bins=30):
    """Multi class version of Logarithmic Loss metric.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    
    # Split data into a development and evaluation set
    X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=0.33, random_state=42)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, 
                                                        test_size=0.33, random_state=seed)
    
    # use subplot to extract axis to add ks and p-value to plot
    fig, ax = plt.subplots()
    
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')
    
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):

        if hasattr(clf,"decision_function"):
            d1 = clf.decision_function(X[y>0.5]).ravel()
            d2 = clf.decision_function(X[y<0.5]).ravel()
        else:
            d1 = clf.predict_proba(X[y>0.5])[:, 1]
            d2 = clf.predict_proba(X[y<0.5])[:, 1]
        
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='signal (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='background (train)')
    
    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='signal (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='background (test)')
    
    # Define signal and background histograms for training & testing 
    hist_sig_train, bins = np.histogram(decisions[0], bins=bins, range=low_high, normed=True)
    hist_bkg_train, bins = np.histogram(decisions[1], bins=bins, range=low_high, normed=True)
    
    hist_sig_test, bins = np.histogram(decisions[2], bins=bins, range=low_high, normed=True)
    hist_bkg_test, bins = np.histogram(decisions[3], bins=bins, range=low_high, normed=True)
    
    # Estimate ks-test and p-values as an indicator of overtraining of fit model
    s_ks, s_pv = ks_2samp(hist_sig_train, hist_sig_test)
    b_ks, b_pv = ks_2samp(hist_bkg_train, hist_bkg_test)
    
    if hasattr(clf, "steps"):
        name = clf.steps[1][0]
    else:
        name = clf.__class__.__name__
    
    ax.set_title("Classifier: %s\nsignal(background) ks: %f(%f), p-value: %f (%f)" 
                 % (name, s_ks, b_ks, s_pv, b_pv))

    plt.xlabel("Decision output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    
    return plt.show()


# In[19]:

## Overfitting evaluation

wall = time.time()
process = time.clock()

# Uncalibrated model predictions
model = pipe_classifiers["GRAD"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["ADA"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["SVM"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["LDA"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["KNN"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["LR"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["CART"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["RFC"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)

# Uncalibrated model predictions
model = pipe_classifiers["NB"]
compare_train_test(model, df_X, df_y, bins=40)

# Calibrated with isotonic calibration
model_isotonic = CalibratedClassifierCV(model, cv=5, method='sigmoid')
model_isotonic.fit(df_X, df_y)
compare_train_test(model_isotonic, df_X, df_y, bins=40)


print "\nWall time to generate over-training plots: ", time.time()-wall
print "Elapsed time to generate over-traing plots: ", time.clock()-process


# ## Probability calibration
# 
# When performing classification you often want not only to predict the class label, but also obtain a probability of the respective label. This probability gives you some kind of confidence on the prediction. Some models can give you poor estimates of the class probabilities and some even do not not support probability prediction. Well calibrated classifiers are probabilistic classifiers for which the output of the predict_proba method can be directly interpreted as a confidence level.
# 
# Two approaches for performing calibration of probabilistic predictions are provided: 
# - a parametric approach based on Platt's sigmoid model and a non-parametric approach based on isotonic regression. 
# 
# Probability calibration should be done on new data not used for model fitting. The modelue uses a cross-validation generator and estimates for each split the model parameter on the train samples and the calibration of the test samples. The probabilities predicted for the folds are then averaged. Already fitted classifiers can be calibrated by CalibratedClassifierCV via the paramter cv="prefit". In this case, the user has to take care manually that data for model fitting and calibration are disjoint.

# In[51]:

## Define calibration curve (reliability curve)

def plot_calibration_curve(est, X, y, fig_index, n_bins=10):
    """Plot calibration curve for est w/o and with calibration. """
    
    # Split data into a development and evaluation set
    X_dev,X_eval, y_dev,y_eval = train_test_split(X, y,
                                                  test_size=0.33, random_state=42)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                        random_state=seed)
    
    name = est.steps[1][0]
    
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    fig = plt.figure(fig_index, figsize=(6, 6))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")

    for clf, name in [(est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]: # Also called Platt Scaling
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos =                 (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("\n\x1b[1;31mclassifier %s:\x1b[0m" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value =             calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "o-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=n_bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    # Customize the major grid
    ax1.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax1.set_facecolor('white')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="best", ncol=1)
    
    # Customize the major grid
    ax2.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax2.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()


# In[52]:

## Plot reliability curve (i.e. calibration curve)

wall = time.time()
process = time.clock()

plot_calibration_curve(make_pipeline(None, SVC()), df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["SVM"],   df_X, df_y, 2)

plot_calibration_curve(make_pipeline(None, GaussianNB()), df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["NB"],   df_X, df_y, 2)

plot_calibration_curve(make_pipeline(None, LogisticRegression()), df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["LR"],   df_X, df_y, 2)

plot_calibration_curve(make_pipeline(None, LinearDiscriminantAnalysis()), df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["LDA"],  df_X, df_y, 2)

plot_calibration_curve(make_pipeline(None, KNeighborsClassifier()), df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["KNN"],  df_X, df_y, 2)

# Tree-based classifier
plot_calibration_curve(pipe_classifiers["CART"], df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["RFC"],  df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["ADA"],  df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["GRAD"], df_X, df_y, 2)

#plot_calibration_curve(pipe_classifiers["MLP"],  df_X, df_y, 2)

print "\nWall time to generate Calibration (reliability) plots: ", time.time()-wall
print "Elapsed time to generate Calibration (reliability) plots: ", time.clock()-process


# In[23]:

## Define confusion matrix plot

def plot_confusion_matrix(clf, X, y, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Split data into a development and evaluation set
    X_dev,X_eval, y_dev,y_eval = train_test_split(X, y,
                                              test_size=0.33, random_state=42)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                        random_state=seed)
    
    classifier = clf.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap);
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    name = clf.steps[1][0]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False, which='both')
    
    return plt.show()


# In[24]:

## Generate confusion matrix plot

wall = time.time()
process = time.clock()
    
np.set_printoptions(precision=2)
class_names = ['Background', 'Signal']

# Plot non-normalized confusion matrix
clf = pipe_classifiers["GRAD"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Gradient-Boosting"))
# Plot normalized confusion matrix
clf = pipe_classifiers["GRAD"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Gradient-Boosting"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["ADA"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Adaptive-Boosting"))
# Plot normalized confusion matrix
clf = pipe_classifiers["ADA"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Adaptive-Boosting"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["SVM"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Support Vector"))
# Plot normalized confusion matrix
clf = pipe_classifiers["SVM"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Super Vector"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["LDA"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Linear Discriminant Analysis"))
# Plot normalized confusion matrix
clf = pipe_classifiers["LDA"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Linear Discriminant Analysis"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["LR"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Logistic Regression"))
# Plot normalized confusion matrix
clf = pipe_classifiers["LR"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Logistic Regression"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["RFC"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Random Forest"))
# Plot normalized confusion matrix
clf = pipe_classifiers["RFC"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Random Forest"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["NB"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Guassian Naive Bayes"))
# Plot normalized confusion matrix
clf = pipe_classifiers["NB"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Gaussian Naive Bayes"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["CART"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("Decision Tree"))
# Plot normalized confusion matrix
clf = pipe_classifiers["CART"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("Decision Tree"))

# Plot non-normalized confusion matrix
clf = pipe_classifiers["KNN"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names,
                      title="Classifier: %s\nConfusion matrix, without normalization"%("K-Nearest Neighbor"))
# Plot normalized confusion matrix
clf = pipe_classifiers["KNN"]
plot_confusion_matrix(clf, df_X, df_y, classes=class_names, normalize=True,
                      title="Classifier: %s\nNormalized confusion matrix"%("K-Nearest Neighbor"))

print "\nWall time to generate confusion matrix plots: ", time.time()-wall
print "Elapsed time to generate confusion matrix plots: ", time.clock()-process


# ## Early stopping: Validation Plots
# 
# To validate a model we need a scoring function, for example accuracy for classifiers. The proper way of choosing multiple hyperparameters of an estimator are of course grid search or similar methods that select the hyperparameter with the maximum score on a validation set or multiple validation sets. Note that if we optimized the hyperparameters based on a validation score the validation score is biased and not a good estimate of the generalization any longer. To get a proper estimate of the generalization we have to compute the score on another test set.
# 
# However, it is sometimes helpful to plot the influence of a single hyperparameter on the training score and the validation score to find out whether the estimator is overfitting or underfitting for some hyperparameter values.
# 
# measure the performance of our ensemble as we go along and stop adding trees 
# once we think we have reached the minimum (min. test error).
# 
# It will repeatedly add one more base estimator to the ensemble, measure the performance, and check if we reached minimum. If we reached the minimum it stops, otherwise it keeps adding base estimators until it reaches the maximum number of iterations.
# 
# - There is a minimum number of trees required to skip over the noisier part of the score function
# - Early stopping does not actually stop at the minimum, instead it continues on until the score has increased by scale above the current minimum. This is a simple solution to the problem that we only know we reached the minimum by seeing the score increase again.

# In[53]:

## Validation curve definition

def validation_curve(clf, X, y):
    """Validation curve.

    Parameters
    ----------
    clf : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    X : array, shape = [n_samples, n_classes]
    y : array, shape = [n_samples, n_classes]
    Returns
    -------
    plt : matplotlib
    """
    
    # Split data into a development and evaluation set
    X_dev,X_eval, y_dev, y_eval = train_test_split(X, y, test_size=.33,
                                                   random_state=seed)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                        random_state=seed+31415)

    clf.fit(X_train, y_train)

    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')
    
    test_score = np.empty(len(clf.estimators_))
    train_score = np.empty(len(clf.estimators_))

    for i, pred in enumerate(clf.staged_predict_proba(X_test)):
        test_score[i] = 1-roc_auc_score(y_test, pred[:,1])

    for i, pred in enumerate(clf.staged_predict_proba(X_train)):
        train_score[i] = 1-roc_auc_score(y_train, pred[:,1])

    best_iter = np.argmin(test_score)
    learn = clf.get_params()['learning_rate']
    depth = clf.get_params()['max_depth']
        
    test_line = plt.plot(test_score, label='test (1-roc_auc=%.3f)'%(test_score[best_iter]))

    colour = test_line[-1].get_color()
    plt.plot(train_score, '--', color=colour, label='train (1-roc_auc=%.3f)\nlearn=%.1f depth=%i'
             %(train_score[best_iter],learn,depth))

    plt.title("Validation curve")
    plt.xlabel("Number of boosting iterations")
    plt.ylabel("1 - area under ROC")
    plt.legend(loc='best')
    plt.axvline(x=best_iter, color=colour)
    
    return plt.show()


# In[29]:

##  plot the validation curve for our fitted classifier
# and check with the test set at which number of n_estimators we reach the minimum test error.

wall = time.time()
process = time.clock()

# Set of hyper-parameter selected
opts = dict(max_depth=2, learning_rate=0.1, n_estimators=200)

clf = GradientBoostingClassifier(**opts)

validation_curve(clf, df_X, df_y)

print "\nWall time to generate Validation plots: ", time.time()-wall
print "Elapsed time to generate Validation plots: ", time.clock()-process


# ## Threshold optimisation
# 
# what happens if we use a different hyper-parameter: the threshold applied to decide which class a sample falls in during prediction time, that is normally defaulted to 0.5 threshold.

# ##  Learning curve
# 
# A learning curve shows the validation and training score of an estimator for varying numbers of training samples. It is a tool to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error. If both the validation score and the training score converge to a value that is too low with increasing size of the training set, we will not benefit much from more training data. 

# In[65]:

# Define learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_val_score>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    
    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, 
                                                            n_jobs=n_jobs, 
                                                            train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")

    plt.legend(loc="best")

    
    return plt.show()


# In[66]:

## Plot learning curve

wall = time.time()
process = time.clock()

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.33, random_state=0)

estimator = GradientBoostingClassifier()
plot_learning_curve(estimator, "Learning Curves (Gradient Boosting)", df_X, df_y, ylim=(0.4, 1.01), cv=cv, n_jobs=-1);

print "\nWall time to generate Learning curve plots: ", time.time()-wall
print "Elapsed time to generate Learning curve plots: ", time.clock()-process


# ## Feature Ranking and Feature Selection
# 
# There are various methods for determining feature importance, namely, model-based selection, variance based selection, univariate statistics selections, recursive feature elemination, iterative selection.

# In[67]:

## Define feature ranking 

def feature_ranking_plot(X, indices, title):

    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')
    
    print(title)
    for i in range(X.shape[1]):
        print("%d. %s (%f)" % (i + 1, X.columns[indices[i]], importances[indices[i]]))
    
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #         axis=0)
    
    # Plot the feature importances of the model
    plt.title(title)
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", align="center") #yerr=std[indices]
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()

    return plt.show()


# In[71]:

## Feature ranking study

process = time.time()
process = time.clock()

# Univariate Statistics
#select = VarianceThreshold(threshold=(.8 * (1 - .8)))

# Features selected by variance threshold
#_= select.fit_transform(df_X);

#importances = select.variances_
#indices = np.argsort(importances)[::-1]

#feature_ranking_plot(df_X, indices, "Feature importances based on variance ranking")

# The SelectFromModel class selects all features that have an importance measure 
# of the feature (as provided by the supervised model) greater than the provided 
# threshold.
#select_model = SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=seed), 
#                         threshold="median") #threshold=0.25

# Features selected by SelectFromModel using the RandomForestClassifier
#_= select_model.fit_transform(df_X, df_y)

#importances = select_model.estimator_.feature_importances_
#std = np.std([tree.feature_importances_ for tree in select_model.estimator_], axis=0)
#indices = np.argsort(importances)[::-1]

#feature_ranking_plot(df_X, indices, "Model-based feature importances ranking")

# Build a forest and compute the feature importances
extra_forest = ExtraTreesClassifier(random_state=0)
select_model = SelectFromModel(extra_forest, threshold="median")
_= select_model.fit(df_X, df_y)

importances = select_model.estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in select_model.estimator_], axis=0)
indices = np.argsort(importances)[::-1]

feature_ranking_plot(df_X, indices, "Feature importances based on Extra-Tree classification ranking")

# Univariate Statistics
# Features selected k-highest scores
#score_func=f_classif, k=6
#select = SelectKBest(score_func=f_classif, k=12)

#_= select.fit_transform(df_X);

#importances = select.importances_
#indices = np.argsort(importances)[::-1]

#feature_ranking_plot(df_X, indices, "Feature importances based on variance ranking")

#  Recursive feature elimination (RFE-model base method)
rfe = RFE(estimator=AdaBoostClassifier(), n_features_to_select=11, step=1) #n_features_to_select=8  
_= rfe.fit(df_X, df_y)

importances = rfe.ranking_
indices = np.argsort(importances)[::-1]

feature_ranking_plot(df_X, indices,"Feature importances based on RFE model ranking")

print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[101]:

## define extract feature selection

def extract_feature_selected(clf, X, y):
    
    # Split data into a development and evaluation set
    X_dev,X_eval, y_dev, y_eval = train_test_split(df_X, df_y, test_size=.33, 
                                                   random_state=seed)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                        random_state=seed+31415)

    clf.fit(X_train, y_train)
    select_indices = clf.named_steps['SELECT'].transform(
    np.arange(len(X_train.columns)).reshape(1, -1))

    feature_names = X_train.columns[select_indices]
    
    return feature_names


# In[102]:

## Feature selection study

def features_selection_model_performance(clf, X, y, parameter_set):

    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')
    
    this_scores = list()
    score_means = list()
    score_stds = list()

    params = {'SELECT__k': 'top k features', 
              'SELECT__threshold': 'feature threshold',
              'SELECT__n_features_to_select': 'n features to select',
              'SELECT__percentile': 'percentile',
              'SELECT__cv': 'k-fold',
              'SELECT__selection_threshold':'selection threshold'}
    
    label = [keyname for keyname in clf.get_params().keys() if keyname in params.keys()][0]
    
    for k in parameter_set:

        param = {label: k}
        clf.set_params(**param) 
        
        # Compute cross-validation score using 1 CPU
        this_scores = cross_val_score(clf, X, y, cv=3, n_jobs=1)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(parameter_set, score_means, np.array(score_stds))

    model = clf.steps[1][0]

    title = 'Performance of the {}-{} varying for features selected'.format(model,
                                                                            clf.get_params().keys()[1])
    
    plt.title(title)
    plt.xlabel(params[label])
    plt.ylabel('Prediction rate')

    print  extract_feature_selected(clf, X, y)
    
    return plt.show()


# In[170]:

## Univariate Statistics

process = time.time()
process = time.clock()

# Removing features with low variance
select = VarianceThreshold()
p = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

clf = Pipeline([('SELECT', select), ('rf', RandomForestClassifier())])

# Features selected by variance threshold
_= clf.named_steps["SELECT"].fit_transform(df_X)

importances = clf.named_steps["SELECT"].variances_
indices = np.argsort(importances)[::-1]

# Plot feature ranking
feature_ranking_plot(df_X, indices, "Feature importances based on variance ranking")

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)

print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[171]:

# K-best features
process = time.time()
process = time.clock()

#  Univariate feature selection based on the k highest scores
select = SelectKBest(score_func=f_classif, k=6)
p = [2, 4, 6, 8, 10, 12]

clf = Pipeline([('SELECT', select), ('rf', RandomForestClassifier())])

# Features selected based on k-highest scores
_= clf.named_steps["SELECT"].fit_transform(df_X, df_y)

importances = clf.named_steps["SELECT"].scores_
std = np.std([score for score in clf.named_steps["SELECT"].scores_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot feature ranking
feature_ranking_plot(df_X, indices, "Feature importances based k-best features classification ranking")

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)


print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[172]:

# Percentile of the highest feature scores 

process = time.time()
process = time.clock()

# Univariate feature selection according to a percentile of the highest scores
select = feature_selection.SelectPercentile(f_classif)
p = (6, 10, 15, 20, 30, 40, 60, 80, 100)

clf = Pipeline([('SELECT', select), ('rf', RandomForestClassifier())])

# Features selected based on k-highest scores
_= clf.named_steps["SELECT"].fit_transform(df_X, df_y)

importances = clf.named_steps["SELECT"].scores_
std = np.std([score for score in clf.named_steps["SELECT"].scores_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot feature ranking
feature_ranking_plot(df_X, indices, "Feature importances based percentile of the highest feature scores classification ranking")

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)

print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[173]:

# Recursive feature elimination

process = time.time()
process = time.clock()

# Recursive feature elimination by recursively removing attributes 
# and building a model on those attributes that remain.
select = RFE(estimator=RandomForestClassifier(), step=1)
p = [2, 4, 6, 8, 10, 12]

clf = Pipeline([('SELECT', select), ('rf', RandomForestClassifier())])

# Features selected based on recursive feature elmination
_= clf.named_steps["SELECT"].fit_transform(df_X, df_y)

importances = clf.named_steps["SELECT"].ranking_
std = np.std([tree.feature_importances_ for tree in clf.named_steps["SELECT"].estimator_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot feature ranking
feature_ranking_plot(df_X, indices, "Feature importances based recursive feature elimination classification ranking")

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)


print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[174]:

# Recursive feature elimination with cross-validation

process = time.time()
process = time.clock()

# Recursive feature elimination with cross-validation
select = RFECV(estimator=RandomForestClassifier(), step=1, 
               cv=StratifiedKFold(3), scoring='accuracy')
p = [2, 3, 5, 7]

# Features selected based on k-highest scores
_= clf.named_steps["SELECT"].fit_transform(df_X, df_y)

importances = clf.named_steps["SELECT"].estimator_.feature_importances_ #.ranking_
std = np.std([tree.feature_importances_ for tree in clf.named_steps["SELECT"].estimator_], axis=0)
indices = np.argsort(importances)[::-1]

clf = Pipeline([('SELECT', select), ('rf', RandomForestClassifier())])

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)

# Plot feature ranking
feature_ranking_plot(df_X, indices, "RFECV odel-based feature importances ranking")


print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[181]:

# Build a forest and compute the feature importances
extra_forest = ExtraTreesClassifier(random_state=0)
select = SelectFromModel(extra_forest, threshold="median")
p = [0.001, 0.01, 0.1] # any higher crash

clf = Pipeline([('SELECT', select), ('rfe', RandomForestClassifier())])

# Features selected by SelectFromModel using the ExtraTreesClassifier
_= clf.named_steps["SELECT"].fit(df_X, df_y)

importances = clf.named_steps["SELECT"].estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.named_steps["SELECT"].estimator_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)

# Plot feature ranking
feature_ranking_plot(df_X, indices, "ExtraTree Model-based feature importances ranking")

print "\nWall time to generate features selection performance plots: ", time.time()-wall
print "Elapsed time to generate features selection performance plots: ", time.clock()-process


# In[182]:

# The SelectFromModel class selects all features that have an importance measure 
# of the feature (as provided by the supervised model) greater than the provided 
# threshold.
#select = SelectFromModel(RandomForestClassifier(min_samples_leaf=0.0001), threshold=0.5)
select = SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=seed))
#p = ["mean", "median"]
p = [0.0001, 0.0005, 0.001, 0.01, 0.1] # any higher crash

clf = Pipeline([('SELECT', select), ('rfe', RandomForestClassifier())])

# Features selected by SelectFromModel using the ExtraTreesClassifier
_= clf.named_steps["SELECT"].fit(df_X, df_y)

importances = clf.named_steps["SELECT"].estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.named_steps["SELECT"].estimator_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot feature ranking
feature_ranking_plot(df_X, indices, "RandomForestClassifier model-based feature importances ranking")

# Plot feature selection
features_selection_model_performance(clf, df_X, df_y, p)


print "Wall time to produce feature ranking plots: ", time.time()-wall
print "Elapsed time to prodduce feature ranking plots: ", time.clock()-process


# In[ ]:

#fig, ax = plt.subplots(figsize=(10, 10))

#sns.heatmap(np.c_[df_X.values, df_y.values],names=features+['y'])

#sns.corrplot(data, ax=ax,names=features+['y'])


# In[43]:

## Overly inblanced datat set between signal and background composition

RANDOM_STATE = 42

class DummySampler(object):

    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)

samplers = [
    ['Standard', DummySampler()],
    ['ADASYN', ADASYN(random_state=RANDOM_STATE)],
    ['ROS', RandomOverSampler(random_state=RANDOM_STATE)],
    ['SMOTE', SMOTE(random_state=RANDOM_STATE)],
]



#smote = SMOTE(random_state=RANDOM_STATE)
#cart = DecisionTreeClassifier(random_state=RANDOM_STATE)
#pipeline = make_pipeline(smote, cart)

#param_range = range(1, 11)
#train_scores, test_scores = ms.validation_curve(
#    pipeline, X, y, param_name="smote__k_neighbors", param_range=param_range,
#    cv=3, scoring=scorer, n_jobs=1)
#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#test_scores_std = np.std(test_scores, axis=1)

#plt.title("Validation Curve with SMOTE-CART")
#plt.xlabel("k_neighbors")
#plt.ylabel("Cohen's kappa")
#plt.plot(param_range, test_scores_mean, color="navy", lw=2)
#plt.legend(loc="best")
#plt.show()


# In[44]:

# Apply the random over-sampling
ros = RandomOverSampler()
X_overresampled, y_overresampled = ros.fit_sample(df_X, df_y)

# Apply the random under-sampling
rus = RandomUnderSampler()
X_underresampled, y_underresampled = rus.fit_sample(df_X, df_y)

# Apply SMOTE SVM
sm = SMOTE(kind='svm')
X_resampled, y_resampled = sm.fit_sample(df_X, df_y)


# In[45]:

# Split data into a development and evaluation set
X_dev,X_eval, y_dev, y_eval = train_test_split(df_X, df_y, test_size=.33, 
                                                   random_state=seed)
# Split development set into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                        random_state=seed+31415)


# ## Repeated nested k-fold cross-validatation
# 
# Change random splitting after each cross-validaiton

# In[63]:

# Split data into a development and evaluation set
X_dev,X_eval, y_dev,y_eval = train_test_split(df_X, df_y,
                                              test_size=0.33, random_state=seed)
# Split development set into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                    random_state=seed)

# Removing features with low variance
select = VarianceThreshold()
#pipe_param_grid = {'variancethreshold__threshold': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]}
pipe_param_grid = {'randomforestclassifier__max_depth': [100],#[70, 100],
                   'randomforestclassifier__max_features': [7], #[4, 6],
                   'variancethreshold__threshold': [0.9] #[0.4, 0.6, 0.8, 1.0]
                  }

grids = [GridSearchCV(make_pipeline(select, RandomForestClassifier(random_state=seed+n_iter)),
                      param_grid=pipe_param_grid, n_jobs=1, verbose=0).fit(X_train, y_train)
         for n_iter in range(10)]

scores = [grid.best_score_ for grid in grids]
print("Average score: %.4f+-%.4f" %(np.mean(scores), sp.stats.sem(scores)))



# ## Machine Learning Algorithms for Model Building
# 
# ### Hyper-parameter Optimization and Model Evaluation
# 
# We employ a nested k-fold cross-validation utilizaiton a grid search for hyper-parameter optimization to avoid leaking information from the training dataset used to validate the hyper-parameters into the model evaluation which uses testing datasets.
# 
# We preform a hyper-parameter optimization to improve the accuary of our Gradient Boosting Classifier model then we evaluate the best cross-validated model in a hold out dataset that was not used during hyper-parameter validation.

# In[46]:

# Default pipeline setup with dummy place holder steps
pipe = Pipeline([('feature_scaling', None), 
                 ('feature_selection', None), 
                 ('classifier', DummyClassifier())])

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = SelectKBest(k=8)

# create classifier for use in scikit-learn
model = GradientBoostingClassifier()

# prepare models: create a mapping of ML classifier name to algorithm
param_grid = [
    {'classifier': [model],
     'classifier__n_estimators': [100],
     'classifier__learning_rate': [0.1],
     'feature_selection': [select],
     'feature_selection__k': [10],
     'feature_scaling': [scaler]
    }
]

pcv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=1)
pcv.fit(X_train, y_train)

y_pred = pcv.predict(X_test)
report = classification_report( y_test, y_pred )
print report


# In[52]:

print("Pipeline steps:\n{}".format(pipe.steps))
# extract the first step 
components = pipe.named_steps["feature_scaling"]
print("components: {}".format(components))
classifier = pcv.best_estimator_.named_steps["classifier"]
print("GradientBoostingClassifier classifier step:\n{}".format(classifier))
print("Best cross-validation accuracy: {:.2f}".format(pcv.best_score_)) 
print("Test set score: {:.2f}".format(pcv.score(X_test, y_test))) 
print("Best parameters: {}".format(pcv.best_params_))


# In[47]:

# Default pipeline setup with dummy place holder steps
pipe = Pipeline([('feature_scaling', None), 
                 ('feature_selection', None), 
                 ('classifier', DummyClassifier())])

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = VarianceThreshold()

# create classifier for use in scikit-learn
model = GradientBoostingClassifier()

# prepare models: create a mapping of ML classifier name to algorithm
param_grid = [
    {'classifier': [model],
     'classifier__n_estimators': [100],
     'classifier__learning_rate': [0.1],
     'feature_selection': [select],
     'feature_selection__threshold': [0.4],
     'feature_scaling': [scaler]
    }
]

pcv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=1)
pcv.fit(X_train, y_train)

y_pred = pcv.predict(X_test)
report = classification_report( y_test, y_pred )
print report


# In[53]:

print("Pipeline steps:\n{}".format(pipe.steps))
# extract the first step 
components = pipe.named_steps["feature_scaling"]
print("components: {}".format(components))
classifier = pcv.best_estimator_.named_steps["classifier"]
print("GradientBoostingClassifier classifier step:\n{}".format(classifier))
print("Best cross-validation accuracy: {:.2f}".format(pcv.best_score_)) 
print("Test set score: {:.2f}".format(pcv.score(X_test, y_test))) 
print("Best parameters: {}".format(pcv.best_params_))


# In[48]:

# Default pipeline setup with dummy place holder steps
pipe = Pipeline([('feature_scaling', None), 
                 ('feature_selection', None), 
                 ('classifier', DummyClassifier())])

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = feature_selection.SelectPercentile(f_classif)

# create classifier for use in scikit-learn
model = GradientBoostingClassifier()

# prepare models: create a mapping of ML classifier name to algorithm
param_grid = [
    {'classifier': [model],
     'classifier__n_estimators': [100],
     'classifier__learning_rate': [0.1],
     'feature_selection': [select],
     'feature_selection__percentile': [90],
     'feature_scaling': [scaler]
    }
]

pcv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=1)
pcv.fit(X_train, y_train)

y_pred = pcv.predict(X_test)
report = classification_report( y_test, y_pred )
print report


# In[54]:

print("Pipeline steps:\n{}".format(pipe.steps))
# extract the first step 
components = pipe.named_steps["feature_scaling"]
print("components: {}".format(components))
classifier = pcv.best_estimator_.named_steps["classifier"]
print("GradientBoostingClassifier classifier step:\n{}".format(classifier))
print("Best cross-validation accuracy: {:.2f}".format(pcv.best_score_)) 
print("Test set score: {:.2f}".format(pcv.score(X_test, y_test))) 
print("Best parameters: {}".format(pcv.best_params_))


# In[49]:

# Default pipeline setup with dummy place holder steps
pipe = Pipeline([('feature_scaling', None), 
                 ('feature_selection', None), 
                 ('classifier', DummyClassifier())])

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = RFE(estimator=RandomForestClassifier(), n_features_to_select=8, step=1)

# create classifier for use in scikit-learn
model = GradientBoostingClassifier()

# prepare models: create a mapping of ML classifier name to algorithm
param_grid = [
    {'classifier': [model],
     'classifier__n_estimators': [100],
     'classifier__learning_rate': [0.1],
     'feature_selection': [select],
     'feature_selection__n_features_to_select': [10],
     'feature_scaling': [scaler]
    }
]

pcv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=1)
pcv.fit(X_train, y_train)

y_pred = pcv.predict(X_test)
report = classification_report( y_test, y_pred )
print report


# In[55]:

print("Pipeline steps:\n{}".format(pipe.steps))
# extract the first step 
components = pipe.named_steps["feature_scaling"]
print("components: {}".format(components))
classifier = pcv.best_estimator_.named_steps["classifier"]
print("GradientBoostingClassifier classifier step:\n{}".format(classifier))
print("Best cross-validation accuracy: {:.2f}".format(pcv.best_score_)) 
print("Test set score: {:.2f}".format(pcv.score(X_test, y_test))) 
print("Best parameters: {}".format(pcv.best_params_))


# In[50]:

# Default pipeline setup with dummy place holder steps
pipe = Pipeline([('feature_scaling', None), 
                 ('feature_selection', None), 
                 ('classifier', DummyClassifier())])

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = RFECV(estimator=RandomForestClassifier(), step=1,  
               cv=StratifiedKFold(3), scoring='accuracy')

# create classifier for use in scikit-learn
model = GradientBoostingClassifier()

# prepare models: create a mapping of ML classifier name to algorithm
param_grid = [
    {'classifier': [model],
     'classifier__n_estimators': [100],
     'classifier__learning_rate': [0.1],
     'feature_selection': [select],
     'feature_selection__cv': [7],
     'feature_scaling': [scaler]
    }
]

pcv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=1)
pcv.fit(X_train, y_train)

y_pred = pcv.predict(X_test)
report = classification_report( y_test, y_pred )
print report


# In[56]:

print("Pipeline steps:\n{}".format(pipe.steps))
# extract the first step 
components = pipe.named_steps["feature_scaling"]
print("components: {}".format(components))
classifier = pcv.best_estimator_.named_steps["classifier"]
print("GradientBoostingClassifier classifier step:\n{}".format(classifier))
print("Best cross-validation accuracy: {:.2f}".format(pcv.best_score_)) 
print("Test set score: {:.2f}".format(pcv.score(X_test, y_test))) 
print("Best parameters: {}".format(pcv.best_params_))


# In[ ]:

# now you can save it to a file
with open('gradboost_model.pkl', 'wb') as f:
    joblib.dump(clf, f,compress=9)

# and later you can load it
clf_load = joblib.load('gradboost_model.pkl')


# In[END]:
