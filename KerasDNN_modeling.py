
# coding: utf-8

# # Data science analysis: Deep  neural network modeling  with Keras
# 
# **Notebook by Christian Contreras-Campana, PhD**
# 
# Lab-notebook (not yet deliverable-notebook)

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
# - maxDeltaEta_jet_jet:  Invariant mass of jet pair DR
# - centrality_tags:  The ratio of the sum of the transverse momentum of all b-tagged jets
# 
# While we have some grasp on the matter, we're not experts, so the following might contain inaccuracies or even outright errors. Feel free to point them out, either in the comments or privately.

# ## Load Libraries
# 
# We load all the necessary python libraries that will permit us to load the data files, pre-process and clean the data, perform data validation, produce statistical summaries, conduct exploratory data analysis, as well as feature transformation, feature ranking, and feature selection. Python libraries will also be needed for model selection, evaluating overfitting, executing standard nested k-fold cross validation for hyper-parameter optimization and model evaluation.  

# In[32]:

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


## Keras deep neural network library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import WeightRegularizer, l1, l2
from keras.models import model_from_json

# Plotting Variables and Correlations 
import pandas.core.common as com
from pandas.core.index import Index

# Import imblearn
import imblearn
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import defaultdict, Counter

import re

# Print version information of some packages
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

get_ipython().magic(u'matplotlib inline')

# Specifying which nodes should be run interactively
InteractiveShell.ast_node_interactivity = "all"
print(__doc__)


# ## Load Data Files
# 
# Most data files contain approximately 15K events. There are a total of 4 files totaling 80K data events. We list the features and response names. We store the data in a Pandas DataFrame for greater ease of data manipulation.
# 
# **Note: To reduce running time of the program we use our most signal-like category which is statistically limited**

# In[33]:

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


# In[34]:

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
signal_sample = "combined/signalData.root"
background_sample = "combined/backgroundData.root"
tree_category = "event_mvaVariables_step7_cate4"

data = load(signal_sample, background_sample, tree_category, features)

print "Total number of events: {}\nNumber of features: {}".format(data.shape[0], data.shape[1])

# Store a copy for later use
df_archived = data.copy(deep=True)

print "\nWall time to read in file input: ", time.time()-wall
print "Elapsed time to read in file input: ", time.clock()-process


# In[35]:

## Print statistical summary of dataset

# To print out all rows and columns to the terminal
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

wall = time.time()
process = time.clock()

print "Head:"
data.head()
print "Information:" 
data.info()
print "Describe:"
data.describe()

print "\nWall time to print statistical summary: ", time.time()-wall
print "Elapsed time to print statistical summary: ", time.clock()-process


# In[36]:

## Define class label counts and percentages

def class_info(classes):
    # Store the number of signal and background events
    class_count = {}
    counts = Counter(classes)
    total = sum(counts.values())

    for cls in counts.keys():
        class_count[class_label[cls]] = counts[cls]
        print("%6s: % 7d  =  % 5.1f%%" 
              % (class_label[cls], counts[cls], float(counts[cls])/float((total))*100.0))

    return (class_count["signal"], class_count["background"])


# In[37]:

# Determine class label counts and percentages
class_label = {0.0: "background", 1.0: "signal"}
class_info(data["y"]);


# In[38]:

## Create features dataframe and target array

df_X = data.drop("y", axis=1, inplace=False)
df_y = data["y"]


# In[39]:

# Create network with Keras: Function to create model, 
# required for KerasClassifier (model architecture)

def create_model(optimizer='rmsprop', init='glorot_uniform', dropout_rate=0.0):
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
    
    # create model: create a simple multi-layer neural network for the problem.

    # Note: initialization of the weights was chose as default to be 
    # randomly drawn from a uniform distribution (if normal then the distribution
    # would have mean 0 and standard deviation 0.05 in keras)

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    # Rectify Linear Unit (Relu) = relu, Exponential Linear Unit (Elu) =  elu
    model.add(Dense(12, input_dim=12, init=init, activation='elu')) 
    # ReLu(x) = {0 for x <=0 else x for x > 0}
    model.add(Dropout(dropout_rate))
     # 8 neurons in the hidden layer and 12 in the visible layer 
    model.add(Dense(8, init=init, activation='elu')) # 8 neurons in the hidden layer 
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, init=init, activation='sigmoid')) # 1 neuron in the output layer

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# In[40]:

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# create classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, nb_epoch=50, batch_size=3, verbose=0)

pipe_classifiers = {
    'SVM':  make_pipeline(scaler, SVC()),
    'NB' :  make_pipeline(scaler, GaussianNB()), 
    'MLP':  make_pipeline(scaler, MLPClassifier()),
    'LR' :  make_pipeline(scaler, LogisticRegression()),
    'ADA':  make_pipeline(None,   AdaBoostClassifier()),
    'KNN':  make_pipeline(scaler, KNeighborsClassifier()),
    'RFC':  make_pipeline(None,   RandomForestClassifier()),
    'CART': make_pipeline(None,   DecisionTreeClassifier(min_samples_leaf=10)),
    'LDA':  make_pipeline(scaler, LinearDiscriminantAnalysis()),
    'GRAD': make_pipeline(None,   GradientBoostingClassifier()),
    'BAGG': make_pipeline(None,   BaggingClassifier()),
    'DNN':  make_pipeline(scaler, model)
}


# In[41]:

## Compute ROC curve and area under the curve

def roc_plot(models, X, y):
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
    ax.set_axis_bgcolor('white')
    
    # Include random by chance 'luck' curve
    plt.plot([1, 0], [0, 1], '--', color=(0.1, 0.1, 0.1), label='Luck')
        
    # Loop through classifiers
    for (name, model) in models.items():
        
        print "\n\x1b[1;31mBuilding model ...\x1b[0m"
        process = time.clock()
        model.fit(X_train, y_train)

        print "\t%s fit time: %.3f"%(name, time.clock()-process)
        
        y_predicted = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            decisions = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            decisions = model.decision_function(X_test)
        
        print "\tArea under ROC curve for %s: %.4f"%(name, roc_auc_score(y_test,decisions))
        
        process = time.clock()
        #scores = cross_val_score(model, X_test, y_test, scoring="roc_auc",
        #                         n_jobs=1, cv=3) #n_jobs=-1
        #print "\tAUC ROC accuracy: %0.5f (+/- %0.5f)"%(scores.mean(), scores.std())
        #print "\tDuration of cross-validation score: ", time.clock()-process

        print classification_report(y_test, y_predicted, target_names=['signal', 'background'])
        print("\tScore of test dataset: {:.5f}".format(model.score(X_test, y_test)))
        
        process = time.clock()
        fpr[name], tpr[name], thresholds = roc_curve(y_test, decisions)
        print "\tArea under ROC time: ", time.clock()-process
        
        roc_auc[name] = auc(fpr[name], tpr[name])
    
    # color choices: https://css-tricks.com/snippets/css/named-colors-and-hex-equivalents/
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 
                    'green', 'yellow', 'SlateBlue', 'DarkSlateGrey',
                    'CadetBlue', 'Chocolate', 'darkred', 'GoldenRod'])
  
    for (name, model), color in zip(models.items(), colors):

        plt.plot(tpr[name], 1-fpr[name], color=color, lw=2,
                 label='%s (AUC = %0.3f)'%(name, roc_auc[name]))                 
    
    # Plot all ROC curves
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic ({} events)".format(X.shape[0]))
    leg = plt.legend(loc="lower left", frameon=True, fancybox=True, fontsize=8) # loc='best'
    leg.get_frame().set_edgecolor('w')
    frame = leg.get_frame()
    frame.set_facecolor('White')
    
    return plt.show()


# In[42]:

# Plot ROC curve

wall = time.time()
process = time.clock()

# Assessing a Classifier's Performance
roc_plot(pipe_classifiers, df_X, df_y)
                   
print "\nWall time to generate ROC curves: ", time.time()-wall
print "Elapsed time to generate ROC curves: ", time.clock()-process


# In[43]:

## Define calibration curve (reliability curve)

def plot_calibration_curve(est, X, y, fig_index):
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

    # We take the no calibration as baseline
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
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("\n\x1b[1;31mclassifier %s:\x1b[0m" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value =             calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "o-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')
 
    # Customize the major grid
    ax1.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax1.set_axis_bgcolor('white')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="best", ncol=1)
    
    # Customize the major grid
    ax2.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax2.set_axis_bgcolor('white')
    
    plt.tight_layout()
    plt.show()


# In[46]:

## Plot reliability curve (i.e. calibration curve)

wall = time.time()
process = time.clock()

#plot_calibration_curve(make_pipeline(None, model), df_X, df_y, 2)
plot_calibration_curve(pipe_classifiers["DNN"], df_X, df_y, 2)

print "\nWall time to generate Calibration curves: ", time.time()-wall
print "Elapsed time to generate Calibration curves: ", time.clock()-process


# In[47]:

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
    ax.set_axis_bgcolor('white')
    
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


# In[48]:

## Overfitting evaluation

wall = time.time()
process = time.clock()

# Uncalibrated model predictions
compare_train_test(pipe_classifiers["DNN"], df_X, df_y, bins=30)

print "\nWall time to generate over-training plots: ", time.time()-wall
print "Elapsed time to generate over-training plots: ", time.clock()-process


# In[30]:

## Visualize weights: Heat Map of neural network weights

# Heat map of the first layer weights in a neural network learned on the HEP dataset
# Source: http://share.pkbigdata.com/ID.16719/CAPTCHA-breaking/src/master/keras-master/tests/manual/check_wrappers.py

# Get the neural network weights
# TensorFlow based weight object (for Theano use a different code syntax)

def dnn_weight_map(classifier):
    
    #W,b = pcv.best_estimator_.named_steps['classifier'].model.layers[0].get_weights()
    W,b = classifier.model.layers[0].get_weights()

    W = np.squeeze(W)
    print("W shape : ", W.shape)

    _= plt.figure(figsize=(10, 8))
    _= plt.imshow(W, interpolation='nearest', cmap='viridis')

    # Heat map
    _= plt.yticks(range(12), features)
    _= plt.xlabel("Columns in weight matrix")
    _= plt.ylabel("Input feature")
    _= plt.colorbar()
    _= plt.grid("off")
    
    return plt.show()
    


# In[49]:

## Visualize weights: Heat Map of neural network weights

dnn_weight_map( pipe_classifiers["DNN"].named_steps['kerasclassifier'])


# In[50]:

# Load dataset
rec_np_data = root2array("combined/run2016Data.root", 
                         "event_mvaVariables_step7_cate4", features)
np_data = rec2array(rec_np_data)


# convert to numpy ndarray into pandas dataframe
df_raw_data = pd.DataFrame(data=np_data, columns=features)

df_raw_data.describe()
df_raw_data.info()

X_data = df_raw_data.values


# In[51]:

# Plot a mva distribution

def over_training_curve(model, mc_X, mc_y, data_X):
    # use subplot to extract axis to add ks and p-value to plot
    fig, ax = plt.subplots()
    
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_axis_bgcolor('white')

    decisions = []
    
    prob_pos = model.predict_proba(data_X)[:, 1] # [:, 1]-->[:][1]
    d1 = model.predict_proba(mc_X[mc_y>0.5])[:, 1]
    d2 = model.predict_proba(mc_X[mc_y<0.5])[:, 1]
        
    decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)  

    plt.hist([decisions[0], decisions[1]], color=['r','b'], alpha=0.5, histtype='stepfilled', 
             normed=False, label=['signal','background'], bins=30, stacked=True)
    
    bins = 30
    hist, bins = np.histogram(prob_pos, bins=bins, range=low_high, normed=False)
    
    scale = len(decisions) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    #err = np.sqrt(hist)
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='black', label='data')

    plt.xlabel("Decision output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    
    return plt.show()


# In[52]:

over_training_curve(pipe_classifiers["DNN"], df_X, df_y, X_data)


# In[ ]:

df_loaded_model = pd.DataFrame(pipe_classifiers["DNN"].predict_proba(X_data)[:,1], columns=['DNN distribution'])
df_loaded_model.hist(bins=30)

plt.show()


# In[53]:

## Prediction score comparison between pipeline model, keras model, and keras classifier model

wall = time.time()
process = time.clock()

#model_json = pipe_classifiers["DNN"].named_steps["kerasclassifier"].model.to_json()
print pipe_classifiers["DNN"].predict_proba(X_data)[0]
scaler.fit(df_X)
Z = scaler.transform(X_data)
print pipe_classifiers["DNN"].named_steps["kerasclassifier"].predict_proba(Z)[0]

print pipe_classifiers["DNN"].named_steps["kerasclassifier"].model.predict_proba(Z)[0]

print "\nTotal wall time of program: ", time.time()-wall
print "Total elapsed process time of program: ", time.clock()-process


# In[54]:

## Model persistence: Store DNN modeling

wall = time.time()
process = time.clock()

model_json = pipe_classifiers["DNN"].named_steps["kerasclassifier"].model.to_json()

with open("keras_dnn_model_tensorflow.json", "w") as json_file:
    json_file.write(model_json)
    
    
# serialize weights to HDF5
pipe_classifiers["DNN"].named_steps["kerasclassifier"].model.save_weights("model_weights.h5")
print("Saved model to disk")


print "\nTotal wall time of program: ", time.time()-wall
print "Total elapsed process time of program: ", time.clock()-process



# In[55]:

# later...

# input file name
filename = 'keras_dnn_model_tensorflow.json'

# load json and create model
json_file = open(filename, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")


# In[56]:

# loaded model prediction scores on unseen data

predict_pos = loaded_model.predict_proba(Z, batch_size=32)
print predict_pos[0]

df_loaded_model =pd.DataFrame(predict_pos, columns=['DNN distribution'])
df_loaded_model.hist(bins=30)

plt.show()


# ## Deep  Neural Network Model Optimization

# In[18]:

## Keras Deep Neural Netork modeling

# split the data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33,
                                                    random_state=seed)

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = SelectKBest(k=4)

clf = KerasClassifier(build_fn=create_model, nb_epoch=20, batch_size=2, verbose=0)

steps = [('scaler', scaler),
         ('feature_selection', select),
         ('keras_dnn', clf)]

pipeline = Pipeline(steps)

parameters = dict(feature_selection__k=[12], 
              keras_dnn__dropout_rate=[0.2])

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)

y_predictions = cv.predict(X_test)
report = classification_report( y_test, y_predictions )
print report


# In[23]:

# Default pipeline setup with dummy place holder steps
pipe = Pipeline([('feature_scaling', None), 
                 ('feature_selection', None), 
                 ('classifier', DummyClassifier())])

# preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# feature selection
select = SelectKBest(k=4)

# create classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, nb_epoch=20, batch_size=2, verbose=0)

# prepare models: create a mapping of ML classifier name to algorithm
param_grid = [
    {'classifier': [model],
     'classifier__dropout_rate': [0.2],
     'feature_selection': [select],
     'feature_selection__k': [12],
     'feature_scaling': [scaler]
    }
]

#pcv = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=2, n_jobs=1)

pcv = GridSearchCV(estimator=pipe, param_grid=param_grid)
pcv.fit(X_train, y_train)

y_pred = pcv.predict(X_test)
report = classification_report( y_test, y_pred )
print report

# LOOK INTO: JSON TO XML converter
# http://stackoverflow.com/questions/28459651/is-there-a-way-to-convert-json-file-to-xml-by-using-groovy-script-on-soapui


# In[27]:


print("Pipeline steps:\n{}".format(pipe.steps))
# extract the first step 
components = pipe.named_steps["feature_scaling"]
print("components: {}".format(components))
classifier = pcv.best_estimator_.named_steps["classifier"]
print("Keras DNN classifier step:\n{}".format(classifier))
print("Best cross-validation accuracy: {:.2f}".format(pcv.best_score_)) 
print("Test set score: {:.2f}".format(pcv.score(X_test, y_test))) 
print("Best parameters: {}".format(pcv.best_params_))


# In[31]:

dnn_weight_map(pcv.best_estimator_.named_steps['classifier'])


# In[57]:

# tuning the following design parameters in our neural networks:
# - The number of hidden layers.
# - The number of neurons per hidden layer.
# - The level of dropout.
# - The learning rate to use with the ADAM optimizer [3].
# - The L1 weight penalty.
# - The L2 weight penalty.

# grid search dropout rate, epochs, batch size, and optimizer
#dropout_rate = [0.2] # [0.0, 0.2, 0.5]
#optimizers = ['adam'] # ['rmsprop', 'adam']
#init = ['uniform'] # ['glorot_uniform', 'normal', 'uniform']
#epochs = np.array([50]) # [50, 100, 150]
#batches = np.array([5])  # [5, 10, 20]

#param_grid = dict(dropout_rate=dropout_rate, optimizer=optimizers,
#                  nb_epoch=epochs, batch_size=batches, init=init)

#pipe_param_grid = {'kerasclassifier__dropout_rate': [0.2]#,
                  #'kerasclassifier__optimizers': ['adam'],
                  #'kerasclassifier__init': ['uniform'],
                  #'kerasclassifier__epochs': np.array([50]),
                  #'kerasclassifier__batches': np.array([5]) 
#                  }


# In[ ]:



