# -*- coding: utf-8 -*-
'''
Run scikit-learn random forest classifier on a features/response dataframe.

The implementRF() function is the whole module; encapsulates typical scikit-learn
formulation to run RandomForestClassifier on a Pandas dataframe containing both
feature columns and a binary `response` column.  Outputs scores and optional
plots.

'''

from collections import Counter
import datetime

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
# Only used in create_tree_plots():
import pygraphviz as pgv
from sklearn import tree


def plotROC(fpr,tpr,auc,ax):
    '''
    Plot ROC curve.

    Example
    -------
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    implement_rf.plotROC(roc[0],roc[1],scores[-1],ax=ax)
    '''
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (AUC = %0.2f)'%auc)


def plotFI(fimp,ax):
    '''
    Plot feature importances

    Example
    -------
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    implement_rf.plotFI(fimp,ax=ax)
    '''

    fimp.sort_values(by='fimport',ascending=False).plot(use_index=False,legend=False,grid=True,ax=ax,style='.-',lw=2,ms=10)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+1, 1))
    ax.set_xticklabels(fimp.sort_values(by='fimport',ascending=False)['fnames'], rotation=90)
    plt.title('Feature Importances')


def implement_sklrf(features, split_ratio=0.33, n_estimators=100,
                    rebal=None, rebal_test=False, textout_scores=False,
                    do_plots=False,textout_valcounts=False,textout_fullsummary=True):
    '''
    Run scikit-learn random forest classifier on Pandas features/response dataframe.

    Parameters
    ----------
    features : Pandas dataframe
        Dataframe containing feature columns as well as a binary response column.
        Feature columns can have any name, but response column must be `response`.
        No NaNs allowed in this dataframe.
    split_ratio : int, optional
        The ratio of the dataset held out for test data (default 0.33).
    textout_valcounts : bool, optional
        Output to screen the response value counts for original, train,
        resampled train, and test datasets.  (default True)
    textout_scores : bool, optional
        Output to screen the scores via held-out test dataset; the same scores
        as in the returned scores output.
    do_plots : bool, optional
        True to output plots of ROC and feature importances to screen (default False).

    Returns
    -------
    scores : list [float, int, int, int, int, float, float, float, float]
        Contains: [accuracy, TP, TN, FP, FN, precision, recall, F1, ROC_AUC]
    response_counts : tuple of Counters
        Contains value_counts for binary response in original dataset, training
        set, resampled training set, and test dataset: (c_orig,c_train,c_res,c_test)
    fimp : Pandas dataframe
        Contains feature names column and feature importance scores from fitted forest.
    roc : tuple of ndarrays of float64s
        Contains the two arrays (fpr,tpr) used to produce the ROC curve.

    Examples
    --------
    scores,(c_orig,c_train,c_res,c_test),fimp,(fpr,tpr),clf = implement_sklrf(
        features, textout_valcounts=False, textout_scores=False, do_plots=False
    )

    '''

    class _NoResampler(object):
        '''
        A placeholder to replace imbalance-learn resamplers when not using them.
        '''
        def sample(self, X, y):
            return X, y
        def fit(self, X, y):
            return self
        def fit_sample(self, X, y):
            return self.sample(X, y)


    all_but_response = list(features.columns.values)
    all_but_response.remove('response')
    X = features[all_but_response].values
    y = features['response'].values.astype(int)  # ensuring discrete response var for classification

    fimp = pd.DataFrame({'fnames':features[all_but_response].columns.values})
    scores = pd.DataFrame(columns=['acc','tp','tn','fp','fn','prec','recall','f1'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, stratify=y)
    # (note the above call with "stratify" arg is a wrapper for StratifiedShuffleSplit()
    # rather than StratifiedKFold()...)

    if rebal=='under':
        smp = RandomUnderSampler()
        clf = RandomForestClassifier(n_estimators=n_estimators)
    elif rebal=='overs':
        smp = RandomOverSampler()
        clf = RandomForestClassifier(n_estimators=n_estimators)
    elif rebal=='smote':
        smp = SMOTE()
        clf = RandomForestClassifier(n_estimators=n_estimators)
    elif rebal=='cwbal':
        smp = _NoResampler()
        clf = RandomForestClassifier(n_estimators=n_estimators,class_weight='balanced')
    elif rebal==None:
        smp = _NoResampler()
        clf = RandomForestClassifier(n_estimators=n_estimators)

    X_res, y_res = smp.fit_sample(X_train, y_train)
    c_orig = Counter(y)
    c_train = Counter(y_train)
    c_test = Counter(y_test)
    c_trainres = Counter(y_res)

    if rebal_test:
        if rebal=='rfc':
            print('WARNING: note that setting rebal_test=True with rebal=\'rfc\' does nothing.')
        X_test, y_test = smp.fit_sample(X_test, y_test)
        c_testres = Counter(y_test)
        response_counts = (c_orig,c_train,c_trainres,c_test,c_testres)
    else:
        c_testres = c_test
        response_counts = (c_orig,c_train,c_trainres,c_test)

    if textout_valcounts:
        print('Test/train split ratio:  test set is %0.2f of original dataset.'%split_ratio)
        print('Original dataset response value counts:  {0: %d, 1: %d}'%(c_orig[0],c_orig[1]))
        if (rebal=='under') or (rebal=='overs') or (rebal=='smote'):
            print('Training set response values before/after under-sampling:  {0: %d, 1: %d} / {0: %d, 1: %d}'%(c_train[0],c_train[1],c_trainres[0],c_trainres[1]))
        else:
            print('Training set response values:  {0: %d, 1: %d}'%(c_train[0],c_train[1]))
        if (rebal_test) and ((rebal=='under') or (rebal=='overs') or (rebal=='smote')):
            print('Test set response values before/after under-sampling:  {} / {}'.format(c_test,c_testres))
        else:
            print('Test set response values:  {0: %d, 1: %d}'%(c_test[0],c_test[1]))
        print(' ')

    probas_ = clf.fit(X_res, y_res).predict_proba(X_test)  # predicted class probabilities
    ypred = clf.predict(X_test)  # predicted classes

    acc = accuracy_score(y_test, ypred)
    tn, fp, fn, tp = confusion_matrix(y_test, ypred).ravel()
    prec = precision_score(y_test, ypred)
    recall = recall_score(y_test, ypred)
    f1 = f1_score(y_test, ypred)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    scores=[acc,tp,tn,fp,fn,prec,recall,f1,roc_auc]

    fimp = pd.DataFrame({'fnames':features[all_but_response].columns.values,
                         'fimport':pd.Series(clf.feature_importances_)})

    if textout_fullsummary:
        counts = ' train:%d/%d  test:%d/%d'%(c_trainres[0],c_trainres[1],
                                             c_testres[0],c_testres[1])
        print('Pre=%0.2f'%prec,'Rec=%0.2f'%recall,
              'F1=%0.2f'%f1,'AUC=%0.2f'%roc_auc,'Acc=%0.2f'%acc,
              'TP=%-3d'%tp,'TN=%-3d'%tn,'FP=%-3d'%fp,'FN=%-3d'%fn,
              'n=%0d '%n_estimators,'%5s'%rebal,counts
              )
    elif textout_scores:
        print('Prec=%0.2f '%prec,'Recall=%0.2f '%recall,
              'F1=%0.2f '%f1,'AUC=%0.2f '%roc_auc,'Accu=%0.2f '%acc,
              'TP=%-d '%tp,'TN=%-d '%tn,'FP=%-d '%fp,'FN=%-d '%fn,
              'n=%0d '%n_estimators
              )

    if do_plots:
        fig = plt.figure(figsize=(14,4))
        ax = fig.add_subplot(121)
        plotROC(fpr,tpr,roc_auc,ax)  # plotting ROC curve
        ax = fig.add_subplot(122)
        plotFI(fimp,ax)  # plotting feature importances
        plt.show()

    return scores,response_counts,fimp,(fpr,tpr),clf



def create_tree_plots(clf,feature_names,class_names,N,max_depth=1,out_html=False):
    '''
    Create *.png tree plots for the given RF classifier's first N trees.

    Parameters
    ----------
    clf : class sklearn.ensemble.RandomForestClassifier
        The estimator class instance output from implement_sklrf().

    '''

    for i,tree_in_forest in enumerate(clf.estimators_[:N]):
        dot_data = tree.export_graphviz(tree_in_forest, out_file=None,
                                        filled=True, rounded=True,
                                        max_depth=max_depth,
                                        feature_names=feature_names,
                                        class_names=class_names)
        g=pgv.AGraph(dot_data)
        g.draw('tree_'+str(i)+'.png',prog='dot')

    # Listing tree text contents rather than graphics, here are top split
    # rules for each tree:
    #for line in dot_data.splitlines():
    #    if line.startswith('0 [label="'):
    #        print( re.sub( 'ngini.*$', '', line.replace('0 [label="','')) )

    if out_html:
        print('To cut/paste into Jupyter cell or a webpage...\n')
        print('<table align="left" width="90%"><tr>')
        for j in range(N//4):
            for i in range(4):
                print('<td width="20%"><img src="tree_{}.png"></td>'.format(i+4*j))
            if j < N//4-1:
                print('</tr><tr>')
        print('</tr></table>')
