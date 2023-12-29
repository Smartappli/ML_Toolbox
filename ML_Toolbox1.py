# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:45:19 2023

@author: UMONS - 532807
"""

import random
import tkinter as tk
from tkinter import ttk
from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.regression import *

session_seed = random.randrange(1,1000)

variables = {}
models1 = {}
models1_output = {}
models2 = {}
models2_output = {}

root = tk.Tk()
root.title('Ai Toolbox - Machine Learning - Supervised Learning')
root.columnconfigure(0, weight=1)

mc = ttk.Frame(root)
mc.grid(padx=10, pady=10, sticky=(tk.W + tk.E))
mc.columnconfigure(0, weight=1)


def selectAllRegression():
    """Method to select all regressinon models"""
    for i1 in models1.keys():
        models1[i1].set(True)
    models1["regression_lightgbm"].set(False)
    models1["regression_bagging"].set(False)
    models1["regression_stacking"].set(False)
    models1["regression_voting"].set(False)

def unselectAllRegression():
    for i2 in models1.keys():
        models1[i2].set(False)


regression_model = ttk.LabelFrame(mc,
                                  text='Regression Models')
regression_model.grid(padx=5,
                      pady=5,
                      sticky=(tk.W + tk.E))
for i in range(4):
    regression_model.columnconfigure(i, weight=1)

models1["regression_lr"] = tk.BooleanVar()
regression_lr = ttk.Checkbutton(regression_model,
                                text="Linear Regression (lr)",
                                variable=models1["regression_lr"],
                                onvalue=True,
                                offvalue=False)
regression_lr.grid(row=0,
                   column=0,
                   sticky=(tk.W + tk.E))

models1["regression_lasso"] = tk.BooleanVar()
regression_lasso = ttk.Checkbutton(regression_model,
                                   text="Lasso Regression (lasso)",
                                   variable=models1["regression_lasso"],
                                   onvalue=True,
                                   offvalue=False)
regression_lasso.grid(row=0,
                      column=1,
                      sticky=(tk.W + tk.E))

models1["regression_ridge"] = tk.BooleanVar()
regression_ridge = ttk.Checkbutton(regression_model,
                                   text="Ridge Regression (ridge)",
                                   variable=models1["regression_ridge"],
                                   onvalue=True,
                                   offvalue=False)
regression_ridge.grid(row=0,
                      column=2,
                      sticky=(tk.W + tk.E))

models1["regression_en"] = tk.BooleanVar()
regression_en = ttk.Checkbutton(regression_model,
                                text="Elastic Net (en)",
                                variable=models1["regression_en"],
                                onvalue=True,
                                offvalue=False)
regression_en.grid(row=0,
                   column=3,
                   sticky=(tk.W + tk.E))

models1["regression_lar"] = tk.BooleanVar()
regression_lar = ttk.Checkbutton(regression_model,
                                 text="Least Angle Regression (lar)",
                                 variable=models1["regression_lar"],
                                 onvalue=True,
                                 offvalue=False)
regression_lar.grid(row=1,
                    column=0,
                    sticky=(tk.W + tk.E))

models1["regression_llar"] = tk.BooleanVar()
regression_llar = ttk.Checkbutton(regression_model,
                                  text="Lasso Least Angle Regression (llar)",
                                  variable=models1["regression_llar"],
                                  onvalue=True,
                                  offvalue=False)
regression_llar.grid(row=1,
                     column=1,
                     sticky=(tk.W + tk.E))

models1["regression_omp"] = tk.BooleanVar()
regression_omp = ttk.Checkbutton(regression_model,
                                 text="Orthogonal Matching Pursuit (omp)",
                                 variable=models1["regression_omp"],
                                 onvalue=True,
                                 offvalue=False)
regression_omp.grid(row=1,
                    column=2,
                    sticky=(tk.W + tk.E))

models1["regression_br"] = tk.BooleanVar()
regression_br = ttk.Checkbutton(regression_model,
                                text="Bayesian Ridge (br)",
                                variable=models1["regression_br"],
                                onvalue=True,
                                offvalue=False)
regression_br.grid(row=1,
                   column=3,
                   sticky=(tk.W + tk.E))

models1["regression_ard"] = tk.BooleanVar()
regression_ard = ttk.Checkbutton(regression_model,
                                 text="Automatic Relevance Determination (ard)",
                                 variable=models1["regression_ard"],
                                 onvalue=True,
                                 offvalue=False)
regression_ard.grid(row=2,
                    column=0,
                    sticky=(tk.W + tk.E))

models1["regression_par"] = tk.BooleanVar()
regression_par = ttk.Checkbutton(regression_model,
                                 text="Passive Aggressive Regressor (par)",
                                 variable=models1["regression_par"],
                                 onvalue=True,
                                 offvalue=False)
regression_par.grid(row=2,
                    column=1,
                    sticky=(tk.W + tk.E))

models1["regression_ransac"] = tk.BooleanVar()
regression_ransac = ttk.Checkbutton(regression_model,
                                    text="Random Sample Consensus (ransac)",
                                    variable=models1["regression_ransac"],
                                    onvalue=True,
                                    offvalue=False)
regression_ransac.grid(row=2,
                       column=2,
                       sticky=(tk.W + tk.E))

models1["regression_tr"] = tk.BooleanVar()
regression_tr = ttk.Checkbutton(regression_model,
                                text="TheilSen Regressor (tr)",
                                variable=models1["regression_tr"],
                                onvalue=True,
                                offvalue=False)
regression_tr.grid(row=2,
                   column=3,
                   sticky=(tk.W + tk.E))

models1["regression_huber"] = tk.BooleanVar()
regression_huber = ttk.Checkbutton(regression_model,
                                   text="Huber Regressor (huber)",
                                   variable=models1["regression_huber"],
                                   onvalue=True,
                                   offvalue=False)
regression_huber.grid(row=3,
                      column=0,
                      sticky=(tk.W + tk.E))

models1["regression_kr"] = tk.BooleanVar()
regression_kr = ttk.Checkbutton(regression_model,
                                text="Kernel Ridge (kr)",
                                variable=models1["regression_kr"],
                                onvalue=True,
                                offvalue=False)
regression_kr.grid(row=3,
                   column=1,
                   sticky=(tk.W + tk.E))

models1["regression_svm"] = tk.BooleanVar()
regression_svm = ttk.Checkbutton(regression_model,
                                 text="Support Vector Regression (svm)",
                                 variable=models1["regression_svm"],
                                 onvalue=True,
                                 offvalue=False)
regression_svm.grid(row=3,
                    column=2,
                    sticky=(tk.W + tk.E))

models1["regression_knn"] = tk.BooleanVar()
regression_knn = ttk.Checkbutton(regression_model,
                                 text="K Neighbors Regressor (knn)",
                                 variable=models1["regression_knn"],
                                 onvalue=True,
                                 offvalue=False)
regression_knn.grid(row=3,
                    column=3,
                    sticky=(tk.W + tk.E))

models1["regression_dt"] = tk.BooleanVar()
regression_dt = ttk.Checkbutton(regression_model,
                                text="Decision Tree Regressor (dt)",
                                variable=models1["regression_dt"],
                                onvalue=True,
                                offvalue=False)
regression_dt.grid(row=4,
                   column=0,
                   sticky=(tk.W + tk.E))

models1["regression_rf"] = tk.BooleanVar()
regression_rf = ttk.Checkbutton(regression_model,
                                text="Random Forest Regressor (rf)",
                                variable=models1["regression_rf"],
                                onvalue=True,
                                offvalue=False)
regression_rf.grid(row=4,
                   column=1,
                   sticky=(tk.W + tk.E))

models1["regression_et"] = tk.BooleanVar()
regression_et = ttk.Checkbutton(regression_model,
                                text="Extra Trees Regressor (et)",
                                variable=models1["regression_et"],
                                onvalue=True,
                                offvalue=False)
regression_et.grid(row=4,
                   column=2,
                   sticky=(tk.W + tk.E))

models1["regression_ada"] = tk.BooleanVar()
regression_ada = ttk.Checkbutton(regression_model,
                                 text="AdaBoost Regressor (ada)",
                                 variable=models1["regression_ada"],
                                 onvalue=True,
                                 offvalue=False)
regression_ada.grid(row=4,
                    column=3,
                    sticky=(tk.W + tk.E))

models1["regression_gbr"] = tk.BooleanVar()
regression_gbr = ttk.Checkbutton(regression_model,
                                 text="Gradient Boosting Regressor (gbr)",
                                 variable=models1["regression_gbr"],
                                 onvalue=True,
                                 offvalue=False)
regression_gbr.grid(row=5,
                    column=0,
                    sticky=(tk.W + tk.E))

models1["regression_mlp"] = tk.BooleanVar()
regression_mlp = ttk.Checkbutton(regression_model,
                                 text="MLP Regressor (mlp)",
                                 variable=models1["regression_mlp"],
                                 onvalue=True,
                                 offvalue=False)
regression_mlp.grid(row=5,
                    column=1,
                    sticky=(tk.W + tk.E))

models1["regression_xgboost"] = tk.BooleanVar()
regression_xgboost = ttk.Checkbutton(regression_model,
                                     text="Extreme Gradient Boosting (xgboost)",
                                     variable=models1["regression_xgboost"],
                                     onvalue=True,
                                     offvalue=False)
regression_xgboost.grid(row=5,
                        column=2,
                        sticky=(tk.W + tk.E))

models1["regression_lightgbm"] = tk.BooleanVar()
regression_lightgbm = ttk.Checkbutton(regression_model,
                                      text="Light Gradient Boosting Machine (lightgbm)",
                                      variable=models1["regression_lightgbm"],
                                      onvalue=True,
                                      offvalue=False)
regression_lightgbm.grid(row=5,
                         column=3,
                         sticky=(tk.W + tk.E))
regression_lightgbm['state'] = 'disabled'

models1["regression_catboost"] = tk.BooleanVar()
regression_catboost = ttk.Checkbutton(regression_model,
                                      text="CatBoost (catboost)",
                                      variable=models1["regression_catboost"],
                                      onvalue=True,
                                      offvalue=False)
regression_catboost.grid(row=6,
                         column=0,
                         sticky=(tk.W + tk.E))

models1["regression_dummy"] = tk.BooleanVar()
regression_dummy = ttk.Checkbutton(regression_model,
                                   text="Dummy Regressor (dummy)",
                                   variable=models1["regression_dummy"],
                                   onvalue=True,
                                   offvalue=False)
regression_dummy.grid(row=6,
                      column=1,
                      sticky=(tk.W + tk.E))

models1["regression_bagging"] = tk.BooleanVar()
regression_bagging = ttk.Checkbutton(regression_model,
                                     text="Bagging Regressor (bagging)",
                                     variable=models1["regression_bagging"],
                                     onvalue=True,
                                     offvalue=False)
regression_bagging.grid(row=6,
                        column=2,
                        sticky=(tk.W + tk.E))
regression_bagging['state'] = 'disabled'

models1["regression_stacking"] = tk.BooleanVar()
regression_stacking = ttk.Checkbutton(regression_model,
                                      text="Stacking Regressor (stacking)",
                                      variable=models1["regression_stacking"],
                                      onvalue=True,
                                      offvalue=False)
regression_stacking.grid(row=6,
                         column=3,
                         sticky=(tk.W + tk.E))
regression_stacking['state'] = 'disabled'

models1["regression_voting"] = tk.BooleanVar()
regression_voting = ttk.Checkbutton(regression_model,
                                    text="Voting Regressor (voting)",
                                    variable=models1["regression_voting"],
                                    onvalue=True,
                                    offvalue=False)
regression_voting.grid(row=7,
                       column=0,
                       sticky=(tk.W + tk.E))
regression_voting['state'] = 'disabled'


regression_output = ttk.LabelFrame(mc,
                                   text='Output')
regression_output.grid(padx=5,
                       pady=5,
                       sticky=(tk.W + tk.E))
for i in range(4):
    regression_output.columnconfigure(i, weight=1)

models1_output["feature"] = tk.BooleanVar()
regression_feature = ttk.Checkbutton(regression_output,
                                     text="Feature",
                                     variable=models1_output["feature"],
                                     onvalue=True,
                                     offvalue=False)
regression_feature.grid(row=0,
                        column=0,
                        sticky=(tk.W + tk.E))

models1_output["residuals"] = tk.BooleanVar()
regression_residuals = ttk.Checkbutton(regression_output,
                                       text="Residuals",
                                       variable=models1_output["residuals"],
                                       onvalue=True,
                                       offvalue=False)
regression_residuals.grid(row=0,
                          column=1,
                          sticky=(tk.W + tk.E))

models1_output["error"] = tk.BooleanVar()
regression_error = ttk.Checkbutton(regression_output,
                                   text="Error",
                                   variable=models1_output["error"],
                                   onvalue=True,
                                   offvalue=False)
regression_error.grid(row=0,
                      column=2,
                      sticky=(tk.W + tk.E))


regression_info = ttk.LabelFrame(mc, text='Information')
regression_info.grid(padx=5,
                     pady=5,
                     sticky=(tk.W + tk.E))
regression_info.columnconfigure(1, weight=1)


def run1():
    models1_to_compare = []
    if models1["regression_lr"].get():
        models1_to_compare.append("lr")
    if models1["regression_lasso"].get():
        models1_to_compare.append("lasso")
    if models1["regression_ridge"].get():
        models1_to_compare.append("ridge")
    if models1["regression_en"].get():
        models1_to_compare.append("en")
    if models1["regression_lar"].get():
        models1_to_compare.append("lar")
    if models1["regression_llar"].get():
        models1_to_compare.append("llar")
    if models1["regression_omp"].get():
        models1_to_compare.append("omp")
    if models1["regression_br"].get():
        models1_to_compare.append("br")
    if models1["regression_ard"].get():
        models1_to_compare.append("ard")
    if models1["regression_par"].get():
        models1_to_compare.append("par")
    if models1["regression_ransac"].get():
        models1_to_compare.append("ransac")
    if models1["regression_tr"].get():
        models1_to_compare.append("tr")
    if models1["regression_huber"].get():
        models1_to_compare.append("huber")
    if models1["regression_kr"].get():
        models1_to_compare.append("kr")
    if models1["regression_svm"].get():
        models1_to_compare.append("svm")
    if models1["regression_knn"].get():
        models1_to_compare.append("knn")
    if models1["regression_dt"].get():
        models1_to_compare.append("dt")
    if models1["regression_rf"].get():
        models1_to_compare.append("rf")
    if models1["regression_et"].get():
        models1_to_compare.append("et")
    if models1["regression_ada"].get():
        models1_to_compare.append("ada")
    if models1["regression_gbr"].get():
        models1_to_compare.append("gbr")
    if models1["regression_mlp"]:
        models1_to_compare.append("mlp")
    if models1["regression_xgboost"]:
        models1_to_compare.append("xgboost")
    # if models1["regression_lightgbm"]:
        # models1_to_compare.append("lightgdm")
    if models1["regression_catboost"]:
        models1_to_compare.append("catboost")
    if models1["regression_dummy"].get():
        models1_to_compare.append("dummy")
    # if models1["regression_bagging"]:
        # models1_to_compare.append("bagging")
    # if models1["regression_stacking"]:
        # models1_to_compare.append("stacking")
    # if models1["regression_voting"]:
        # models1_to_compare.append("voting")

    ttk.Label(regression_info,
              text="Selected Models: "
                   + ", ".join(str(x) for x in models1_to_compare)).grid(row=0,
                                                                         column=0)

    data1 = get_data('insurance')
    setup(data1,
          target='charges',
          session_id=session_seed)

    best = compare_models(include=models1_to_compare)

    # plot residuals
    if models1_output["residuals"].get():
        plot_model(best, plot='residuals')

    # plot error
    if models1_output["error"].get():
        plot_model(best, plot='error')

    # plot feature importance
    if models1_output["feature"].get():
        plot_model(best, plot='feature')


regression_action = ttk.LabelFrame(mc, text='Actions')
regression_action.grid(padx=5,
                       pady=5,
                       sticky=(tk.W + tk.E))
for i in range(4):
    regression_action.columnconfigure(i, weight=1)


ttk.Button(regression_action,
           text="Select All",
           command=selectAllRegression).grid(row=8,
                                             column=0,
                                             padx=5,
                                             pady=5,
                                             sticky=(tk.W + tk.E))
ttk.Button(regression_action,
           text="Unselect All",
           command=unselectAllRegression).grid(row=8,
                                               column=1,
                                               padx=5,
                                               pady=5,
                                               sticky=(tk.W + tk.E))
ttk.Button(regression_action,
           text="Run Comparison",
           command=run1).grid(row=8,
                              column=3,
                              padx=5,
                              pady=5,
                              sticky=(tk.W + tk.E))


def selectAllClassification():
    for i3 in models2.keys():
        models2[i3].set(True)
    models2["classification_lightgbm"].set(False)


def unselectAllClassification():
    for i4 in models2.keys():
        models2[i4].set(False)


classification_info = ttk.LabelFrame(mc, text='Classification')
classification_info.grid(padx=5,
                         pady=5,
                         sticky=(tk.W + tk.E))
for i in range(4):
    classification_info.columnconfigure(i, weight=1)

models2["classification_lr"] = tk.BooleanVar()
classification_lr = ttk.Checkbutton(classification_info,
                                    text="Logistic Regression (lr)",
                                    variable=models2["classification_lr"],
                                    onvalue=True,
                                    offvalue=False)
classification_lr.grid(row=0,
                       column=0,
                       sticky=(tk.W + tk.E))

models2["classification_knn"] = tk.BooleanVar()
classification_knn = ttk.Checkbutton(classification_info,
                                     text="K Neightbors Classifier (knn)",
                                     variable=models2["classification_knn"],
                                     onvalue=True,
                                     offvalue=False)
classification_knn.grid(row=0,
                        column=1,
                        sticky=(tk.W + tk.E))

models2["classification_nb"] = tk.BooleanVar()
classification_nb = ttk.Checkbutton(classification_info,
                                    text="Gaussian Naive Bayes (nb)",
                                    variable=models2["classification_nb"],
                                    onvalue=True,
                                    offvalue=False)
classification_nb.grid(row=0,
                       column=2,
                       sticky=(tk.W + tk.E))

models2["classification_dt"] = tk.BooleanVar()
classification_dt = ttk.Checkbutton(classification_info,
                                    text="Decision Tree Classifier (dt)",
                                    variable=models2["classification_dt"],
                                    onvalue=True,
                                    offvalue=False)
classification_dt.grid(row=0,
                       column=3,
                       sticky=(tk.W + tk.E))

models2["classification_svm"] = tk.BooleanVar()
classification_svm = ttk.Checkbutton(classification_info,
                                     text="SVM - Linear Kernel (svm)",
                                     variable=models2["classification_svm"],
                                     onvalue=True,
                                     offvalue=False)
classification_svm.grid(row=1,
                        column=0,
                        sticky=(tk.W + tk.E))

models2["classification_rbfsvm"] = tk.BooleanVar()
classification_rbfsvm = ttk.Checkbutton(classification_info,
                                        text="SVM - Radial Kernel (rbfsvm)",
                                        variable=models2["classification_rbfsvm"],
                                        onvalue=True,
                                        offvalue=False)
classification_rbfsvm.grid(row=1,
                           column=1,
                           sticky=(tk.W + tk.E))

models2["classification_gpc"] = tk.BooleanVar()
classification_gpc = ttk.Checkbutton(classification_info,
                                     text="Gaussian Process Classifier (gpc)",
                                     variable=models2["classification_gpc"],
                                     onvalue=True,
                                     offvalue=False)
classification_gpc.grid(row=1,
                        column=2,
                        sticky=(tk.W + tk.E))

models2["classification_mlp"] = tk.BooleanVar()
classification_mlp = ttk.Checkbutton(classification_info,
                                     text="MLP Classifier (mlp)",
                                     variable=models2["classification_mlp"],
                                     onvalue=True,
                                     offvalue=False)
classification_mlp.grid(row=1,
                        column=3,
                        sticky=(tk.W + tk.E))

models2["classification_ridge"] = tk.BooleanVar()
classification_ridge = ttk.Checkbutton(classification_info,
                                       text="Ridge Classifier (ridge)",
                                       variable=models2["classification_ridge"],
                                       onvalue=True,
                                       offvalue=False)
classification_ridge.grid(row=2,
                          column=0,
                          sticky=(tk.W + tk.E))

models2["classification_rf"] = tk.BooleanVar()
classification_rf = ttk.Checkbutton(classification_info,
                                    text="Random Forest Classifier (rf)",
                                    variable=models2["classification_rf"],
                                    onvalue=True,
                                    offvalue=False)
classification_rf.grid(row=2,
                       column=1,
                       sticky=(tk.W + tk.E))

models2["classification_qda"] = tk.BooleanVar()
classification_qda = ttk.Checkbutton(classification_info,
                                     text="Quadratic Discriminant Analysis (qda)",
                                     variable=models2["classification_qda"],
                                     onvalue=True,
                                     offvalue=False)
classification_qda.grid(row=2,
                        column=2,
                        sticky=(tk.W + tk.E))

models2["classification_ada"] = tk.BooleanVar()
classification_ada = ttk.Checkbutton(classification_info,
                                     text="Ada Boost Classifier (ada)",
                                     variable=models2["classification_ada"],
                                     onvalue=True,
                                     offvalue=False)
classification_ada.grid(row=2,
                        column=3,
                        sticky=(tk.W + tk.E))

models2["classification_gbc"] = tk.BooleanVar()
classification_gbc = ttk.Checkbutton(classification_info,
                                     text="Grandient Boost Classifier (bgc)",
                                     variable=models2["classification_gbc"],
                                     onvalue=True,
                                     offvalue=False)
classification_gbc.grid(row=3,
                        column=0,
                        sticky=(tk.W + tk.E))

models2["classification_lda"] = tk.BooleanVar()
classification_lda = ttk.Checkbutton(classification_info,
                                     text="Linear Discriminant Analysis (lda)",
                                     variable=models2["classification_lda"],
                                     onvalue=True,
                                     offvalue=False)
classification_lda.grid(row=3,
                        column=1,
                        sticky=(tk.W + tk.E))

models2["classification_et"] = tk.BooleanVar()
classification_et = ttk.Checkbutton(classification_info,
                                    text="Extra Trees Classifier (et)",
                                    variable=models2["classification_et"],
                                    onvalue=True, offvalue=False)
classification_et.grid(row=3,
                       column=2,
                       sticky=(tk.W + tk.E))

models2["classification_xgboost"] = tk.BooleanVar()
classification_xgboost = ttk.Checkbutton(classification_info,
                                         text="Extreme Gradient Boosting (xgboost)",
                                         variable=models2["classification_xgboost"],
                                         onvalue=True,
                                         offvalue=False)
classification_xgboost.grid(row=3,
                            column=3,
                            sticky=(tk.W + tk.E))
# classification_xgboost['state'] = 'disabled'

models2["classification_lightgbm"] = tk.BooleanVar()
classification_lightgbm = (
    ttk.Checkbutton(classification_info,
                    text="light Gradient Boosting Machine (lightgbm)",
                    variable=models2["classification_lightgbm"],
                    onvalue=True,
                    offvalue=False))
classification_lightgbm.grid(row=4,
                             column=0,
                             sticky=(tk.W + tk.E))
classification_lightgbm['state'] = 'disabled'

models2["classification_catboost"] = tk.BooleanVar()
classification_catboost = (
    ttk.Checkbutton(classification_info,
                    text="CatBoost Classifier (catboost)",
                    variable=models2["classification_catboost"],
                    onvalue=True,
                    offvalue=False))
classification_catboost.grid(row=4,
                             column=1,
                             sticky=(tk.W + tk.E))
# classification_catboost['state'] = 'disabled'

models2["classification_dummy"] = tk.BooleanVar()
classification_dummy = (
    ttk.Checkbutton(classification_info,
                    text="Dummy Classifier (dummy)",
                    variable=models2["classification_dummy"],
                    onvalue=True,
                    offvalue=False))
classification_dummy.grid(row=4,
                          column=2,
                          sticky=(tk.W + tk.E))


classification_output = ttk.LabelFrame(mc, text='Output')
classification_output.grid(padx=5,
                           pady=5,
                           sticky=(tk.W + tk.E))
for i in range(4):
    classification_output.columnconfigure(i, weight=1)



classification_info = ttk.LabelFrame(mc, text='Information')
classification_info.grid(padx=5,
                         pady=5,
                         sticky=(tk.W + tk.E))
classification_info.columnconfigure(1, weight=1)



def run2():
    models2_to_compare = []
    if models2["classification_lr"].get():
        models2_to_compare.append("lr")
    if models2["classification_knn"].get():
        models2_to_compare.append("knn")
    if models2["classification_nb"].get():
        models2_to_compare.append("nb")
    if models2["classification_dt"].get():
        models2_to_compare.append("dt")
    if models2["classification_svm"].get():
        models2_to_compare.append("svm")
    if models2["classification_rbfsvm"].get():
        models2_to_compare.append("rbfsvm")
    if models2["classification_gpc"].get():
        models2_to_compare.append("gpc")
    if models2["classification_mlp"].get():
        models2_to_compare.append("mlp")
    if models2["classification_ridge"].get():
        models2_to_compare.append("ridge")
    if models2["classification_rf"].get():
        models2_to_compare.append("rf")
    if models2["classification_qda"].get():
        models2_to_compare.append("qda")
    if models2["classification_ada"].get():
        models2_to_compare.append("ada")
    if models2["classification_gbc"].get():
        models2_to_compare.append("gbc")
    if models2["classification_lda"].get():
        models2_to_compare.append("lda")
    if models2["classification_et"].get():
        models2_to_compare.append("et")
    if models2["classification_xgboost"]:
        models2_to_compare.append("xgboost")
    # if models2["classification_lightgbm"]:
        # models2_to_compare.append("lightgbm")
    if models2["classification_catboost"]:
        models2_to_compare.append("catboost")
    if models2["classification_dummy"].get():
        models2_to_compare.append("dummy")

    ttk.Label(classification_info,
              text="Selected Models: "
                   + ", ".join(str(x) for x in models2_to_compare)).grid(row=0,
                                                                         column=0)

    data2 = get_data('diabetes')
    exp2 = ClassificationExperiment()
    exp2.setup(data2, target='Class variable', session_id=session_seed)

    compare_classification_models = exp2.compare_models(include=models2_to_compare)
    print(compare_classification_models)


classification_action = ttk.LabelFrame(mc, text='Actions')
classification_action.grid(padx=5,
                           pady=5,
                           sticky=(tk.W + tk.E))
for i in range(4):
    classification_action.columnconfigure(i, weight=1)

ttk.Button(classification_action,
           text="Select All",
           command=selectAllClassification).grid(row=5,
                                                 column=0,
                                                 padx=5,
                                                 pady=5,
                                                 sticky=(tk.W + tk.E))
ttk.Button(classification_action,
           text="Unselect All",
           command=unselectAllClassification).grid(row=5,
                                                   column=1,
                                                   padx=5,
                                                   pady=5,
                                                   sticky=(tk.W + tk.E))
ttk.Button(classification_action,
           text="Run comparison",
           command=run2).grid(row=5,
                              column=3,
                              padx=5,
                              pady=5,
                              sticky=(tk.E + tk.W))

# Show the window
root.mainloop()
