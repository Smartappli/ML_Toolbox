# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:38:04 2023

@author: UMONS - 532807
"""

import tkinter as tk
from tkinter import ttk
from pycaret.datasets import get_data
from pycaret.time_series import *

variables = dict()
models = dict()

root = tk.Tk()
root.title('Ai Toolbox - Machine Learning - Time Series Analysis')
root.columnconfigure(0, weight=1)

mc = ttk.Frame(root)
mc.grid(padx=10, pady=10, sticky=(tk.W + tk.E))
mc.columnconfigure(0, weight=1)


def selectAllTS():
    for i in models.keys():
        models[i].set(True)


def unselectAllTS():
    for i in models.keys():
        models[i].set(False)


ts_info = ttk.LabelFrame(mc, text='Time Series')
ts_info.grid(padx=5, pady=5, sticky=(tk.W + tk.E))
for i in range(4):
    ts_info.columnconfigure(i, weight=1)

models["ts_naive"] = tk.BooleanVar()
ts_naive = ttk.Checkbutton(ts_info, text="Naive Forecaster (naive)", variable=models["ts_naive"], onvalue=1, offvalue=0)
ts_naive.grid(row=0, column=0, sticky=(tk.W + tk.E))

models["ts_grand_means"] = tk.BooleanVar()
ts_grand_means = ttk.Checkbutton(ts_info, text="Grand Means Forecaster (grand_means)",
                                 variable=models["ts_grand_means"], onvalue=1, offvalue=0)
ts_grand_means.grid(row=0, column=1, sticky=(tk.W + tk.E))

models["ts_snaive"] = tk.BooleanVar()
ts_snaive = ttk.Checkbutton(ts_info, text="Seasonal Naive Forecaster (snaive)", variable=models["ts_snaive"], onvalue=1,
                            offvalue=0)
ts_snaive.grid(row=0, column=2, sticky=(tk.W + tk.E))

models["ts_polytrend"] = tk.BooleanVar()
ts_polytrend = ttk.Checkbutton(ts_info, text="Polynomial Trend Forecaster (polytrend)", variable=models["ts_polytrend"],
                               onvalue=1, offvalue=0)
ts_polytrend.grid(row=0, column=3, sticky=(tk.W + tk.E))

models["ts_arima"] = tk.BooleanVar()
ts_arima = ttk.Checkbutton(ts_info, text="ARIMA (arima)", variable=models["ts_arima"], onvalue=1, offvalue=0)
ts_arima.grid(row=1, column=0, sticky=(tk.W + tk.E))

models["ts_auto_arima"] = tk.BooleanVar()
ts_auto_arima = ttk.Checkbutton(ts_info, text="Auto ARIMA (auto_arima)", variable=models["ts_auto_arima"], onvalue=1,
                                offvalue=0)
ts_auto_arima.grid(row=1, column=1, sticky=(tk.W + tk.E))

models["ts_exp_smooth"] = tk.BooleanVar()
ts_exp_smooth = ttk.Checkbutton(ts_info, text="Exponential Smoothing (exp_smooth)", variable=models["ts_exp_smooth"],
                                onvalue=1, offvalue=0)
ts_exp_smooth.grid(row=1, column=2, sticky=(tk.W + tk.E))

models["ts_ets"] = tk.BooleanVar()
ts_ets = ttk.Checkbutton(ts_info, text="ETS (ets)", variable=models["ts_ets"], onvalue=1, offvalue=0)
ts_ets.grid(row=1, column=3, sticky=(tk.W + tk.E))

models["ts_theta"] = tk.BooleanVar()
ts_theta = ttk.Checkbutton(ts_info, text="Theta Forecaster (theta)", variable=models["ts_theta"], onvalue=1, offvalue=0)
ts_theta.grid(row=2, column=0, sticky=(tk.W + tk.E))

models["ts_stlf"] = tk.BooleanVar()
ts_stlf = ttk.Checkbutton(ts_info, text="STLF (stlf)", variable=models["ts_stlf"], onvalue=1, offvalue=0)
ts_stlf.grid(row=2, column=1, sticky=(tk.W + tk.E))

models["ts_croston"] = tk.BooleanVar()
ts_croston = ttk.Checkbutton(ts_info, text="Croston (croston)", variable=models["ts_croston"], onvalue=1, offvalue=0)
ts_croston.grid(row=2, column=2, sticky=(tk.W + tk.E))

models["ts_bats"] = tk.BooleanVar()
ts_bats = ttk.Checkbutton(ts_info, text="BATS (bats)", variable=models["ts_bats"], onvalue=1, offvalue=0)
ts_bats.grid(row=2, column=3, sticky=(tk.W + tk.E))

models["ts_tbats"] = tk.BooleanVar()
ts_tbats = ttk.Checkbutton(ts_info, text="TBATS (tbats)", variable=models["ts_tbats"], onvalue=1, offvalue=0)
ts_tbats.grid(row=3, column=0, sticky=(tk.W + tk.E))

models["ts_prophet"] = tk.BooleanVar()
ts_prophet = ttk.Checkbutton(ts_info, text="Prophet (prophet)", variable=models["ts_prophet"], onvalue=1, offvalue=0)
ts_prophet.grid(row=3, column=1, sticky=(tk.W + tk.E))

models["ts_lr_cds_dt"] = tk.BooleanVar()
ts_lr_cds_dt = ttk.Checkbutton(ts_info, text="Linear w/ Cond. Deseasonalize & Detrending (lr_cds_dt)",
                               variable=models["ts_lr_cds_dt"], onvalue=1, offvalue=0)
ts_lr_cds_dt.grid(row=3, column=2, sticky=(tk.W + tk.E))

models["ts_en_cds_dt"] = tk.BooleanVar()
ts_en_cds_dt = ttk.Checkbutton(ts_info, text="Elastic Net w/ Cond. Deseasonalize & Detrending (en_cds_dt)",
                               variable=models["ts_en_cds_dt"], onvalue=1, offvalue=0)
ts_en_cds_dt.grid(row=3, column=3, sticky=(tk.W + tk.E))

models["ts_ridge_cds_dt"] = tk.BooleanVar()
ts_ridge_cds_dt = ttk.Checkbutton(ts_info, text="Ridge w/ Cond. Deseasonalize & Detrending (ridge_cds_dt)",
                                  variable=models["ts_ridge_cds_dt"], onvalue=1, offvalue=0)
ts_ridge_cds_dt.grid(row=4, column=0, sticky=(tk.W + tk.E))

models["ts_lasso_cds_dt"] = tk.BooleanVar()
ts_lasso_cds_dt = ttk.Checkbutton(ts_info, text="Lasso w/ Cond. Deseasonalize & Detrending (lasso_cds_dt)",
                                  variable=models["ts_lasso_cds_dt"], onvalue=1, offvalue=0)
ts_lasso_cds_dt.grid(row=4, column=1, sticky=(tk.W + tk.E))

models["ts_lar_cds_dt"] = tk.BooleanVar()
ts_lar_cds_dt = ttk.Checkbutton(ts_info,
                                text="Least Angular Regressor w/ Cond. Deseasonalize & Detrending (lar_cds_dt)",
                                variable=models["ts_lar_cds_dt"], onvalue=1, offvalue=0)
ts_lar_cds_dt.grid(row=4, column=2, sticky=(tk.W + tk.E))

models["ts_llar_cds_dt"] = tk.BooleanVar()
ts_llar_cds_dt = ttk.Checkbutton(ts_info,
                                 text="Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending (llar_cds_dt)",
                                 variable=models["ts_llar_cds_dt"], onvalue=1, offvalue=0)
ts_llar_cds_dt.grid(row=4, column=3, sticky=(tk.W + tk.E))

models["ts_br_cds_dt"] = tk.BooleanVar()
ts_br_cds_dt = ttk.Checkbutton(ts_info, text="Bayesian Ridge w/ Cond. Deseasonalize & Detrending (bs_cds_dt)",
                               variable=models["ts_br_cds_dt"], onvalue=1, offvalue=1)
ts_br_cds_dt.grid(row=5, column=0, sticky=(tk.W + tk.E))

models["ts_huber_cds_dt"] = tk.BooleanVar()
ts_huber_cds_dt = ttk.Checkbutton(ts_info, text="Huber w/ Cond. Deseasonalize & Detrending (huber_cds_dt)",
                                  variable=models["ts_huber_cds_dt"], onvalue=1, offvalue=0)
ts_huber_cds_dt.grid(row=5, column=1, sticky=(tk.W + tk.E))

models["ts_par_cds_dt"] = tk.BooleanVar()
ts_par_cds_dt = ttk.Checkbutton(ts_info, text="Passive Aggressive w/ Cond. Deseasonalize & Detrending (par_cds_dt)",
                                variable=models["ts_par_cds_dt"], onvalue=1, offvalue=0)
ts_par_cds_dt.grid(row=5, column=2, sticky=(tk.W + tk.E))

models["ts_omp_cds_dt"] = tk.BooleanVar()
ts_omp_cds_dt = ttk.Checkbutton(ts_info,
                                text="Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending (omp_cds_dt)",
                                variable=models["ts_omp_cds_dt"], onvalue=1, offvalue=0)
ts_omp_cds_dt.grid(row=5, column=3, sticky=(tk.W + tk.E))

models["ts_knn_cds_dt"] = tk.BooleanVar()
ts_knn_cds_dt = ttk.Checkbutton(ts_info, text="K Neighbors w/ Cond. Deseasonalize & Detrending (knn_cds_dt)",
                                variable=models["ts_knn_cds_dt"], onvalue=1, offvalue=0)
ts_knn_cds_dt.grid(row=6, column=0, sticky=(tk.W + tk.E))

models["ts_dt_cds_dt"] = tk.BooleanVar()
ts_dt_cds_dt = ttk.Checkbutton(ts_info, text="Decision Tree w/ Cond. Deseasonalize & Detrending (dt_cds_dt)",
                               variable=models["ts_dt_cds_dt"], onvalue=1, offvalue=0)
ts_dt_cds_dt.grid(row=6, column=1, sticky=(tk.W + tk.E))

models["ts_rf_cds_dt"] = tk.BooleanVar()
ts_rf_cds_dt = ttk.Checkbutton(ts_info, text="Random Forest w/ Cond. Deseasonalize & Detrending (rf_cds_dt)",
                               variable=models["ts_rf_cds_dt"], onvalue=1, offvalue=0)
ts_rf_cds_dt.grid(row=6, column=2, sticky=(tk.W + tk.E))

models["ts_et_cds_dt"] = tk.BooleanVar()
ts_rf_cds_dt = ttk.Checkbutton(ts_info, text="Extra Trees w/ Cond. Deseasonalize & Detrending (et_cds_dt)",
                               variable=models["ts_et_cds_dt"], onvalue=1, offvalue=0)
ts_rf_cds_dt.grid(row=6, column=3, sticky=(tk.W + tk.E))

models["ts_gbr_cds_dt"] = tk.BooleanVar()
ts_gbr_cds_dt = ttk.Checkbutton(ts_info, text="Gradient Boosting w/ Cond. Deseasonalize & Detrending (gbr_cds_dt)",
                                variable=models["ts_gbr_cds_dt"], onvalue=1, offvalue=0)
ts_gbr_cds_dt.grid(row=7, column=0, sticky=(tk.W + tk.E))

models["ts_ada_cds_dt"] = tk.BooleanVar()
ts_ada_cds_dt = ttk.Checkbutton(ts_info, text="AdaBoost w/ Cond. Deseasonalize & Detrending (ada_cds_dt)",
                                variable=models["ts_ada_cds_dt"], onvalue=1, offvalue=0)
ts_ada_cds_dt.grid(row=7, column=1, sticky=(tk.W + tk.E))

models["ts_xgboost_cds_dt"] = tk.BooleanVar()
ts_xgboost_cds_dt = ttk.Checkbutton(ts_info,
                                    text="Extreme Gradient Boosting w/ Cond. Deseasonalize & Detrending (xgboost_cds_dt)",
                                    variable=models["ts_xgboost_cds_dt"], onvalue=1, offvalue=0)
ts_xgboost_cds_dt.grid(row=7, column=2, sticky=(tk.W + tk.E))

models["ts_lightgbm_cds_dt"] = tk.BooleanVar()
ts_lightgbm_cds_dt = ttk.Checkbutton(ts_info,
                                     text="Light Gradient Boosting w/ Cond. Deseasonalize & Detrending (lightgbm_cds_dt)",
                                     variable=models["ts_lightgbm_cds_dt"], onvalue=1, offvalue=0)
ts_lightgbm_cds_dt.grid(row=7, column=3, sticky=(tk.W + tk.E))

models["ts_catboost_cds_dt"] = tk.BooleanVar()
ts_catboost_cds_dt = ttk.Checkbutton(ts_info,
                                     text="CatBoost Regressor w/ Cond. Deseasonalize & Detrending (catboost_cds_dt)",
                                     variable=models["ts_catboost_cds_dt"], onvalue=1, offvalue=0)
ts_catboost_cds_dt.grid(row=8, column=0, sticky=(tk.W + tk.E))

def run():
    models_to_compare = []
    if models["ts_naive"].get():
        models_to_compare.append("naive")
    if models["ts_grand_means"].get():
        models_to_compare.append("grand_mean")
    if models["ts_snaive"].get():
        models_to_compare.append("snaive")
    if models["ts_polytrend"].get():
        models_to_compare.append("polytrend")
    if models["ts_arima"].get():
        models_to_compare.append("arima")
    if models["ts_auto_arima"].get():
        models_to_compare.append("auto_arima")
    if models["ts_exp_smooth"].get():
        models_to_compare.append("exp_smooth")
    if models["ts_ets"].get():
        models_to_compare.append("ets")
    if models["ts_theta"].get():
        models_to_compare.append("theta")
    if models["ts_stlf"].get():
        models_to_compare.append("stlf")
    if models["ts_croston"].get():
        models_to_compare.append("croston")
    if models["ts_bats"].get():
        models_to_compare.append("bats")
    if models["ts_tbats"].get():
        models_to_compare.append("tbats")
    if models["ts_prophet"].get():
        models_to_compare.append("prophet")
    if models["ts_lr_cds_dt"].get():
        models_to_compare.append("lr_cds_dt")
    if models["ts_en_cds_dt"].get():
        models_to_compare.append("en_cds_dt")
    if models["ts_ridge_cds_dt"].get():
        models_to_compare.append("ridge_cds_dt")
    if models["ts_lasso_cds_dt"].get():
        models_to_compare.append("lasso_cds_dt")
    if models["ts_lar_cds_dt"].get():
        models_to_compare.append("lar_cds_dt")
    if models["ts_llar_cds_dt"].get():
        models_to_compare.append("llar_cds_dt")
    if models["ts_br_cds_dt"].get():
        models_to_compare.append("br_cds_dt")
    if models["ts_huber_cds_dt"].get():
        models_to_compare.append("huber_cds_dt")
    if models["ts_par_cds_dt"].get():
        models_to_compare.append("par_cds_dt")
    if models["ts_omp_cds_dt"].get():
        models_to_compare.append("omp_cds_dt")
    if models["ts_knn_cds_dt"].get():
        models_to_compare.append("knn_cds_dt")
    if models["ts_dt_cds_dt"].get():
        models_to_compare.append("dt_cds_dt")
    if models["ts_rf_cds_dt"].get():
        models_to_compare.append("rf_cds_dt")
    if models["ts_et_cds_dt"].get():
        models_to_compare.append("et_cds_dt")
    if models["ts_gbr_cds_dt"].get():
        models_to_compare.append("gbr_cds_dt")
    if models["ts_ada_cds_dt"].get():
        models_to_compare.append("ada_cds_dt")
    if models["ts_xgboost_cds_dt"].get():
        models_to_compare.append("xgboost_cds_dt")
    if models["ts_lightgbm_cds_dt"].get():
        models_to_compare.append("lightgbm_cds_dt")
    if models["ts_catboost_cds_dt"].get():
        models_to_compare.append("catboost_cds_dt")

    data = get_data('airline')
    s = setup(data, fh=3, session_id=123)

    compare_ts_models = compare_models(include=models_to_compare)

    # plot forecast
    plot_model(compare_ts_models, plot='forecast')

    # residuals plot
    plot_model(compare_ts_models, plot='residuals')


ttk.Button(ts_info, text="Select All", command=selectAllTS).grid(row=8, column=0, padx=5, pady=5, sticky=(tk.W + tk.E))
ttk.Button(ts_info, text="unselect All", command=unselectAllTS).grid(row=8, column=1, padx=5, pady=5,
                                                                     sticky=(tk.W + tk.E))
ttk.Button(ts_info, text="Run comparison", command=run).grid(row=8, column=3, padx=5, pady=5, sticky=(tk.W + tk.E))


# Show the window
root.mainloop()