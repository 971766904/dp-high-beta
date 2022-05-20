# @Time : 2022/2/23 19:08 
# @Author : zhongyu 
# @File : shap_analysis.py
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def shap_summary(data_file, model_f):
    df_validation = pd.read_csv(data_file, index_col=0)
    dsp = lgb.Booster(model_file=model_f)

    test_data = df_validation.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
    dsp.params["objective"] = "binary"
    explainer = shap.TreeExplainer(dsp)
    shap_values = explainer.shap_values(test_data)

    plt.figure()
    shap.summary_plot(shap_values[1], test_data)
    plt.tight_layout()


if __name__ == '__main__':
    print('Loading data&model...')
    # load or create your dataset
    # # shap_summary各模型对比
    data_L = 'LHdataset/topdata_test.csv'
    data_H = 'LHdataset/topdata_H_test.csv'

    model_L = 'model/model_L_5_20.txt'
    model_H = 'model/model_H.txt'
    model_mix = 'model/modelL_H_mix9_100.txt'
    model_10 = 'model/model_H10.txt'
    # shap_summary(data_L, model_L)
    # shap_summary(data_H, model_H)
    # shap_summary(data_H, model_mix)
    shap_summary(data_H, model_10)

    # # shap分析
    # df_validation = pd.read_csv('LHdataset/topdata_test.csv', index_col=0)
    #
    # # dsp = lgb.Booster(model_file='model/model_1.txt')
    # # dsp = lgb.Booster(model_file='model/model_1_10_40_300.txt')
    # dsp = lgb.Booster(model_file='model/model_L_5_20.txt')
    # columns = ['Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",
    #            "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
    #            "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
    #            "DENSITY", "W_E", "FIR01", "FIR03"]
    #
    # test_data = df_validation.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1)
    # y = df_validation['disrup_tag']
    # dsp.params["objective"] = "binary"
    # explainer = shap.TreeExplainer(dsp)
    # shap_values = explainer.shap_values(test_data)
    #
    # # 特征聚类
    # clustering = shap.utils.hclust(test_data, y)
    # shap.plots.bar(shap_values,
    #                clustering=clustering,
    #                clustering_cutoff=0.5)
    #
    # # plt.figure(dpi=1200)
    # # fig = plt.gcf()
    # # shap.summary_plot(shap_values[1], test_data, plot_type="bar", show=False)
    # # plt.tight_layout()
    # # plt.savefig('filename.png')
    # # fig = plt.gcf()
    # # shap.dependence_plot("FIR03", shap_values[1], test_data)
    # # plt.tight_layout()
    # #
    # # for name in columns:
    # #     plt.figure(dpi=1200)
    # #     fig = plt.gcf()
    # #     shap.dependence_plot(name, shap_values[1], test_data, interaction_index="BT", show=False)
    # #     plt.tight_layout()
    # #     plt.savefig('shap_fig/{}.png'.format(name))
    # #     plt.close(fig)
