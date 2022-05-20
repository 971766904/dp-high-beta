# @Time : 2022/3/21 16:38 
# @Author : zhongyu 
# @File : roc_compare.py
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

if __name__ == '__main__':
    # roc_data_H_H = np.load('LHdataset\H_H_roc.npy')
    # roc_data_L_H = np.load('LHdataset\L_H_5_roc.npy')
    # roc_data_LH_H = np.load('LHdataset\LHmix_H_roc.npy')
    # roc_data_H10_H = np.load('LHdataset\H10_H_roc.npy')

    dsp = lgb.Booster(model_file='modeltest/model_L_5_20.txt')
    print('Plotting feature importances...')
    ax = lgb.plot_importance(dsp, max_num_features=10)
    plt.show()

    # fig = plt.figure()
    # import matplotlib.font_manager as fm
    #
    # # 微软雅黑,如果需要宋体,可以用simsun.ttc
    # myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
    # fronts = {'family': 'Times New Roman', 'size': 12}
    # plt.plot(roc_data_H_H[0], roc_data_H_H[1], label='H_H')
    # plt.plot(roc_data_H10_H[0], roc_data_H10_H[1], label='H10_H')
    # plt.plot(roc_data_L_H[0], roc_data_L_H[1], label='L_H')
    # plt.plot(roc_data_LH_H[0], roc_data_LH_H[1], label='LHmix_H')
    # plt.xlabel('Fpr', fontproperties=myfont)
    # plt.ylabel('Tpr', fontproperties=myfont)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xticks(fontproperties='Times New Roman', fontsize=12)
    # plt.yticks(fontproperties='Times New Roman', fontsize=12)
    # plt.legend(loc="lower right", prop=myfont)
