# @Time : 2022/2/18 17:03 
# @Author : zhongyu 
# @File : test_analysis_shot.py


import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn import metrics


def assess1(validset_b, df_validation, a1, delta_t, dsp):
    predict_result = np.empty([0, 3])
    for i in range(validset_b.shape[0]):
        va_shot = validset_b[i, 0]
        if validset_b[i, 1]:
            validset_b[i, 1] = 1
        dis = df_validation[df_validation['#'] == va_shot]
        X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y = dsp.predict(X, num_iteration=dsp.best_iteration)
        time_shot = dis['time'].values
        shot_predict = 0
        for j in range(len(y) - delta_t):
            subset = y[j:j + delta_t]
            if subset.min() > a1:
                shot_predict = 1
                break
        if shot_predict:
            t_warn = time_shot[j + delta_t]
        else:
            t_warn = 0
        predict_result = np.append(predict_result, [[va_shot, shot_predict, t_warn]], axis=0)
    return predict_result


if __name__ == '__main__':
    print('Loading data&model...')
    # load or create your dataset
    H_beta_shot = np.load(r'LHdataset\H_beta_test.npy')
    df_validation = pd.read_csv('LHdataset/topdata_H_test.csv', index_col=0)
    roc_data2a7 = np.load('LHdataset\L_H_mix_roc.npy')

    # # 用训练集测试判定规则
    # testset_b = np.load(r'dataset\train_dis_shot.npy')
    # testset_a = np.load(r'dataset\train_undis_shot.npy')
    # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)

    # validset_b = np.append(testset_a, testset_b, axis=0)
    validset_b = H_beta_shot

    # dsp = lgb.Booster(model_file='model/model_1.txt')
    # dsp = lgb.Booster(model_file='model/model_1_10_40_300.txt')
    dsp = lgb.Booster(model_file='model/model_H10.txt')

    # # 1.预警时间
    # a1 = 0.9
    # delta_t = 1
    # predict_result = assess1(validset_b, df_validation, a1, delta_t, dsp)
    # prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_result[:, 1], average='binary')
    # print(metrics.classification_report(validset_b[:, 1], predict_result[:, 1]))
    # import seaborn as sns
    # warn_time = []
    # a=0
    # b=0
    # c=0
    # for i in range(validset_b.shape[0]):
    #     if predict_result[i,1] and validset_b[i,1]:
    #         time_w = validset_b[i,2]/1000-predict_result[i,2]
    #         warn_time.append(time_w)
    #         if time_w>0.005 and time_w<=0.1:
    #             a+=1
    #         if time_w>0.1 and time_w<=0.3:
    #             b+=1
    #         if time_w>0.3:
    #             c+=1
    # ax = sns.distplot(warn_time)

    # # 2.Tp&Fp best:a1=.7,delta_t=6,f1=0.76   2022/2/23
    # Fpr=[]
    # Tpr=[]
    # max_auc = float('0')
    # best_params ={}
    # # shot_predict, precision, recall, f1,fpr,tpr,pr_time = judge_figure(df_testvt,shotnum,test_result ,0.897, 10,0.4)
    # for a1 in [0.5,0.6,0.7,0.8,0.9]:
    #     for delta_t in [1,2,3,4,5,6]:
    #         predict_result = assess1(validset_b, df_validation, a1, delta_t, dsp)
    #         prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_result[:, 1], average='binary')
    #         if prf_r[2] >= max_auc:
    #             max_auc = prf_r[2]
    #             best_params['a1'] = a1
    #             best_params['delta_t'] = delta_t
    #         tn, fp, fn, tp = metrics.confusion_matrix(validset_b[:, 1],  predict_result[:, 1]).ravel()
    #         tpr = tp / (tp + fn)
    #         fpr = fp / (fp + tn)
    #         Fpr.append(fpr)
    #         Tpr.append(tpr)
    # TPFP = np.array([Tpr, Fpr])
    # TPFP = TPFP[np.argsort(TPFP[:,0])]

    # 3.roc
    level = np.linspace(0, 1, 50)
    level = np.sort(np.append(level, [0.98, 0.981, 0.982, 0.995, 0.996, 0.997, 0.998, 0.999]))
    max_auc = float('0')
    best_params = {}
    Fpr = []
    Tpr = []
    # for delta_t in [1,2,3,4,5,6]:    # 寻找最优的delta_t
    #     Fpr = []
    #     Tpr = []
    #     for a1 in level:
    #         predict_result = assess1(validset_b, df_validation, a1, delta_t, dsp)
    #         tn, fp, fn, tp = metrics.confusion_matrix(validset_b[:, 1], predict_result[:, 1]).ravel()
    #         tpr = tp / (tp + fn)
    #         fpr = fp / (fp + tn)
    #         Fpr.append(fpr)
    #         Tpr.append(tpr)
    #     auc = metrics.auc(Fpr, Tpr)
    #     if auc >= max_auc:
    #         max_auc = auc
    #         best_params['delta_t'] = delta_t

    for a1 in level:
        predict_result = assess1(validset_b, df_validation, a1, 1, dsp)
        tn, fp, fn, tp = metrics.confusion_matrix(validset_b[:, 1], predict_result[:, 1]).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        Fpr.append(fpr)
        Tpr.append(tpr)
    roc_data = [Fpr, Tpr]
    # np.save(r'LHdataset\L_H_mix_roc.npy', roc_data)

    fig = plt.figure()
    import matplotlib.font_manager as fm

    # 微软雅黑,如果需要宋体,可以用simsun.ttc
    myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
    fronts = {'family': 'Times New Roman', 'size': 12}
    plt.plot(Fpr, Tpr, label='LmixH_H')
    # plt.plot(roc_data2a7[:,1], roc_data2a7[:,0], label='L_H', linestyle='-.')
    plt.xlabel('Fpr', fontproperties=myfont)
    plt.ylabel('Tpr', fontproperties=myfont)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xticks(fontproperties='Times New Roman', fontsize=12)
    plt.yticks(fontproperties='Times New Roman', fontsize=12)
