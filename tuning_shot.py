# @Time : 2022/2/19 16:37 
# @Author : zhongyu 
# @File : tuning_shot.py
import lightgbm as lgb
import pandas as pd
import numpy as np
import test_analysis_shot as tas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def train_assess(params, lgb_train, lgb_eval, validset_b, df_validation, a1, delta_t):
    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=[lgb_train, lgb_eval],
                    evals_result=evals_result,
                    verbose_eval=20,
                    early_stopping_rounds=30)
    predict_result = tas.assess1(validset_b, df_validation, a1, delta_t, gbm)
    return predict_result


def tuning_ac(params, lgb_train, lgb_eval, validset_b, df_val):
    max_auc = float('0')
    best_params = {}

    # 准确率
    print("调参1：提高准确率")
    for max_depth in [3, 4, 5, 6]:
        for num_leaves in [50, 55, 40, 45, 30, 35, 60, 65, 70]:
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=1,
                nfold=5,
                metrics=['auc'],
                early_stopping_rounds=10,
                verbose_eval=50
            )

            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    print('max auc',max_auc,'best params',best_params)

    return params


if __name__ == '__main__':
    print('Loading data...')
    # # load or create your dataset
    # df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    # df_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    # # testset_b = np.load(r'dataset\train_dis_shot.npy')
    # # testset_a = np.load(r'dataset\train_undis_shot.npy')
    # validset_b = np.load(r'LHdataset\L_beta_val.npy')
    # # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    # # validset_b = np.append(testset_a, testset_b, axis=0)

    # 混合L和H数据集
    df_train = pd.read_csv('LHdataset/topdata_train.csv', index_col=0)
    weight_1 = np.ones([df_train.shape[0]])
    df_H_beta_mix = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    weight_2 = np.ones([df_H_beta_mix.shape[0]]) * 1.5
    w_train = list(np.append(weight_1, weight_2, axis=0))
    df_L_val = pd.read_csv('LHdataset/topdata_val.csv', index_col=0)
    df_H_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    validset_b = np.load(r'LHdataset\L_beta_val.npy')
    validset_a = np.load(r'LHdataset\H_beta_val.npy')
    # df_validation = pd.read_csv('dataset/topdata_train.csv', index_col=0)
    validset_b = np.append(validset_b, validset_a, axis=0)
    newdf_H = pd.DataFrame(np.repeat(df_H_beta_mix.values, 6, axis=0))  # 加倍10炮数据，调节model-mix效果
    newdf_H.columns = df_H_beta_mix.columns
    # df_train = df_train.append(newdf_H, ignore_index=True)
    df_train = df_train.append(df_H_beta_mix, ignore_index=True)
    df_val = df_L_val.append(df_H_val, ignore_index=True)

    # # 10炮训练集
    # df_train = pd.read_csv('LHdataset/topdata_H_beta_mix.csv', index_col=0)
    # df_val = pd.read_csv('LHdataset/topdata_H_val.csv', index_col=0)
    # validset_b = np.load(r'LHdataset\H_beta_val.npy')

    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    train_data, val_data, train_y, val_y = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train, weight=w_train,
                            feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                                          "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                                          "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                          "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 准确率
    print("调参1：提高准确率")
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'is_unbalance': True
    }
    # 最佳迭代次数
    # cv_results = lgb.cv(params, lgb_train, num_boost_round=5000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    #                     early_stopping_rounds=50, seed=0)
    # print('best n_estimators:', len(cv_results['auc-mean']))
    #
    # print('best cv score:', pd.Series(cv_results['auc-mean']).max())

    max_auc = float('0')
    best_params = {}
    a1 = 0.8
    delta_t = 90

    # 准确率
    print("调参1：提高准确率")
    for max_depth in [3, 4, 5, 6]:
        for num_leaves in [50, 55, 65, 60, 25, 30, 45]:
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_val, a1, delta_t)
            prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
            mean_auc = prf_r[1]

            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']

    # 过拟合
    # print("调参2：降低过拟合")
    # for max_bin in [64,128,256,512]:
    #     for min_data_in_leaf in [18,19,20,21,22]:
    #         params['max_bin'] = max_bin
    #         params['min_data_in_leaf'] = min_data_in_leaf
    #
    #         predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #         prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #         mean_auc = prf_r[1]
    #
    #         if mean_auc >= max_auc:
    #             max_auc = mean_auc
    #             best_params['max_bin'] = max_bin
    #             best_params['min_data_in_leaf'] = min_data_in_leaf
    # if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
    #     params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    #     params['max_bin'] = best_params['max_bin']
    #
    # print("调参3：降低过拟合")
    # for feature_fraction in [0.6, 0.8, 1]:
    #     for bagging_fraction in [0.8,0.9,1]:
    #         for bagging_freq in [2,3,4]:
    #             params['feature_fraction'] = feature_fraction
    #             params['bagging_fraction'] = bagging_fraction
    #             params['bagging_freq'] = bagging_freq
    #
    #             predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #             prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #             mean_auc = prf_r[1]
    #
    #             if mean_auc >= max_auc:
    #                 max_auc = mean_auc
    #                 best_params['feature_fraction'] = feature_fraction
    #                 best_params['bagging_fraction'] = bagging_fraction
    #                 best_params['bagging_freq'] = bagging_freq
    #
    # if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
    #     params['feature_fraction'] = best_params['feature_fraction']
    #     params['bagging_fraction'] = best_params['bagging_fraction']
    #     params['bagging_freq'] = best_params['bagging_freq']
    #
    # print("调参4：降低过拟合")
    # for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
    #     for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
    #         params['lambda_l1'] = lambda_l1
    #         params['lambda_l2'] = lambda_l2
    #
    #         predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #         prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #         mean_auc = prf_r[1]
    #
    #         if mean_auc >= max_auc:
    #             max_auc = mean_auc
    #             best_params['lambda_l1'] = lambda_l1
    #             best_params['lambda_l2'] = lambda_l2
    # if 'lambda_l1' and 'lambda_l2' in best_params.keys():
    #     params['lambda_l1'] = best_params['lambda_l1']
    #     params['lambda_l2'] = best_params['lambda_l2']
    #
    # print("调参5：降低过拟合2")
    # for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    #     params['min_split_gain'] = min_split_gain
    #     predict_train = train_assess(params, lgb_train, lgb_eval, validset_b, df_train, a1, delta_t)
    #     prf_r = metrics.precision_recall_fscore_support(validset_b[:, 1], predict_train[:, 1], average='binary')
    #     mean_auc = prf_r[1]
    #
    #     if mean_auc >= max_auc:
    #         max_auc = mean_auc
    #
    #         best_params['min_split_gain'] = min_split_gain
    # if 'min_split_gain' in best_params.keys():
    #     params['min_split_gain'] = best_params['min_split_gain']

    print(best_params)
    print(params)
