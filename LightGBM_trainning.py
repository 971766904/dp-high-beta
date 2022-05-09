# @Time : 2022/2/18 15:39 
# @Author : zhongyu 
# @File : LightGBM_trainning.py


import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    print('Loading data...')
    # load or create your dataset
    df_train = pd.read_csv('dataset/topdata_train.csv', index_col=0)

    # # 平衡数据集
    # undis = df_train[df_train['disrup_tag']==0]
    # disrup = df_train[df_train['disrup_tag']==1]
    # disapp = disrup
    # for i in range(18):
    #     disapp = pd.concat([disapp,disrup])
    # normaldata = undis.sample(frac = 0.5,replace= False,random_state=None,axis=0)
    # data_appen = pd.concat([normaldata,disapp])
    # data_appen = data_appen.sample(frac = 1.0,replace= False,random_state=None,axis=0)

    # 从训练集中分割出训练集与验证集
    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    train_data, val_data, train_y, val_y = \
        train_test_split(X_train, y_train, test_size=0.2, random_state=1, shuffle=True, stratify=y_train)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(train_data, train_y, feature_name=['delta_ip', 'beta_N', "HA", "V_LOOP", "BOLD03", "BOLD06",
                                                               "BOLU03", "BOLU06", "SX03", "SX06", "EFIT_BETA_T",
                                                               "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0",
                                                               "EFIT_QBDRY", "BT", "DENSITY", "W_E", "FIR01", "FIR03"])
    lgb_eval = lgb.Dataset(val_data, val_y, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 14,
        'num_leaves': 340,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 40,
        'verbose': 0,
        'cat_smooth': 17,
        'max_bin': 255,
        'min_data_in_leaf': 1,
        'lambda_l1': 0.5,
        'lambda_l2': 1e-5,
        'is_unbalance': True,
        'min_split_gain': 0.3

    }

    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    evals_result=evals_result,
                    early_stopping_rounds=20)

    # print('Saving model...')
    # # save model to file
    # gbm.save_model('model/model_1.txt')
    #
    # print('Plotting feature importances...')
    # ax = lgb.plot_importance(gbm, max_num_features=10)
    # plt.show()

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(val_data, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(val_y, y_pred) ** 0.5)
    print('the roc is', metrics.roc_auc_score(val_y, y_pred))

    # # shap分析
    # shaptest = df_train.drop(['disrup_tag','#','time'],axis = 1)
    # explainer = shap.TreeExplainer(gbm)
    # shap_values = explainer.shap_values(shaptest)
    #
    # # 各个信号重要性
    # shap.summary_plot(shap_values, shaptest, show= False)
    # f =plt.gcf()
