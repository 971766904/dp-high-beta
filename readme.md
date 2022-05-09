# HL-2A高β破裂

## 1.shot-choose
   - EFIT数据不一定完整（在结束时间点前就没有了）
   - disruptions1.xlsx是破裂或结束时间点，第一列炮号，第二列是否破裂，第三列破裂或结束时间点
   - 训练集测试集划分```np.random.shuffle(dis_shot)```
   - 需要增加验证集
## 2. data_process
   - 部分炮时间不足100ms
   - 使用信号：'Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",                                                       
                "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
                "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
                "BT", "DENSITY", "W_E", "FIR01", "FIR03"
   - 保存数据文件名：topdata_train.csv， topdata_test.csv
## 3. LightGBM_trainning
   - 利用sklearn.model_selection.train_test_split将训练集样本点分为训练和验证两部分，
   - 由于破裂样本点占比0.02211840462832425，模型可能过拟合了，可以对样本进行平衡
   - 保存模型名：model_1.txt
## 4. test_analysis_shot
   - 测试集评估，roc曲线
## 5. params_tuning & tuning_shot
   - 超参数调节
## 6. roc_compare
  - 比较不同模型的roc曲线
## 7. shap_analysis
  - 分析特征对破裂贡献度，包括相关性分析
## 8. shot_SHAP
  - 分析特征对每一炮的破裂贡献度
## 9.incremental_training
  - 用于增量学习训练探索
## 10.shot_signal_analysis
  - 用于对比每一炮的shap值变化，分析引起变化的输入特征

## 训练思路
- 利用低β数据训练一个高效模型用高β数据来测试
- 利用低β数据加如少量高β数据来训练，用高β数据进行测试，对比两个模型性能
- 利用SHAP分析两个模型效果
- 模型L_H的bayes超参数调节：Final result: {'target': 0.9698326022737446, 'params': {'bagging_fraction': 0.7162422901102445,
'feature_fraction': 0.84974664157569, 'lambda_l1': 0.43612384651036784, 'lambda_l2': 0.9290120150434045, 'max_depth': 18.284715365630042, 'min_data_in_leaf': 103.20238402239872, 'num_leaves': 248.6658584137085}}
- 模型L_H_mix的bayes超参数调节：Final result: {'target': 0.9745799565131449, 'params': {'bagging_fraction': 0.7380101641257047, 'feature_fraction': 0.8645917930709374, 'lambda_l1': 0.030588791125829573, 'lambda_l2': 2.778394633628051, 'max_depth': 19.978169661804174, 'min_data_in_leaf': 165.14670031784632, 'num_leaves': 347.1316499103476}}
- 模型H的超参数调节：`'max_depth': 5,'num_leaves': 10,`
- 模型L_H的超参数调节：`'max_depth': 5,'num_leaves': 20,`
- 数据集LHdatabase
  - H_beta.npy包含所有高β炮号（βN>1)
  - L_beta_train.npy、L_beta_val.npy、L_beta_test.npy包含所有低β
  - H_beta_mix.npy用于混合的10炮高β炮
  - H_beta1.npy删除了10炮用于混合后的高β
  - L_H_mix_roc是L_H数据集roc数据
- 混合数据当前使用模型：modelL_H_mix9_100.txt
- 低β数据当前使用模型：model_L_5_20.txt

