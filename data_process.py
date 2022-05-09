# @Time : 2022/2/17 17:06 
# @Author : zhongyu 
# @File : data_process.py


import numpy as np
import h5py
import hdf5Reader2A as h5r
from scipy import signal
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import math
import file_read as fr


def shot_data(shot, disr_tag, channels_down, channels, endtime, col_num):
    # 每一炮的开始和结束时间
    start = h5r.get_attrs("StartTime", shot_number=shot, channel="EFIT_LI")
    end = endtime / 1000

    # 计算Δip
    time1, data_ip = fr.read_data(shot, "IP", start, end)
    time2, data_tip = fr.read_data(shot, "IP_TARGET", start, end)
    delta_ip = (data_ip - data_tip) / data_ip
    data_matrix = np.zeros((len(delta_ip), col_num))
    data_matrix[:, 0] = delta_ip

    # 计算βN
    time3, data_betaT = fr.read_data(shot, "EFIT_BETA_T", start, end)
    time4, data_bt = fr.read_data(shot, "BT", start, end)
    time5, data_r = fr.read_data(shot, "EFIT_MINOR_R", start, end)
    betaN = 1000 * data_betaT * 0.0622 * data_r * data_bt / data_ip
    data_matrix[:, 1] = betaN

    # 降采样：对需要降采样的信号
    i = 2
    for channel in channels_down:
        time, data = fr.read_data(shot, channel, start, end)
        data1 = signal.resample(data, num=len(delta_ip), t=time, axis=0)
        data_matrix[:, i] = data1[0]
        i += 1

    # 不需要降采样的信号直接读取
    for channel in channels:
        time, data = fr.read_data(shot, channel, start, end)
        data_matrix[:, i] = data
        i += 1

    # 是否破裂标签
    if disr_tag == 1:
        data_matrix[-100:, i] = np.ones([1, 100])

    data_matrix[:, i + 1] = np.ones([1, len(delta_ip)]) * shot  # 炮号
    data_matrix[:, i + 2] = time  # 时间
    data_matrix[:, i + 3] = np.ones([1, len(delta_ip)]) * endtime  # 结束时间
    return data_matrix


def set_build(shotset, channels_down, channels, errorshot, col_num):
    train_data_dis = np.empty([0, col_num])  # 空训练集
    for i in range(shotset.shape[0]):  # 破裂炮
        shot = shotset[i, 0]
        endtime = shotset[i, 2]
        try:
            matrix1 = shot_data(shot, shotset[i, 1], channels_down, channels, endtime, col_num)
            train_data_dis = np.append(train_data_dis, matrix1, axis=0)
        except Exception as err:
            print(err)
            print('errorshot:{}'.format(shot))
            errorshot.append(shot)
    return train_data_dis, errorshot


if __name__ == '__main__':
    # train_dis_shot = np.load(r'dataset\train_dis_shot.npy')
    # test_dis_shot = np.load(r'dataset\test_dis_shot.npy')
    # train_undis_shot = np.load(r'dataset\train_undis_shot.npy')
    # test_undis_shot = np.load(r'dataset\test_undis_shot.npy')
    H_beta_shot = np.load(r'LHdataset\H_beta.npy')
    L_beta_train_shot = np.load(r'LHdataset\L_beta_train.npy')
    L_beta_val_shot = np.load(r'LHdataset\L_beta_val.npy')
    L_beta_test_shot = np.load(r'LHdataset\L_beta_test.npy')
    H_beta_train_shot = np.load(r'LHdataset\H_beta_train.npy')
    H_beta_val_shot = np.load(r'LHdataset\H_beta_val.npy')
    H_beta_test_shot = np.load(r'LHdataset\H_beta_test.npy')

    channels = ["EFIT_BETA_T", "EFIT_BETA_P", "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
                "BT", "DENSITY", "W_E", "FIR01", "FIR03"]  # 不需要降采样信号
    channels_down = ["I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03", "BOLU06", "SX03", "SX06"]  # 需要降采样的信号

    disr_tag = 1
    errorshot = []

    # matrix1 = shot_data(36197,disr_tag,channels_down,channels,217,25)  # 调试函数

    # 训练集
    print('训练集数据处理...')
    col_num = 25  # 包括了特征数、破裂标签、炮号、时间与结束时间的列数
    # train_data_dis, errorshot = set_build(train_dis_shot, channels_down, channels, errorshot, col_num)  # 破裂
    # train_data_undis, errorshot = set_build(train_undis_shot, channels_down, channels, errorshot, col_num)  # 非破裂
    train_data, errorshot = set_build(H_beta_train_shot, channels_down, channels, errorshot, col_num)  # 非破裂
    # train_data_dis = np.empty([0, col_num])  # 空训练集
    # for i in range(train_dis_shot.shape[0]):  # 破裂炮
    #     shot = train_dis_shot[i, 0]
    #     endtime = train_dis_shot[i, 2]
    #     try:
    #         matrix1 = shot_data(shot, disr_tag, channels_down, channels, endtime, col_num)
    #         train_data_dis = np.append(train_data_dis, matrix1, axis=0)
    #     except Exception as err:
    #         print(err)
    #         print('errorshot:{}'.format(shot))
    #         errorshot.append(shot)

    # d = np.nonzero(train_data_dis[:, 17] == 1)  # 只取了破裂前100ms的数据
    # data_d = train_data_dis[d]
    #
    # train_data_undis = np.empty([0, 20])  # 空训练集
    # for i in range(train_undis_shot.shape[0]):  # 非破裂炮
    #     shot = train_undis_shot[i, 0]
    #     endtime = train_undis_shot[i, 2]
    #     try:
    #         matrix1 = shot_data(shot, 0, channels_down, channels, endtime)
    #         train_data_undis = np.append(train_data_undis, matrix1, axis=0)
    #     except Exception as err:
    #         print(err)
    #         print('errorshot:{}'.format(shot))
    #         errorshot.append(shot)

    # toptrain_data = np.append(train_data_dis, train_data_undis, axis=0)
    # print('train set:' + str(toptrain_data.shape))
    topdata_train = pd.DataFrame(train_data, columns=['Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",
                                                         "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
                                                         "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
                                                         "BT", "DENSITY", "W_E", "FIR01", "FIR03",
                                                         'disrup_tag', '#', 'time', 'endtime'])  # 训练集

    # 验证集
    print('验证集数据处理...')
    val_data, errorshot = set_build(H_beta_val_shot, channels_down, channels, errorshot, col_num)
    topdata_val = pd.DataFrame(val_data, columns=['Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",
                                                  "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
                                                  "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
                                                  "BT", "DENSITY", "W_E", "FIR01", "FIR03",
                                                  'disrup_tag', '#', 'time', 'endtime'])  # 验证集

    print('测试集数据处理...')
    col_num = 25  # 包括了特征数、破裂标签、炮号、时间与结束时间的列数
    # test_data_dis, errorshot = set_build(test_dis_shot, channels_down, channels, errorshot, col_num)  # 破裂
    # test_data_undis, errorshot = set_build(test_undis_shot, channels_down, channels, errorshot, col_num)  # 非破裂
    test_data, errorshot = set_build(H_beta_test_shot, channels_down, channels, errorshot, col_num)  # 非破裂
    # train_data_dis = np.empty([0, col_num])  # 空训练集
    # for i in range(train_dis_shot.shape[0]):  # 破裂炮
    #     shot = train_dis_shot[i, 0]
    #     endtime = train_dis_shot[i, 2]
    #     try:
    #         matrix1 = shot_data(shot, disr_tag, channels_down, channels, endtime, col_num)
    #         train_data_dis = np.append(train_data_dis, matrix1, axis=0)
    #     except Exception as err:
    #         print(err)
    #         print('errorshot:{}'.format(shot))
    #         errorshot.append(shot)

    # d = np.nonzero(train_data_dis[:, 17] == 1)  # 只取了破裂前100ms的数据
    # data_d = train_data_dis[d]
    #
    # train_data_undis = np.empty([0, 20])  # 空训练集
    # for i in range(train_undis_shot.shape[0]):  # 非破裂炮
    #     shot = train_undis_shot[i, 0]
    #     endtime = train_undis_shot[i, 2]
    #     try:
    #         matrix1 = shot_data(shot, 0, channels_down, channels, endtime)
    #         train_data_undis = np.append(train_data_undis, matrix1, axis=0)
    #     except Exception as err:
    #         print(err)
    #         print('errorshot:{}'.format(shot))
    #         errorshot.append(shot)

    # toptest_data = np.append(test_data_dis, test_data_undis, axis=0)
    # print('test set:' + str(toptest_data.shape))
    topdata_test = pd.DataFrame(test_data, columns=['Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",
                                                       "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
                                                       "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
                                                       "BT", "DENSITY", "W_E", "FIR01", "FIR03",
                                                       'disrup_tag', '#', 'time', 'endtime'])  # 训练集
    # # 高β集
    # print('高β集数据处理...')
    # H_beta_data, errorshot = set_build(H_beta_shot, channels_down, channels, errorshot, col_num)
    # topdata_H_beta = pd.DataFrame(H_beta_data, columns=['Δip', 'βN', "I_HA_N", "V_LOOP", "BOLD03", "BOLD06", "BOLU03",
    #                                                  "BOLU06", "SX03", "SX06", "EFIT_BETA_T", "EFIT_BETA_P",
    #                                                  "EFIT_ELONGATION", "EFIT_LI", "EFIT_Q0", "EFIT_QBDRY",
    #                                                  "BT", "DENSITY", "W_E", "FIR01", "FIR03",
    #                                                  'disrup_tag', '#', 'time', 'endtime'])  # 验证集

    # scaler = StandardScaler().fit(toptrain_data[:, :16])  # 归一化
    # normalized_data = scaler.transform(toptrain_data[:, :16])
    # toptrain_data[:, :16] = normalized_data


    # topdata_train.to_csv('dataset/topdata_train.csv')
    # topdata_test.to_csv('dataset/topdata_test.csv')

    # topdata_test.to_csv('LHdataset/topdata_test.csv')
    # topdata_train.to_csv('LHdataset/topdata_train.csv')
    # topdata_val.to_csv('LHdataset/topdata_val.csv')
    # topdata_H_beta.to_csv('LHdataset/topdata_H_beta.csv')
    #
    # topdata_test.to_csv('LHdataset/topdata_H_test.csv')
    # topdata_train.to_csv('LHdataset/topdata_H_train.csv')
    # topdata_val.to_csv('LHdataset/topdata_H_val.csv')
