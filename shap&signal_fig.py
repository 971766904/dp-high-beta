# @Time : 2022/5/16 16:57 
# @Author : zhongyu 
# @File : shap&signal_fig.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hdf5Reader2A as h5r
import file_read as fr
import matplotlib.font_manager as fm
import lightgbm as lgb
import shap
import test_analysis_shot as tas
import pandas as pd


class Plot2d:
    def __init__(self, array_t, shot):
        self.array_t = array_t
        self.shot = shot

    def array_plot(self):
        time, data = h5r.read_channel(self.shot, channel=self.array_t[0], device="2a")
        local = np.arange(len(self.array_t)) + 1
        array_1 = np.empty([len(self.array_t), data.shape[0]])
        for i in range(len(self.array_t)):
            time, data = h5r.read_channel(self.shot, channel=self.array_t[i], device="2a")
            array_1[i, :] = data
        plt.figure()
        plt.contourf(time, local, array_1)
        plt.colorbar()
        plt.xlabel('time(s)')


class Plotsignal:
    def __init__(self, tag_list, shot):
        self.tag_list = tag_list
        self.shot = shot

    def signal_plot(self):
        va_shot = 36823
        fig, axes = plt.subplots(nrows=int(len(self.tag_list) / 2), ncols=1, sharex=True)
        fig.suptitle('#{}'.format(self.shot))
        # 微软雅黑,如果需要宋体,可以用simsun.ttc
        myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
        fronts = {'family': 'Times New Roman', 'size': 12}
        for i in range(0, len(self.tag_list), 2):
            time, data = h5r.read_channel(self.shot, channel=self.tag_list[i], device="2a")
            time_t, data_t = h5r.read_channel(self.shot, channel=self.tag_list[i + 1], device="2a")
            axes1 = axes[int(i / 2)]

            lns1 = axes1.plot(time, data, 'r', label=self.tag_list[i])
            axes1.set_ylabel(self.tag_list[i], fontproperties=myfont)
            # axes1.set_yticks(fontproperties='Times New Roman', fontsize=12)
            axes2 = axes1.twinx()
            lns2 = axes2.plot(time_t, data_t, 'b', label=self.tag_list[i + 1])
            axes2.set_ylabel(self.tag_list[i + 1], fontproperties=myfont)
            # 合并图例
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            axes2.legend(lns, labs)
        axes1.set_xlabel('time(s)', fontproperties=myfont)


if __name__ == '__main__':
    shot_for_shap = [36755, 36823, 37021]
    # shot_for_shap = [36755]
    feature_tags = ['IP', "DENSITY", "W_E", "V_LOOP", "FIR01", "FIR03", "DH", "DV", "BOLU03", "BOLU06"]
    sxr_array_t = ['SX01', 'SX02', 'SX03', 'SX04', 'SX05', 'SX06', 'SX07', 'SX08', 'SX09', 'SX10', 'SX11', 'SX12',
                   'SX13', 'SX14', 'SX15', 'SX16', 'SX17', 'SX18', 'SX19', 'SX20']
    bol_array_t = ["BOLU03", "BOLU04", "BOLU05", "BOLU06", "BOLU07", "BOLU08", "BOLU09", "BOLU10",
                   "BOLU11", "BOLU12", "BOLU13", "BOLU14", "BOLU15", "BOLU16"]
    # plot2d_1 = Plot2d(sxr_array_t, 36823)
    # plot2d_1.array_plot()

    plotfea = Plotsignal(feature_tags, 36823)
    plotfea.signal_plot()
