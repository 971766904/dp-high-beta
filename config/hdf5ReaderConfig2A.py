# -*- coding: utf-8 -*-
# @Time    : 2018/11/9 10:08
# @Author  : yaoh
# @FileName: tokamak_hdf5_dir.py
# @Software: PyCharm Community Edition
# @email   : hyao666@foxmail.com

# "HL-2A":10000-36923
# "J-TEXT":1038394-1059000
# "EAST":10000-80000

device_name = {
    "2A":     "2A",
    "HL2A":   "2A",
    "HL-2A":  "2A"
}

file_dir = {
    "2A":       r'str(r"H:\2a-data\HL-2A-UDA") +str(r"/HL-2A=") + str(shot_number) + str(r"=DB.h5")',
}


dir = {
    "EFIT_BETA_T": 'Data["SCALAR"]',
    "EFIT_BETA_P": 'Data["SCALAR"]',
    "EFIT_ELONGATION": 'Data["SCALAR"]',
    "EFIT_LI": 'Data["SCALAR"]',
    "EFIT_MINOR_R": 'Data["SCALAR"]',
    "EFIT_Q0": 'Data["SCALAR"]',
    "EFIT_QBDRY": 'Data["SCALAR"]',
    "EFIT_R": 'Data["SCALAR"]',
    "EFIT_Z": 'Data["SCALAR"]',
    "IP": 'Data["SCALAR"]',
    "I_HA_N": 'Data["SCALAR"]',
    "BT": 'Data["SCALAR"]',
    "BH": 'Data["SCALAR"]',
    "BOH": 'Data["SCALAR"]',
    "BV": 'Data["SCALAR"]',
    "DV": 'Data["SCALAR"]',
    "DH": 'Data["SCALAR"]',
    "HX1": 'Data["SCALAR"]',
    "HX2": 'Data["SCALAR"]',
    "IP_TARGET": 'Data["SCALAR"]',
    "I_DIV_DA": 'Data["SCALAR"]',
    "MP1": 'Data["SCALAR"]',
    "MP2": 'Data["SCALAR"]',
    "DENSITY": 'Data["SCALAR"]',
    "V_LOOP": 'Data["SCALAR"]',
    "W_E": 'Data["SCALAR"]',
    "GAS_FEEDBACK": 'Data["SCALAR"]',
    "GAS_FEEDFORWARD": 'Data["SCALAR"]',
    "VUV": 'Data["SCALAR"]',
    "MP04": 'Data["MIR_MP_ARRAY"]',
    "MP05": 'Data["MIR_MP_ARRAY"]',
    "MP12": 'Data["MIR_MP_ARRAY"]',
    "MP13": 'Data["MIR_MP_ARRAY"]',
    "NP02": 'Data["MIR_NP_ARRAY"]',
    "NP03": 'Data["MIR_NP_ARRAY"]',
    "NP04": 'Data["MIR_NP_ARRAY"]',
    "NP07": 'Data["MIR_NP_ARRAY"]',
    "NP09": 'Data["MIR_NP_ARRAY"]',
    "NP10": 'Data["MIR_NP_ARRAY"]',
    "BOLD01": 'Data["BOLD_ARRAY"]',
    "BOLD02": 'Data["BOLD_ARRAY"]',
    "BOLD03": 'Data["BOLD_ARRAY"]',
    "BOLD04": 'Data["BOLD_ARRAY"]',
    "BOLD05": 'Data["BOLD_ARRAY"]',
    "BOLD06": 'Data["BOLD_ARRAY"]',
    "BOLD07": 'Data["BOLD_ARRAY"]',
    "BOLD08": 'Data["BOLD_ARRAY"]',
    "BOLD09": 'Data["BOLD_ARRAY"]',
    "BOLD10": 'Data["BOLD_ARRAY"]',
    "BOLD11": 'Data["BOLD_ARRAY"]',
    "BOLD12": 'Data["BOLD_ARRAY"]',
    "BOLD13": 'Data["BOLD_ARRAY"]',
    "BOLD14": 'Data["BOLD_ARRAY"]',
    "BOLD15": 'Data["BOLD_ARRAY"]',
    "BOLD16": 'Data["BOLD_ARRAY"]',
    "BOLU01": 'Data["BOLU_ARRAY"]',
    "BOLU02": 'Data["BOLU_ARRAY"]',
    "BOLU03": 'Data["BOLU_ARRAY"]',
    "BOLU04": 'Data["BOLU_ARRAY"]',
    "BOLU05": 'Data["BOLU_ARRAY"]',
    "BOLU06": 'Data["BOLU_ARRAY"]',
    "BOLU07": 'Data["BOLU_ARRAY"]',
    "BOLU08": 'Data["BOLU_ARRAY"]',
    "BOLU09": 'Data["BOLU_ARRAY"]',
    "BOLU10": 'Data["BOLU_ARRAY"]',
    "BOLU11": 'Data["BOLU_ARRAY"]',
    "BOLU12": 'Data["BOLU_ARRAY"]',
    "BOLU13": 'Data["BOLU_ARRAY"]',
    "BOLU14": 'Data["BOLU_ARRAY"]',
    "BOLU15": 'Data["BOLU_ARRAY"]',
    "BOLU16": 'Data["BOLU_ARRAY"]',
    "FIR01": 'Data["FIR_ARRAY"]',
    "FIR02": 'Data["FIR_ARRAY"]',
    "FIR03": 'Data["FIR_ARRAY"]',
    "FIR04": 'Data["FIR_ARRAY"]',
    "SX01": 'Data["SX_ARRAY"]',
    "SX02": 'Data["SX_ARRAY"]',
    "SX03": 'Data["SX_ARRAY"]',
    "SX04": 'Data["SX_ARRAY"]',
    "SX05": 'Data["SX_ARRAY"]',
    "SX06": 'Data["SX_ARRAY"]',
    "SX07": 'Data["SX_ARRAY"]',
    "SX08": 'Data["SX_ARRAY"]',
    "SX10": 'Data["SX_ARRAY"]',
    "SX11": 'Data["SX_ARRAY"]',
    "SX12": 'Data["SX_ARRAY"]',
    "SX13": 'Data["SX_ARRAY"]',
    "SX14": 'Data["SX_ARRAY"]',
    "SX15": 'Data["SX_ARRAY"]',
    "SX16": 'Data["SX_ARRAY"]',
    "SX17": 'Data["SX_ARRAY"]',
    "SX18": 'Data["SX_ARRAY"]',
    "SX19": 'Data["SX_ARRAY"]',
    "SX20": 'Data["SX_ARRAY"]'

}
