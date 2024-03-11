import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from numpy.core.fromnumeric import mean
import re
np.set_printoptions(threshold=sys.maxsize)
from pathlib import Path
from config import *
from multicast_method import *
import shutil

act_tag = 1001
wgt_tag = 1002
out_tag_m2c = 1003
out_tag_c2m = 1004

debug = 0


#### a NoP+NoC example #########
# 硬件信息
# memory_param = {"OL1":,"OL2":,"AL1":,"AL2":,"WL1":,"WL2":}
# 卷积配置 从basicParam_noc_nop中import进来

def calPSumAllReduce(output_num, chiplet_num, PC3 ):
    output_flit_num = int(output_num / neu_per_flit_psum_nop)
    delay = (output_flit_num / chiplet_num) * 2 * (PC3-1)
    d2d_energy = (output_num * psum_width / chiplet_num) * 2 * (PC3-1) * chiplet_num * DIE2DIE_energy_ratio
    dram_energy = (output_num * psum_width / chiplet_num) * PC3 * 2 * chiplet_num * DRAM_energy_ratio
    energy_list = [d2d_energy, dram_energy, d2d_energy+dram_energy]
    return delay, energy_list

def calFitness(for_list, comm_patterns, parallel_dim_list, partition_list, network_param, HW_param, memory_param, NoC_param, if_multicast, flag = "ours"):
    route_table = NoC_param["route_table"]
    bw_scales = NoC_param["bw_scales"]
    F = NoC_param["F"]
    link_energy_ratio = NoC_param["energy_ratio"]
    link_energy = F.copy()
   
    #-------------------------------------------------#
    #----------------- Get HW Parm -------------------#
    #-------------------------------------------------#
    CoreNum = HW_param["PE"]['x'] * HW_param["PE"]['y']
    ChipNum = HW_param["Chiplet"]['x'] * HW_param["Chiplet"]['y']
    OL1 = memory_param["OL1"]
    OL2 = memory_param["OL2"]
    AL1 = memory_param["AL1"]
    AL2 = memory_param["AL2"]
    WL1 = memory_param["WL1"]
    WL2 = memory_param["WL2"]

    #-------------------------------------------------#
    #------------------ Get Dataflow -----------------#
    #-------------------------------------------------#
    data_flow = for_list[0]
    ol1_ratio = for_list[1]
    al1_ratio = for_list[2]
    wl1_ratio = for_list[3]
    all_param = for_list[4]
    out_final = for_list[5]

    '''
    if_act_share_PE = for_list[6]
    if_wgt_share_PE = for_list[7]
    if_act_share_Chiplet = for_list[8]
    if_wgt_share_Chiplet = for_list[9]
    '''
    #-------------------------------------------------#
    #------------------- Parameter -------------------#
    #-------------------------------------------------#
    # mapping parameter
    P1,P2,P3 = partition_list["P"][0],partition_list["P"][1],partition_list["P"][2]
    Q1,Q2,Q3 = partition_list["Q"][0],partition_list["Q"][1],partition_list["Q"][2]
    C1,C2,C3 = partition_list["C"][0],partition_list["C"][1],partition_list["C"][2]
    K1,K2,K3 = partition_list["K"][0],partition_list["K"][1],partition_list["K"][2]
    PP2,PQ2,PC2,PK2 = parallel_dim_list[0][0],parallel_dim_list[0][1],parallel_dim_list[0][2],parallel_dim_list[0][3]
    PP3,PQ3,PC3,PK3 = parallel_dim_list[1][0],parallel_dim_list[1][1],parallel_dim_list[1][2],parallel_dim_list[1][3]
    PK0 = HW_param["MAC"]["x"]
    PC0 = HW_param["MAC"]["y"]

    # network parameter
    P = network_param["P"]
    Q = network_param["Q"]
    K = network_param["K"]
    C = network_param["C"]
    R = network_param["R"]
    S = network_param["S"]
    stride = network_param["stride"]

    # runtime param
    runtimeP = PP3*P3*PP2*P2*P1
    runtimeQ = PQ3*Q3*PQ2*Q2*Q1
    runtimeK = PK3*K3*PK2*K2*K1*PK0
    runtimeC = PC3*C3*PC2*C2*C1*PC0
    runtimeR = R # R S不拆分,在PE level的时序for参数里
    runtimeS = S
    runtimeCoreNum = PK2*PQ2*PP2*PC2
    runtimeChipNum = PP3*PQ3*PK3*PC3

    assert(runtimeP>=P);assert(runtimeQ>=Q);assert(runtimeK>=K);assert(runtimeC>=C)
    assert(runtimeCoreNum <= CoreNum);assert(runtimeChipNum <= ChipNum)

    energy_MAC = P*Q*K*C*R*S * MAC_energy_ratio
    compuation_num = runtimeP*runtimeQ*runtimeK*runtimeC*runtimeR*runtimeS
    compuation_cycles = compuation_num/runtimeCoreNum/runtimeChipNum/PC0/PK0
    #print ("compuation_num=",compuation_num)
    #print ("compuation_cycles=",compuation_cycles)
			
    # storage size
    AL1_mem = AL1*8*1024/act_wgt_width/2 # /2是因为ping-pong
    OL1_mem = OL1*8*1024/psum_width/2 
    WL1_mem = WL1*8*1024/act_wgt_width/2
    AL2_mem = AL2*8*1024/act_wgt_width/2
    OL2_mem = OL2*8*1024/psum_width/2
    WL2_mem = WL2*8*1024/act_wgt_width/2 
    A_PE_mem = PC0
    W_PE_mem = PC0*PK0	

    OL1_need = {}; AL1_need = {}; WL1_need = {}; L1_need = {}
    OL2_need = {}; AL2_need = {}; WL2_need = {}; L2_need = {}

    cal_cycles = {}
    if_out_final = {}


    ol1_need = PK0; al1_need_CKpart= PC0; wl1_need = PK0*PC0; cal =1
    al1_need_Qpart = 1; al1_need_Ppart = 1; al1_need_Rpart = 1; al1_need_Spart = 1
    # ------------------ 计算6个buffer存储需求&每级for循环循环次数 ------------------

    for id in range(len(data_flow)):
        param = data_flow[id]
        ol1_need = ol1_need * ol1_ratio[id] # 单位:neuron

        # al1 need calculation
        if "C" == param[0]:
            al1_need_CKpart = al1_need_CKpart * all_param[id]
        elif "Q" == param[0]:
            al1_need_Qpart = al1_need_Qpart * all_param[id]
        elif "P" == param[0]:
            al1_need_Ppart = al1_need_Ppart * all_param[id]
        elif "R" == param[0]:
            al1_need_Rpart = al1_need_Rpart * all_param[id]
        elif "S" == param[0]:
            al1_need_Spart = al1_need_Spart * all_param[id]

        al1_need_Q_final = al1_need_Qpart * stride + al1_need_Spart - stride
        al1_need_P_final = al1_need_Ppart * stride + al1_need_Rpart - stride
        al1_need = al1_need_CKpart * al1_need_Q_final * al1_need_P_final

        
        wl1_need = wl1_need * wl1_ratio[id]

        cal = cal * all_param[id]

        cal_cycles[param] = cal
        OL1_need[param] = ol1_need
        AL1_need[param] = al1_need
        WL1_need[param] = wl1_need
        L1_need[param] = wl1_need + al1_need + ol1_need
        if_out_final[param] = out_final[id]
        # L2
        al2_need_Qpart = al1_need_Qpart * PQ2 
        al2_need_Ppart = al1_need_Ppart * PP2        

        al2_need_Q_final = al2_need_Qpart * stride + al1_need_Spart - stride
        al2_need_P_final = al2_need_Ppart * stride + al1_need_Rpart - stride
        al2_need = al1_need_CKpart * PC2 * al2_need_Q_final * al2_need_P_final
        
        AL2_need[param] = al2_need
        WL2_need[param] = wl1_need * PK2  * PC2
        OL2_need[param] = ol1_need * PK2 * PQ2 * PP2

    repeat = 1
    repeat_num = {}
        
    for id in range(len(data_flow)):
        real_id = len(data_flow) - id -1
        param = data_flow[real_id] 
        repeat = repeat * all_param[real_id]
        repeat_num[param] = repeat

    # ------------------ 决定存储临界点 ------------------

    def find_cp(the_data_flow,storage_need,storage_size):
        for id in range(len(the_data_flow)):
            param = the_data_flow[id]
            if storage_need[param] > storage_size: 
                the_cp = param
                the_cp_id = id
                break
            the_cp = "top"
            the_cp_id = id
        utilization_ratio = storage_need[the_data_flow[the_cp_id-1]] / storage_size
        return the_cp,the_cp_id,utilization_ratio

    ol1_cp,ol1_cp_id,ol1_utilization_ratio = find_cp(data_flow,OL1_need,OL1_mem)
    al1_cp,al1_cp_id,al1_utilization_ratio = find_cp(data_flow,AL1_need,AL1_mem)
    wl1_cp,wl1_cp_id,wl1_utilization_ratio = find_cp(data_flow,WL1_need,WL1_mem)
    ol2_cp,ol2_cp_id,ol2_utilization_ratio = find_cp(data_flow,OL2_need,OL2_mem)
    al2_cp,al2_cp_id,al2_utilization_ratio = find_cp(data_flow,AL2_need,AL2_mem)
    wl2_cp,wl2_cp_id,wl2_utilization_ratio = find_cp(data_flow,WL2_need,WL2_mem)
    ape_cp,ape_cp_id,ape_utilization_ratio = find_cp(data_flow,AL1_need,A_PE_mem)
    wpe_cp,wpe_cp_id,wpe_utilization_ratio = find_cp(data_flow,WL1_need,W_PE_mem)

    if debug == 1:
        print("Debug in find_cp:")
        print("---OL1_mem:{} OL1_need:{}".format(OL1_mem, OL1_need))
        print("---ol1_cp:{} ol1_cp_id:{}".format(ol1_cp, ol1_cp_id))
        print("---AL1_mem:{} AL1_need:{}".format(AL1_mem, AL1_need))
        print("---al1_cp:{} al1_cp_id:{}".format(al1_cp, al1_cp_id))
        print("---WL1_mem:{} WL1_need:{}".format(WL1_mem, WL1_need))
        print("---wl1_cp:{} wl1_cp_id:{}".format(wl1_cp, wl1_cp_id))

    # ------------------ 构建mem cal core 位置和属性等 ------------------
    # 从wxy import进来
    # comm patterns : {tag1:{src_node: {packet_id: [dst_list], xxx}}, tag2}
    act_comms = comm_patterns[act_tag]
    wgt_comms = comm_patterns[wgt_tag]
    out_m2c_comms = comm_patterns[out_tag_m2c]
    out_c2m_comms = comm_patterns[out_tag_c2m]

    # ------------------ 性能预测：计算整层所有计算和通信数据的数目 ------------------
    # REG <-> L1 用于统计通信总量 & prediction
    pe_neu_num_rd_wgt = 0 # 单位 neuron数目 
    pe_neu_num_rd_act = 0

    # --- L1_act
    cur = data_flow[ape_cp_id]; inner = data_flow[ape_cp_id-1]  
    if ape_cp == "top":
        pe_neu_num_rd_act += AL1_need[inner] * 1
    else:
        pe_neu_num_rd_act += AL1_need[inner] * repeat_num[cur]

    # --- L1_wgt
    cur = data_flow[wpe_cp_id]; inner = data_flow[wpe_cp_id-1]  
    pe_neu_num_rd_wgt += WL1_need[inner] * repeat_num[cur]

    pe_neu_num_rd_wgt = pe_neu_num_rd_wgt * CoreNum * ChipNum # 考虑到片上有CoreNum * ChipNum个PE
    pe_neu_num_rd_act = pe_neu_num_rd_act * CoreNum * ChipNum # 考虑到片上有CoreNum * ChipNum个PE
    energy_rd_wgt_L1 = pe_neu_num_rd_wgt * SRAM_energy(WL1) * act_wgt_width 
    energy_rd_act_L1 = pe_neu_num_rd_act * SRAM_energy(AL1) * act_wgt_width  

    # L1 用于统计通信总量 & prediction
    core_pkt_num_wr_opt = 0; core_neu_num_wr_opt = 0  # 单位分别是 packet | neuron数目 
    core_pkt_num_rd_opt = 0; core_neu_num_rd_opt = 0
    core_pkt_num_rd_wgt = 0; core_neu_num_rd_wgt = 0
    core_pkt_num_rd_act = 0; core_neu_num_rd_act = 0

    # L1 用于生成task file的变量
    core_rd_out_data_num = 0
    core_out_data_num = 0 
    core_act_data_num = 0
    core_wgt_data_num = 0
    
    # --- L2->L1 : out
    cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
    if (if_out_final[cur]!=1): 
        #print("CORE: read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
        core_pkt_num_rd_opt += int(math.ceil(OL1_need[inner]*repeat_num[cur]/(flit_per_pkt-1)/neu_per_flit_psum))
        core_neu_num_rd_opt += int(OL1_need[inner] * repeat_num[cur])
        core_rd_out_data_num += OL1_need[inner]
    else:
        core_pkt_num_rd_opt += 0
        core_neu_num_rd_opt += 0
        core_rd_out_data_num += 0
    #print("CORE: write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
    
    # --- L2<-L1 : out_wr
    if (if_out_final[cur]!=1):
        core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]*repeat_num[cur]/(flit_per_pkt-1)/neu_per_flit_psum))
    else:
        core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]*repeat_num[cur]/neu_per_pkt_act_wgt))
    core_out_data_num += OL1_need[inner] # 用于生成仿真指令
    core_neu_num_wr_opt += int(OL1_need[inner] * repeat_num[cur])
    
    # --- L2->L1 : act
    cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
    #print("CORE: read act mem ",AL1_need[inner],"repeat ",repeat_num[cur])
    core_pkt_num_rd_act +=  int(math.ceil(AL1_need[inner]/neu_per_pkt_act_wgt))*repeat_num[cur]
    core_act_data_num += AL1_need[inner] # 用于生成仿真指令
    if al1_cp == "top":
        core_neu_num_rd_act += int(AL1_need[inner] * 1)
    else:
        core_neu_num_rd_act += int(AL1_need[inner] * repeat_num[cur])

    # --- L2->L1 : wgt
    cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
    #print("CORE: read wgt mem ",WL1_need[inner],"repeat ",repeat_num[cur]) 
    core_pkt_num_rd_wgt += int(math.ceil(WL1_need[inner]*repeat_num[cur]/neu_per_pkt_act_wgt))
    core_wgt_data_num += WL1_need[inner] # 用于生成仿真指令
    core_neu_num_rd_wgt += int(WL1_need[inner] * repeat_num[cur])

    # 考虑上并行度带来的数据复用机会 (多播)
    neu_num_wr_ol1 = core_neu_num_rd_opt * CoreNum * ChipNum
    neu_num_wr_al1 = core_neu_num_rd_act * CoreNum * ChipNum
    neu_num_wr_wl1 = core_neu_num_rd_wgt * CoreNum * ChipNum
    neu_num_rd_ol1 = core_neu_num_wr_opt * CoreNum * ChipNum
    if if_multicast == 1:
        core_neu_num_wr_opt = core_neu_num_wr_opt * CoreNum * ChipNum  # 没有机会复用
        core_neu_num_rd_opt = core_neu_num_rd_opt * CoreNum * ChipNum /PC2
        core_neu_num_rd_wgt = core_neu_num_rd_wgt * CoreNum * ChipNum /PP2 / PQ2
        core_neu_num_rd_act = core_neu_num_rd_act * CoreNum * ChipNum /PK2 
    elif if_multicast == 0:
        core_neu_num_wr_opt = core_neu_num_wr_opt * CoreNum * ChipNum  # 没有机会复用
        core_neu_num_rd_opt = core_neu_num_rd_opt * CoreNum * ChipNum 
        core_neu_num_rd_wgt = core_neu_num_rd_wgt * CoreNum * ChipNum  
        core_neu_num_rd_act = core_neu_num_rd_act * CoreNum * ChipNum  

    energy_l2 = SRAM_energy(OL2)
    if flag == "nnbaton":
        energy_l2_w = DRAM_energy_ratio
    else:
        energy_l2_w = energy_l2


    if (if_out_final[data_flow[ol1_cp_id]]!=1):
        energy_wr_opt_L2 = core_neu_num_wr_opt * energy_l2 * psum_width 
        energy_rd_opt_L1 = neu_num_rd_ol1 * SRAM_energy(OL1) * psum_width
    else: 
        energy_wr_opt_L2 = core_neu_num_wr_opt * energy_l2 * act_wgt_width 
        energy_rd_opt_L1 = neu_num_rd_ol1 * SRAM_energy(OL1) * act_wgt_width
    energy_rd_opt_L2 = core_neu_num_rd_opt * energy_l2 * psum_width
    energy_rd_wgt_L2 = core_neu_num_rd_wgt * energy_l2_w * act_wgt_width
    energy_rd_act_L2 = core_neu_num_rd_act * energy_l2 * act_wgt_width

    energy_rd_wgt_L1 += neu_num_wr_wl1 * SRAM_energy(WL1) * act_wgt_width
    energy_rd_act_L1 += neu_num_wr_al1 * SRAM_energy(AL1) * act_wgt_width
    energy_rd_opt_L1 += neu_num_wr_ol1 * SRAM_energy(OL1) * act_wgt_width

    # L2 用于统计通信总量 & prediction
    chip_pkt_num_wr_opt = 0; chip_neu_num_wr_opt = 0
    chip_pkt_num_rd_opt = 0; chip_neu_num_rd_opt = 0
    chip_pkt_num_rd_wgt = 0; chip_neu_num_rd_wgt = 0
    chip_pkt_num_rd_act = 0; chip_neu_num_rd_act = 0

    # L2 用于生成task file的变量
    chip_rd_out_data_num = 0
    chip_out_data_num = 0 
    chip_act_data_num = 0
    chip_wgt_data_num = 0

    # --- DRAM->L2 : out_rd
    cur = data_flow[ol2_cp_id]; inner = data_flow[ol2_cp_id-1]  
    if (if_out_final[cur]!=1): 
        #print("Chip: read opt mem ", OL2_need[inner],"repeat ",repeat_num[cur]) 
        chip_pkt_num_rd_opt += int(math.ceil(OL2_need[inner]*repeat_num[cur]/neu_per_pkt_psum))
        chip_rd_out_data_num += OL2_need[inner]
        chip_neu_num_rd_opt += OL2_need[inner] * repeat_num[cur]
    else:
        chip_pkt_num_rd_opt += 0
        chip_rd_out_data_num += 0
        chip_neu_num_rd_opt += 0
    #print("Chip: write opt mem ", OL2_need[inner],"repeat ",repeat_num[cur])

    # --- DRAM<-L2 : out_wr
	# -- update in 22.7.20 : 一旦片上放得下所有的输出结果，就不再将输出输出到DRAM
    if (if_out_final[cur]!=1): 
        chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]*repeat_num[cur]/neu_per_pkt_psum))
    elif ol2_cp == "top":
        chip_pkt_num_wr_opt += 0
    else:
        chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]*repeat_num[cur]/neu_per_pkt_act_wgt))
    if ol2_cp == "top":
        chip_out_data_num += 0 # 用于生成仿真指令
        chip_neu_num_wr_opt += 0
    else:
        chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
        chip_neu_num_wr_opt += OL2_need[inner] * repeat_num[cur]

    #if (if_out_final[cur]!=1): 
    #    chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
    #else:
    #    chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    #chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
    #chip_neu_num_wr_opt += OL2_need[inner] * repeat_num[cur]
    
    # --- DRAM->L2 : act
    cur = data_flow[al2_cp_id]; inner = data_flow[al2_cp_id-1]  
    #print("Chip: read act mem ",AL2_need[inner],"repeat ",repeat_num[cur])
    chip_pkt_num_rd_act = {"DRAM":0, "chiplet":0}
    if al2_cp == "top":
        chip_pkt_num_rd_act["DRAM"] = int(math.ceil(AL2_need[inner]/neu_per_pkt_act_wgt)) * 1
    else:
        chip_pkt_num_rd_act["DRAM"] = int(math.ceil(AL2_need[inner]*repeat_num[cur]/neu_per_pkt_act_wgt))
        #chip_pkt_num_rd_act += int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * repeat_num[cur]
    
    #chip_pkt_num_rd_act +=  int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
    chip_act_data_num += AL2_need[inner] # 用于生成仿真指令
    chip_neu_num_rd_act = {"DRAM":0, "chiplet":0}
    if al2_cp == "top":
        chip_neu_num_rd_act["DRAM"] = AL2_need[inner] * 1
    else:
        chip_neu_num_rd_act["DRAM"] = AL2_need[inner] * repeat_num[cur]      
        #chip_neu_num_rd_act += AL2_need[inner] * repeat_num[cur]

    # --- DRAM->L2 : wgt
    cur = data_flow[wl2_cp_id]; inner = data_flow[wl2_cp_id-1]  
    #print("Chip: read wgt mem ",WL2_need[inner],"repeat ",repeat_num[cur]) 
    chip_pkt_num_rd_wgt = {"DRAM":0, "chiplet":0}
    if flag == "nnbaton":
        pass
    else:
        chip_pkt_num_rd_wgt["DRAM"] = int(math.ceil(WL2_need[inner]*repeat_num[cur]/neu_per_pkt_act_wgt))
    chip_wgt_data_num += WL2_need[inner] # 用于生成仿真指令
    chip_neu_num_rd_wgt = {"DRAM":0, "chiplet":0}
    if flag == "nnbaton":
        pass
    else:
        chip_neu_num_rd_wgt["DRAM"] = WL2_need[inner] *repeat_num[cur]
    #chip_neu_num_rd_wgt += WL2_need[inner] * repeat_num[cur]

    # 考虑上并行度带来的数据复用机会
    if if_multicast == 1:
        chip_neu_num_wr_opt = chip_neu_num_wr_opt * ChipNum  # 没有机会复用
        chip_neu_num_rd_opt = chip_neu_num_rd_opt * ChipNum 
        if flag == "nnbaton":
            chip_neu_num_rd_wgt = {"DRAM":0, "chiplet":0}
        else:
            chip_neu_num_rd_wgt["DRAM"] = chip_neu_num_rd_wgt["DRAM"] * ChipNum /PP3 / PQ3
        chip_neu_num_rd_act["DRAM"] = chip_neu_num_rd_act["DRAM"] * ChipNum /PK3
    elif if_multicast == 0:
        chip_neu_num_wr_opt = chip_neu_num_wr_opt * ChipNum  # 没有机会复用
        chip_neu_num_rd_opt = chip_neu_num_rd_opt * ChipNum 
        if flag == "nnbaton":
            chip_neu_num_rd_wgt = {"DRAM":0, "chiplet":0}
        else:
            chip_neu_num_rd_wgt["DRAM"] = chip_neu_num_rd_wgt["DRAM"] * ChipNum
        chip_neu_num_rd_act["DRAM"] = chip_neu_num_rd_act["DRAM"] * ChipNum 

    if (if_out_final[cur]!=1): 
        energy_wr_opt_dram = chip_neu_num_wr_opt * DRAM_energy_ratio * psum_width 
    else:
        energy_wr_opt_dram = chip_neu_num_wr_opt * DRAM_energy_ratio * act_wgt_width 
    energy_rd_opt_dram = chip_neu_num_rd_opt * DRAM_energy_ratio * psum_width
    energy_rd_wgt_dram = chip_neu_num_rd_wgt["DRAM"] * DRAM_energy_ratio * act_wgt_width
    energy_rd_act_dram = chip_neu_num_rd_act["DRAM"] * DRAM_energy_ratio * act_wgt_width
    F_cur=F.copy()

    # 对core构建通信需求
    # 用到的信息: core_pkt_num_wr_opt; core_pkt_num_rd_opt; core_pkt_num_rd_wgt; core_pkt_num_rd_act
    bw_needed = (core_pkt_num_rd_act) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle
    for act_mem, comms in act_comms.items():
        for _, dst_list in comms.items():
            if if_multicast == 0:
                for dst in dst_list:
                    for link in route_table[(act_mem + 1000, dst + 1000)]:
                        F_cur[link] += ( bw_needed / bw_scales[link] )
            else:
                link_set = simple_multicast(act_mem + 1000, [dst + 1000 for dst in dst_list], route_table) 
                for link in link_set:
                    F_cur[link] += ( bw_needed / bw_scales[link] ) 

    bw_needed = (core_pkt_num_rd_wgt) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
    for wgt_mem, comms in wgt_comms.items():
        for _, dst_list in comms.items():
            if if_multicast == 0:
                for dst in dst_list:
                    for link in route_table[(wgt_mem + 1000, dst + 1000)]:
                        F_cur[link] += ( bw_needed / bw_scales[link] )
            else:
                link_set = simple_multicast(wgt_mem + 1000, [dst + 1000 for dst in dst_list], route_table) 
                for link in link_set:
                    F_cur[link] += ( bw_needed / bw_scales[link] )

    out_nodes = []
    bw_needed = (core_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
    for out_mem, comms in out_m2c_comms.items():
        for _, dst_list in comms.items():
            if if_multicast == 0:
                for dst in dst_list:
                    for link in route_table[(out_mem + 1000, dst + 1000)]:
                        F_cur[link] += ( bw_needed / bw_scales[link] )
            else:
                link_set = simple_multicast(out_mem + 1000, [dst + 1000 for dst in dst_list], route_table) 
                for link in link_set:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        out_nodes.append(out_mem)

    bw_needed = (core_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
    for c_node, comms in out_c2m_comms.items():
        for _, dst_list in comms.items():
            for dst in dst_list:
                for link in route_table[(c_node + 1000, dst+1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )

    # 对chip构建通信需求
    dram_to_L2_F_cur = L2_to_DRAM_F_cur = 0
    bw_needed_rd_nop = 0
    # 用到的信息: chip_pkt_num_wr_opt; chip_pkt_num_rd_opt; chip_pkt_num_rd_wgt; chip_pkt_num_rd_act
    bw_needed = (chip_pkt_num_rd_act["DRAM"]) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle 
    dram_to_L2_F_cur += bw_needed / (in_nop_bandwidth/noc_bandwidth)
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_input/noc_bandwidth)
    
    bw_needed = (chip_pkt_num_rd_wgt["DRAM"]) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
    dram_to_L2_F_cur += bw_needed / (in_nop_bandwidth/noc_bandwidth)
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_weight/noc_bandwidth)

    bw_needed = (chip_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
    dram_to_L2_F_cur += bw_needed / (out_nop_bandwidth/noc_bandwidth)
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_output/noc_bandwidth)

    bw_needed = (chip_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
    L2_to_DRAM_F_cur = bw_needed / (out_nop_bandwidth/noc_bandwidth)

    nop_F_cur = 0
    for out_node in out_nodes:
        F_cur[(out_node, out_node + 1000)] = 0
        F_cur[(out_node + 1000, out_node)] = 0
    F_cur[(act_mem + 1000, act_mem)] = 0
    F_cur[(wgt_mem + 1000, wgt_mem)] = 0
    degrade_ratio_dict = {"NoC":max(F_cur.values()), "L2_to_DRAM":L2_to_DRAM_F_cur, "DRAM_to_L2":dram_to_L2_F_cur}
    degrade_ratio = max ( max(F_cur.values()), dram_to_L2_F_cur)
    if (degrade_ratio < 1):
            degrade_ratio = 1

    runtime_calNum = runtimeP*runtimeQ*runtimeR*runtimeS*runtimeC*runtimeK
    runtime_list = [runtimeP, runtimeQ, runtimeC, runtimeK, runtimeChipNum, runtimeCoreNum,runtime_calNum]
    cp_list = [ol1_cp_id, al1_cp_id, wl1_cp_id, ol2_cp_id, al2_cp_id, wl2_cp_id]
    utilization_ratio_list = [ol1_utilization_ratio,al1_utilization_ratio,wl1_utilization_ratio, \
                              ol2_utilization_ratio,al2_utilization_ratio,wl2_utilization_ratio]
    energy_L1_list = [energy_rd_wgt_L1, energy_rd_act_L1, energy_rd_opt_L1]
    energy_dram_list = [energy_wr_opt_dram, energy_rd_opt_dram, energy_rd_wgt_dram, energy_rd_act_dram]
    energy_L2_list = [energy_wr_opt_L2, energy_rd_opt_L2, energy_rd_wgt_L2, energy_rd_act_L2]
    energy_die2die = 0;	energy_core2core = 0
    assert(DIE2DIE_energy_ratio!=NOC_energy_ratio)
    for item in link_energy:
        if link_energy_ratio[item] == DIE2DIE_energy_ratio:
            energy_die2die += link_energy[item]
        elif link_energy_ratio[item] == NOC_energy_ratio:
            energy_core2core += link_energy[item]
        elif link_energy_ratio[item] == DIE2DIE_energy_ratio + DRAM_energy_ratio:
            energy_die2die += link_energy[item]
            energy_dram_list[2] += link_energy[item]
        else:
            print ("FATAL: link's energy ratio is incorrect!")
            sys.exit()
    if PC3 > 1:
        output_num = runtimeP * runtimeQ * runtimeK
        chiplet_num = runtimeChipNum
        delay_psum, energy_psum_list = calPSumAllReduce(output_num, chiplet_num, PC3)
    else:
        delay_psum = 0
        energy_psum_list = [0,0,0]

    worstlinks = []
    for item in F_cur:
        if F_cur[item] == degrade_ratio: 
            worstlinks.append(item)
        if dram_to_L2_F_cur == degrade_ratio:
            worstlinks.append("dram2L2")
        if L2_to_DRAM_F_cur == degrade_ratio:
            worstlinks.append("L2toDRAM")
        if nop_F_cur == degrade_ratio:
            worstlinks.append("NoP")
	
    pkt_needed = {}
    pkt_needed["input_L1"] = core_pkt_num_rd_act
    pkt_needed["weight_L1"] = core_pkt_num_rd_wgt
    pkt_needed["output_rd_L1"] = core_pkt_num_rd_opt
    pkt_needed["output_wr_L1"] = core_pkt_num_wr_opt
    pkt_needed["input_DRAM"] = (chip_pkt_num_rd_act["DRAM"])
    pkt_needed["weight_DRAM"] = (chip_pkt_num_rd_wgt["DRAM"])
    pkt_needed["output_rd"] = (chip_pkt_num_rd_opt)
    pkt_needed["output_wr"] = (chip_pkt_num_wr_opt)
    pkt_needed["chiplet_parallel"] = [PK3,PQ3,PP3,PC3]
    
    neu_needed = {}
    neu_needed["input_L1"] = core_neu_num_rd_act / CoreNum / ChipNum
    neu_needed["weight_L1"] = core_neu_num_rd_wgt / CoreNum / ChipNum
    neu_needed["output_rd_L1"] = core_neu_num_rd_opt / CoreNum / ChipNum
    neu_needed["output_wr_L1"] = core_neu_num_wr_opt / CoreNum / ChipNum
    neu_needed["input_DRAM"] = (chip_neu_num_rd_act["DRAM"]/ ChipNum)
    neu_needed["weight_DRAM"] = (chip_neu_num_rd_wgt["DRAM"]/ ChipNum)
    neu_needed["output_rd"] = (chip_neu_num_rd_opt/ ChipNum)
    neu_needed["output_wr"] = (chip_neu_num_wr_opt/ ChipNum)

    return(degrade_ratio*compuation_cycles, degrade_ratio, degrade_ratio_dict, pkt_needed, neu_needed, compuation_cycles,runtime_list,cp_list,utilization_ratio_list, \
        energy_dram_list, energy_L2_list,energy_L1_list, energy_die2die, energy_MAC, energy_psum_list, delay_psum, worstlinks)


'''
INPUT : for_list, comm_patterns, parallel_dim_list, partition_list, network_param, HW_param, memory_param, NoC_param, if_multicast, flag = "ours"
'''