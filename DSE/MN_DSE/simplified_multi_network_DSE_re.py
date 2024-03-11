from linecache import lazycache
import os
import math
import copy
import random
from matplotlib import pyplot as plt
import argparse
import numpy as np
import shutil
import datetime
import sys
from workload_partition import * 
from communication_model import * 
from itertools import *

PE_Frequency = 1000 * 1000 * 1000
Optimization_Objective_index = {"edp": None, "energy": 'E', "latency": 'L'}
comm_tag = True
# --- debug param
debug_in_getIdealParam = False
debug_in_BWR = False
debug_in_BWR_simple = False
debug_in_evaluation_tp_sp = False
debug_in_evoluation_temporal_spatial = False
debug_in_evoluation_temporal_spatial_simple = False
debug_in_record_fitness_iter = False

cur_dir = os.path.dirname(os.path.abspath(__file__))
nn_param_dir = "/home/wangxy/workspace/chiplet/wxy_chiplet/DSE/SE_DSE/nn_file"

result_outdir = os.path.join(cur_dir,"multi_nn_result")
os.makedirs(result_outdir, exist_ok=True)
result_plot = os.path.join(result_outdir, "plot")
os.makedirs(result_plot, exist_ok=True)

def getNNParam(nn_name):
    # 获取神经网络的计算量和参数量信息
    # --- 对神经网络性能进行理论分析
    nn_file_path = os.path.join(nn_param_dir, nn_name + ".txt")
    nn_f = open(nn_file_path)

    # print("network model ----- " + nn_name + " -------------")

    layer_computation_num_dict = {}
    layer_param_num_dict = {}
    lines = nn_f.readlines()
    nnParam_dict = {}
    layer_id = 1
    for line in lines:
        if line.startswith("#") or line.startswith("*"):
            pass
        else:
            line = line.replace("\n","")
            line_item = line.split(" ")
            layer_name = (line_item[0].split('-'))[0]
            #layer_id = int(layer_name.replace("layer",""))
            H = int(line_item[1])
            M = int(line_item[2])
            P = int(line_item[8])
            Q = int(line_item[9])
            C = int(line_item[3])
            K = int(line_item[7])
            R = int(line_item[4])
            S = int(line_item[4])
            layer_computation_num = P * Q * K * R * S * C
            layer_param_num = H*M*C + P*Q*K + R*S*C*K
            layer_computation_num_dict[layer_name] = layer_computation_num
            layer_param_num_dict[layer_name] = layer_param_num
            nnParam_dict[layer_name] = layer_computation_num * layer_param_num / 1000000000000
    nn_f.close()
    return nnParam_dict

def get_workload_partition(architecture, nn_list, ratio, objective):
    fitnesses = {}
    workload_fitnesses = {}
    workload_dict = {}
    for nn_name in nn_list:
        workload_dict[nn_name] = getWorkloadPartition(nn_name, 'param_vs_compute', ratio)
        fitnesses[nn_name], unique_fitnesses = extract_fitness(architecture, nn_name, objective)
        workload_fitnesses[nn_name] = merge_layer(workload_dict[nn_name], fitnesses)
    return workload_dict, workload_fitnesses

# -------------------------------------------------------------
# get_total_group : 获得num个物体的所有组合方案，TH为划分数目阈值
# -------------------------------------------------------------
def get_total_group(num, TH, TH_min = 1):
    nodes = [i for i in range(num)]
    groups_3 = [ [{0}, {1}, {2}], [{0, 1}, {2}], [{0}, {1, 2}], [{0, 2}, {1}], [{0, 1, 2}] ]
    
    nodes = [0, 1, 2, 3]
    groups_4 = [[{0}, {1}, {2}, {3}], [{0, 1, 2, 3}], [{0}, {1, 2, 3}], [{1}, {0, 2, 3}], [{2}, {1, 0, 3}], [{3}, {1, 2, 0}] ]
    for ss in combinations(nodes, 2):
        group = [set(ss)]
        for node in nodes:
            if node not in ss:
                group.append(set([node]))
        groups_4.append(group)
    
    groups_2 = [[{0}, {1}], [{0, 1}]]
    groups_1 = [[{0}]]
    groups = {1: groups_1, 2: groups_2, 3: groups_3, 4: groups_4}
    groups_new = {}
    for n, group in groups.items():
        for g in group:
            if len(g) > TH or len(g) < TH_min:
                pass
            else:
                if n not in groups_new:
                    groups_new[n] = []
                groups_new[n].append(g)
    return groups_new
        
class multi_network_DSE:
    def __init__(self, architecture, chiplet_num, mem_num, nn_list, Optimization_Objective, tp_TH, sp_TH, topology='ours', partition_ratio=0.5, BW_tag = 1, debug_tag = 0, layout_mapping_method = 'balance'):
        self.architecture = architecture
        self.chiplet_num = chiplet_num
        self.mem_num = mem_num
        self.cluster_size = 4
        self.cluster_num = math.ceil(chiplet_num / self.cluster_size)
        self.nn_list = nn_list
        self.partition_ratio = partition_ratio
        self.topology = topology
        
        # --- workload partition ---
        self.workload_dict = None
        self.workload_list = []
        
         # --- workload fitness variable ---
        self.ideal_param_dict           = None
        self.ideal_param_dict_workload  = None
        self.workload_fitness_dict      = {}
        self.workload_NeuNeeded_dict    = {}
       
        
        self.merge_workload_fitness_dict    = None
        self.merge_workload_NeuNeeded_dict  = None
        self.merge_workload_dict            = None
        
        # --- config parameter --- 
        self.debug_tag = debug_tag
        self.BW_Reallocate_tag = BW_tag
        self.layout_mapping_method = layout_mapping_method
        self.Optimization_Objective = Optimization_Objective
        
        # --- TH Param ---
        self.space_tp_ratio = tp_TH         # max tp num ratio
        self.space_sp_ratio = sp_TH         # max sp num ratio
        self.chiplet_partition_TH = 80     # max chiplet partition num per TP Tile
        self.workload_schedule_TH = 20000    # max workload schedule method num

        self.chiplet_partition_dict = None

        # --- record variable ---
        self.fitness_best = None
        self.fitness_record = []
        self.schedule_best = None
        self.schedule_record = []
        self.Nchip_partition_best = None
        self.Nchip_par_record = []
        self.tp_sp_space_best = None
        self.tp_sp_space_record = []
        self.sample_num = 0
        self.merge_workload_dict_best = None
        
        self.fitness_best_comm = None
        self.fitness_record_comm = []
        self.schedule_best_comm = None
        self.schedule_record_comm = []
        self.Nchip_partition_best_comm = None
        self.Nchip_par_record_comm = []
        self.tp_sp_space_best_comm = None
        self.sample_num_comm = 0
        self.merge_workload_dict_best_comm = None

        # --- final result ---
        self.final_workload_BWNeeded_dict = None
        self.final_workload_FN_dict = None
        self.final_merge_workload_fitness_dict = None
        self.final_merge_workload_BWNeeded_dict = None
        
        self.final_workload_BWNeeded_dict_comm = None
        self.final_workload_FN_dict_comm = None
        self.final_merge_workload_fitness_dict_comm = None
        self.final_merge_workload_BWNeeded_dict_comm = None
        
        # ---- nop comm ---
        self.route_table = None
        self.linkBW = None
        self.links = None

    # ---------------------
    # initialize: 初始化操作
    # ---------------------
    def initialize(self):
        self.setTotalWorkloadFitness()
        self.getIdealParam()
        self.best_fitness = None
        self.fitness_record = []
        self.sample_record = []
        self.distance_record = []
        
        self.route_table_mesh, self.linkBW_mesh, self.links_mesh = construct_Mesh(6, 4, BWperLink * 2)
        self.route_table_cmesh, self.linkBW_cmesh, self.links_cmesh = construct_CMesh(2, 2, 5, BWperLink * 2)
        if self.topology == 'mesh':
            self.route_table, self.linkBW, self.links = construct_Mesh(6, 4, BWperLink * 2)
        elif self.topology == 'cmesh':
            self.route_table, self.linkBW, self.links = construct_CMesh(2, 2, 5, BWperLink * 2)
        else:
            pass  

    # -----------------------------------------------
    # getIdealParam: 获得理论计算的各任务的参数量信息
    # --- ideal_param_dict: [nn_name]{layer_name: xx}
    # --- ideal_param_dict_workload: {w_name: xx}
    # -----------------------------------------------
    def getIdealParam(self):
        self.ideal_param_dict = {}
        self.ideal_param_dict_workload = {}
        if debug_in_getIdealParam:
            print("DEBUG IN getIdealParam()--------------------------")
            print("---ideal_param_dict_workload: ")
            
        for nn_name, workloads in self.workload_dict.items():
            self.ideal_param_dict[nn_name] = copy.deepcopy(getNNParam(nn_name))
            for w_name, layers in workloads.items():
                param = 0
                for layer in layers:
                    param += self.ideal_param_dict[nn_name][layer]
                self.ideal_param_dict_workload[w_name] = param

                if debug_in_getIdealParam:
                    print("---{}: {}".format(workload_name, param))
        
        if debug_in_getIdealParam:
            print("END------------------------------------------------")
            exit()
    
    # ---------------------------------------------------------------------------------------------------------------------------
    # setTotalWorkloadFitness: 获得每个Workload的性能参数
    # -- workload_dict : [nn_name][w_name][layer_name_list]; w_name 'resnet18w0'
    # -- workload_list : [w_name]
    # -- workload_fitness_dict : [w_name][c_num][layer]{"E": xx, "L": xx}
    # -- workload_NeuNeeded_dict : [w_name][c_num][layer]{"ifmap_DRAM": xx, "ofmap_DRAM": xx, "weight_DRAM": xx, "parallel": []}
    # ---------------------------------------------------------------------------------------------------------------------------
    def setTotalWorkloadFitness(self):
        workloads, fitnesses = get_workload_partition(self.architecture, self.nn_list, self.partition_ratio, self.Optimization_Objective)
        
        # ----------------
        # --- Workload ---
        # ----------------
        self.workload_dict = {}
        for nn_name, items in workloads.items():
            self.workload_dict[nn_name] = {}
            for w_name, item in items.items():
                self.workload_dict[nn_name][w_name] = item["layer_name_list"]
                self.workload_list.append(w_name)
                
        # -----------------------------
        # --- Fitness and NeuNeeded ---
        # -----------------------------
        self.workload_fitness_dict = {}
        self.workload_NeuNeeded_dict = {}
        for nn_name in fitnesses:
            for w_name, items in fitnesses[nn_name].items():
                self.workload_fitness_dict[w_name] = {}
                self.workload_NeuNeeded_dict[w_name] = {}
                for c_num, item in items.items():
                    self.workload_fitness_dict[w_name][c_num] = {}
                    self.workload_NeuNeeded_dict[w_name][c_num] = {}
                    for layer, fitness in item.items():
                        self.workload_fitness_dict[w_name][c_num][layer] = {"E": fitness["energy"], "L": fitness["latency"]}
                        self.workload_NeuNeeded_dict[w_name][c_num][layer] = copy.deepcopy(fitness["neu_needed"])
                        self.workload_NeuNeeded_dict[w_name][c_num][layer]["parallel"] = fitness["chiplet_parallel"]
        
    # -----------------------------------------------------
    # calIdealParam: 计算每个[tp_id, sp_id]内任务的理论参数量
    # -----------------------------------------------------
    def calIdealParam(self, tp_sp_space):
        tp_sp_idealParam = {}
        for tp_id, sp_space  in tp_sp_space.items():
            tp_sp_idealParam[tp_id] = {}
            for sp_id, workload_list in sp_space.items():
                tp_sp_idealParam[tp_id][sp_id] = 0
                for workload in workload_list:
                    tp_sp_idealParam[tp_id][sp_id] += self.ideal_param_dict_workload[workload]
        return tp_sp_idealParam

    # -----------------------------------------------------
    # mergeWorkload: 对于一个Tile内的相同网络的子任务进行合并
    # -----------------------------------------------------
    def merge(self, w_name_list):
        w_name_new = ''
        for w_name in w_name_list:
            w_name_new += w_name
        self.merge_workload_fitness_dict[w_name_new]    = copy.deepcopy(self.workload_fitness_dict[w_name_list[0]])
        self.merge_workload_NeuNeeded_dict[w_name_new]  = copy.deepcopy(self.workload_NeuNeeded_dict[w_name_list[0]])
        for w_name in w_name_list[1:]:
            for c_num, layers in self.workload_fitness_dict[w_name].items():
                # self.merge_workload_fitness_dict[w_name_new][c_num].update(copy.deepcopy(layers))
                # self.merge_workload_NeuNeeded_dict[w_name_new][c_num].update(copy.deepcopy(layers))
                layer_pre = list(self.merge_workload_NeuNeeded_dict[w_name_new][c_num].keys())[-1]
                layer_cur = list(layers.keys())[0]
                if self.merge_workload_NeuNeeded_dict[w_name_new][c_num][layer_pre]["ofmap_store_onchip_tag"] == 1 and self.merge_workload_NeuNeeded_dict[w_name_new][c_num][layer_pre]["parallel"] == self.workload_NeuNeeded_dict[w_name][c_num][layer_cur]["parallel"]:
                    layer_name = '{}_{}'.format(layer_pre, layer_cur)
                    self.merge_workload_NeuNeeded_dict[w_name_new][c_num][layer_name] = self.merge_workload_NeuNeeded_dict[w_name_new][c_num].pop(layer_pre)
                    self.merge_workload_fitness_dict[w_name_new][c_num][layer_name] = self.merge_workload_fitness_dict[w_name_new][c_num].pop(layer_pre)
                    self.merge_workload_fitness_dict[w_name_new][c_num][layer_name]["E"] += self.workload_fitness_dict[w_name][c_num][layer_cur]["E"]
                    self.merge_workload_fitness_dict[w_name_new][c_num][layer_name]["L"] += self.workload_fitness_dict[w_name][c_num][layer_cur]["L"]
                    self.merge_workload_NeuNeeded_dict[w_name_new][c_num][layer_name]["ofmap_DRAM"] = self.workload_NeuNeeded_dict[w_name][c_num][layer_cur]["ofmap_DRAM"]
                    self.merge_workload_NeuNeeded_dict[w_name_new][c_num][layer_name]["ofmap_store_onchip_tag"] = self.workload_NeuNeeded_dict[w_name][c_num][layer_cur]["ofmap_store_onchip_tag"]
                    
                    self.merge_workload_fitness_dict[w_name_new][c_num].update(copy.deepcopy(layers))
                    self.merge_workload_fitness_dict[w_name_new][c_num].pop(layer_cur)
                    self.merge_workload_NeuNeeded_dict[w_name_new][c_num].update(copy.deepcopy(self.workload_NeuNeeded_dict[w_name][c_num]))
                    self.merge_workload_NeuNeeded_dict[w_name_new][c_num].pop(layer_cur)
                else:
                    self.merge_workload_fitness_dict[w_name_new][c_num].update(copy.deepcopy(layers))
                    self.merge_workload_NeuNeeded_dict[w_name_new][c_num].update(self.workload_NeuNeeded_dict[w_name][c_num])
        return w_name_new
    
    def mergeWorkload(self, tp_sp_space):
        self.merge_workload_fitness_dict = {}
        self.merge_workload_NeuNeeded_dict = {}
        self.merge_workload_dict = {}
        merge_tp_sp_space = {}
        for tp_id, sp_space in tp_sp_space.items():
            merge_tp_sp_space[tp_id] = {}
            for sp_id, w_list in sp_space.items():
                app_name_list = {}
                merge_tp_sp_space[tp_id][sp_id] = []
                for workload in w_list:
                    app_name = (workload.split('w'))[0]
                    w_id = int((workload.split('w'))[1])
                    if app_name not in app_name_list:
                        app_name_list[app_name] = []
                    app_name_list[app_name].append(w_id)
                
                for app_name, w_list in app_name_list.items():
                    w_list.sort()
                    w_name_list = []
                    layer_name_list = []
                    for id in w_list:
                        w_name = '{}w{}'.format(app_name, id)
                        w_name_list.append(w_name)
                        layer_name_list += self.workload_dict[app_name][w_name]
                    w_name = self.merge(w_name_list)
                    if app_name not in self.merge_workload_dict:
                        self.merge_workload_dict[app_name] = {}
                    self.merge_workload_dict[app_name][w_name] = layer_name_list
                    merge_tp_sp_space[tp_id][sp_id].append(w_name)
        
        return merge_tp_sp_space
                    
    # ---------------------------------
    # getFinalBWFN：得到最终的BW与FN参数
    # ---------------------------------
    def getFinalBWFN(self):
        self.final_workload_BWNeeded_dict = {}
        self.final_workload_FN_dict = {}
        tp_sp_space = self.tp_sp_space_best
        chiplet_partition = self.Nchip_partition_best

        for tp_id, sp_space in tp_sp_space.items():
            for sp_id, w_name_list in sp_space.items():
                Nchip = chiplet_partition[tp_id][sp_id]

                for w_name in w_name_list:
                    self.final_workload_BWNeeded_dict[w_name] = {}
                    self.final_workload_FN_dict[w_name] = {}
                    w_name_list = w_name.split("w")
                    app_name = w_name_list[0]
                    w_id_list = w_name_list[1:]
                    workload_tag = {}
                    for w_id in w_id_list:
                        if len(w_id_list) == 1:
                            tag = "aloneTile"
                        elif w_id == w_id_list[0]:
                            tag = "headTile"
                        elif w_id == w_id_list[-1]:
                            tag = "tailTile"
                        else:
                            tag = "midTile"
                        w_id_name = app_name + "w" + w_id
                        workload_tag[w_id_name] = tag

                    layer_offset = 0
                    for w_id_name, tag in workload_tag.items():
                        for layer_id, BW_list in self.workload_BWNeeded_dict[w_id_name][Nchip].items():
                                layer_id_real = layer_id + layer_offset
                                assert(layer_id_real not in self.final_workload_BWNeeded_dict[w_name])
                                self.final_workload_BWNeeded_dict[w_name][layer_id_real] = BW_list[tag]
                                self.final_workload_FN_dict[w_name][layer_id_real] = self.workload_FlitNeeded_dict[w_id_name][Nchip][layer_id][tag]
                        layer_offset = layer_id_real

        MNN_result_outdir = os.path.join(cur_dir,"multi_nn_result")
        os.makedirs(MNN_result_outdir, exist_ok=True)
        MNN_result_outdir = os.path.join(MNN_result_outdir,"final_result_output")
        if os.path.exists(MNN_result_outdir):
            shutil.rmtree(MNN_result_outdir)
        os.makedirs(MNN_result_outdir, exist_ok=True)
        BW_dir = os.path.join(MNN_result_outdir,"BW_result")
        os.makedirs(BW_dir, exist_ok=True)
        FN_dir = os.path.join(MNN_result_outdir,"FN_result")
        os.makedirs(FN_dir, exist_ok=True)
        
        title_FN = "layer_id\tiact_DRAM_FN\tweight_DRAM_FN\tiact_L2_FN\tweight_L2_FN\toact_rd_FN\toact_wr_FN\tchiplet_spatial_parallel"
        title_BW = "layer_id\tlatency\tNoC_DR\tL2_to_DRAM_DR\tDRAM_to_L2_DR\t"
        for workload_name in self.final_workload_BWNeeded_dict:
            BW_file = BW_dir + "/" + workload_name + ".txt"
            FN_file = FN_dir + "/" + workload_name + ".txt"
            BW_f = open(BW_file, 'w')
            FN_f = open(FN_file, 'w')
            print(title_BW, file = BW_f)
            print(title_FN, file = FN_f)
            for layer_id in self.final_workload_BWNeeded_dict[workload_name]:
                BW_list = self.final_workload_BWNeeded_dict[workload_name][layer_id]
                FN_list = self.final_workload_FN_dict[workload_name][layer_id]

                BW_line = "{}\t{}\t{}\t{}\t{}".format(layer_id, BW_list[0], BW_list[1], BW_list[2], BW_list[3])
                FN_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(layer_id, FN_list[0], FN_list[1], FN_list[2], FN_list[3], FN_list[4], FN_list[5], FN_list[6])
                
                print(BW_line, file = BW_f)
                print(FN_line, file = FN_f)
            
            BW_f.close()
            FN_f.close()
        
        sp_tp_result_file = MNN_result_outdir + "/sp_tp_out.txt"
        sp_tp_result_f = open(sp_tp_result_file, 'w')
        for tp_id, sp_space in tp_sp_space.items():
            line = "tp" + str(tp_id)
            for sp_id, workload_list in sp_space.items():
                Nchip = chiplet_partition[tp_id][sp_id]
                line += "\t(sp{}:{})".format(sp_id, Nchip)
                for workload in workload_list:
                    line += " " + workload
            print(line, file = sp_tp_result_f)
        sp_tp_result_f.close()

    # -------------------------------------------------
    # getChipletPartionDict: 获得chiplet数目的可能拆分
    # chiplet_partition_dict : 
        ## { 1:[[36]],
        ##   2:[[1,35],[2,34]...],
        ##   ...
        ##   max_sp_num:[[],[],...] }
    # -------------------------------------------------
    def getChipletPartionDict(self, max_sp_num):
        par_per_sp_TH = 100
        par_num = 0
        unchange_TH = 50
        unchange_times = 0
        total_block_num = math.ceil(self.chiplet_num / self.mem_num)
        
        chiplet_partition_dict = {}
        for i in range(max_sp_num):
            sp_num = i + 1
            par_num = 0
            unchange_times = 0
            chiplet_partition_dict[sp_num] = []
            while(1):
                chiplet_partition = []
                Nchip_rest = total_block_num - sp_num
                for i in range(sp_num-1):
                    Nchip = random.randint(0, Nchip_rest)
                    chiplet_partition.append(Nchip+1)
                    Nchip_rest -= Nchip
                chiplet_partition.append(Nchip_rest+1)
                chiplet_partition.sort()
                if chiplet_partition not in chiplet_partition_dict[sp_num]:
                    chiplet_partition_dict[sp_num].append(chiplet_partition)
                    unchange_times = 0
                    par_num += 1
                else:
                    unchange_times += 1
                    
                if unchange_times == unchange_TH or par_num == par_per_sp_TH:
                    break
        for sp_num, partitions in chiplet_partition_dict.items():
            for i in range(len(partitions)):
                assert(len(partitions[i]) == sp_num)
                for j in range(sp_num):
                    chiplet_partition_dict[sp_num][i][j] *= self.mem_num
        return chiplet_partition_dict

    # ----------------------------------
    # calSP: SP Tile Fitness Computation
    # ----------------------------------
    def calSP(self, fitness_dict, id):
        if id == None:
            sp_name = "("
        else:
            sp_name = "sp" + str(id) + "("
        latency_sp = 0
        energy_sp = 0
        for name, fitness in fitness_dict.items():
            sp_name += name + "_"
            latency_sp += fitness['L']
            energy_sp += fitness['E']
        sp_name = sp_name.strip("_")
        sp_name += ")"

        fitnesses = {'E': energy_sp, 'L': latency_sp}
        return sp_name, fitnesses
    
    # ----------------------------------
    # calTP: TP Tile Fitness Computation
    # ----------------------------------
    def calTP(self, fitness_dict, id):
        tp_name = "tp" + str(id) + "("
        latency_tp = 0
        energy_tp = 0
        for name, fitness in fitness_dict.items():
            tp_name += name + "_"
            if fitness['L'] > latency_tp:
                latency_tp = fitness['L']
            energy_tp += fitness['E']
        tp_name = tp_name.strip("_")
        tp_name += ")"

        fitnesses = {'E': energy_tp, 'L': latency_tp}
        return tp_name, fitnesses

    # ----------------------------------------------------------------------
    # getChipletAllocateMethods_random: get chiplet allocate method randomly
    # ----------------------------------------------------------------------
    def getChipletAllocateMethods_random(self, sp_idealParam, TH):
        sp_num = len(sp_idealParam)
        method_num = len(self.chiplet_partition_dict[sp_num])
        

        # id_list: 随机化选取的方案的id
        # --- 随机获取chiplet allocate方案中的min(method_num, TH)个方案
        id_list = list(range(method_num))
        random.shuffle(id_list)

        sp_idealParam_order = sorted(sp_idealParam.items(), key = lambda x: x[1])
        sp_Nchip_list = []

        # 按比例分配
        for i in range(TH):
            if i >= method_num:
                break
            index = id_list[i]
            chip_num_list = self.chiplet_partition_dict[sp_num][index]    # 1. 获取的划分的方案
            
            # 2. 根据SP Tile理论的参数量sp_idealParam_order对比，分配对应的chiplet数目
            sp_Nchip = {}
            for ii in range(sp_num):
                chip_num = chip_num_list[ii]
                sp_id = sp_idealParam_order[ii][0]
                sp_Nchip[sp_id] = chip_num
            assert(sp_Nchip not in sp_Nchip_list)
            sp_Nchip_list.append(sp_Nchip)
        
        # 随机生成
        random.shuffle(id_list)
        for i in range(TH):
            index = id_list[i%method_num]
            chip_num_list = self.chiplet_partition_dict[sp_num][index]    # 1. 获取的划分的方案
            random.shuffle(chip_num_list)

            sp_Nchip = {}
            for ii in range(sp_num):
                chip_num = chip_num_list[ii]
                sp_Nchip[ii] = chip_num
            if sp_Nchip not in sp_Nchip_list:
                sp_Nchip_list.append(sp_Nchip)
        return sp_Nchip_list

    # ----------------------------------------------------------------
    # evaluation_tp_sp: 评估函数，评估任务调度方案
    # ---include: chiplet allocate + bandwidth reallocate + evaluation
    # ----------------------------------------------------------------
    def evaluation_tp_sp(self, tp_sp_space, tp_sp_idealParam):
        
        if debug_in_evaluation_tp_sp:
            print("DEBUG IN evaluation_tp_sp()----------------------")

        tp_sp_fitness = {}
        best_Nchip_partition = {}
        
        tp_sp_fitness_comm = {}
        best_Nchip_partition_comm = {}

        # 评估开始：

        # 1. 每个TP Tile内部探索与评估，以TP Tile的fitness最优为指标迭代
        # ---每个TP Tile内部进行chiplet Allocate的探索
        # ---每个SP Tile内部进行BW Reallocate的探索
        for tp_id, sp_space in tp_sp_space.items():
            # Evaluation in the same TP Tile
            
            sp_idealParam = tp_sp_idealParam[tp_id]    
            if debug_in_evaluation_tp_sp:
                print('tp_id = ', tp_id)
                print('---sp_idealParam = ', sp_idealParam)

            # 1.1 Randomize Chiplet Allocate Methods
            sp_Nchip_list = self.getChipletAllocateMethods_random(sp_idealParam, self.chiplet_partition_TH)

            # 1.2 Chiplet Allocate 方案遍历
            best_tp_fitness = None
            best_tp_name = None
            best_Nchip = None
            
            best_tp_fitness_comm = None
            best_tp_name_comm = None
            best_Nchip_comm = None
            
            for sp_Nchip in sp_Nchip_list:
                # 1.2.1 Get Workload Fitness
                sp_workload_fitness = {}
                for sp_id in sp_Nchip:
                    sp_workload_fitness[sp_id] = {}
                    Nchip = sp_Nchip[sp_id]
                    for workload in sp_space[sp_id]:
                        fitness = self.merge_workload_fitness_dict[workload][Nchip]
                        sp_workload_fitness[sp_id][workload] = {'L': 0, 'E': 0}
                        for layer, items in fitness.items():
                            sp_workload_fitness[sp_id][workload]['L'] += items['L']
                            sp_workload_fitness[sp_id][workload]['E'] += items['E']
                if debug_in_evaluation_tp_sp:
                    print('sp_Nchip = ', sp_Nchip)
                    print('sp_workload_fitness = ', sp_workload_fitness)
                
                # 1.2.3 Cal Fitness
                sp_fitness_dict = {}
                for sp_id, workload_fitness in sp_workload_fitness.items():
                    # TODO
                    sp_name, fitness = self.calSP(workload_fitness, sp_id)
                    sp_fitness_dict[sp_name] = fitness
                tp_name, tp_fitness = self.calTP(sp_fitness_dict, tp_id)

                # 1.2.4 迭代比较
                Objective_id = Optimization_Objective_index[self.Optimization_Objective]
                if best_tp_fitness == None or tp_fitness[Objective_id] < best_tp_fitness[Objective_id]:
                    best_tp_fitness = copy.deepcopy(tp_fitness)
                    best_tp_name = tp_name
                    best_Nchip = copy.deepcopy(sp_Nchip)

                
                self.sample_num += 1

            for sp_Nchip in sp_Nchip_list:
                # 1.2.1 Get Workload Fitness
                sp_workload_fitness_comm = {}
                for sp_id in sp_Nchip:
                    sp_workload_fitness_comm[sp_id] = {}
                    Nchip = sp_Nchip[sp_id]
                    for workload in sp_space[sp_id]:
                        fitness = self.merge_workload_fitness_dict[workload][Nchip]
                        sp_workload_fitness_comm[sp_id][workload] = {'L': 0, 'E': 0}
                        for layer, items in fitness.items():
                            comm_num = self.merge_workload_NeuNeeded_dict[workload][Nchip][layer]
                            ifmapD = comm_num['ifmap_DRAM']
                            weightD = comm_num['weight_DRAM']
                            ofmapD = comm_num['ofmap_DRAM']
                            parallel = comm_num['parallel']
                            # commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, 'mesh', self.route_table, self.linkBW, self.links)
                            commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, 'ours', None, None, None)
                            sp_workload_fitness_comm[sp_id][workload]['L'] += items['L'] + commPerf
                            sp_workload_fitness_comm[sp_id][workload]['E'] += items['E']

                
                # 1.2.3 Cal Fitness
                sp_fitness_dict = {}
                for sp_id, workload_fitness in sp_workload_fitness.items():
                    # TODO
                    sp_name, fitness = self.calSP(workload_fitness, sp_id)
                    sp_fitness_dict[sp_name] = fitness
                tp_name, tp_fitness = self.calTP(sp_fitness_dict, tp_id)

                # 1.2.4 迭代比较
                Objective_id = Optimization_Objective_index[self.Optimization_Objective]
                if best_tp_fitness == None or tp_fitness[Objective_id] < best_tp_fitness[Objective_id]:
                    best_tp_fitness = copy.deepcopy(tp_fitness)
                    best_tp_name = tp_name_comm
                    best_Nchip = copy.deepcopy(sp_Nchip)
                    
                # 1.2.3 Cal Fitness
                sp_fitness_dict_comm = {}
                for sp_id, workload_fitness in sp_workload_fitness_comm.items():
                    # TODO
                    sp_name, fitness = self.calSP(workload_fitness, sp_id)
                    sp_fitness_dict_comm[sp_name] = fitness
                tp_name_comm, tp_fitness_comm = self.calTP(sp_fitness_dict_comm, tp_id)

                # 1.2.4 迭代比较
                Objective_id = Optimization_Objective_index[self.Optimization_Objective]
                if best_tp_fitness_comm == None or tp_fitness_comm[Objective_id] < best_tp_fitness_comm[Objective_id]:
                    best_tp_fitness_comm = copy.deepcopy(tp_fitness_comm)
                    best_tp_name_comm = tp_name_comm
                    best_Nchip_comm = copy.deepcopy(sp_Nchip)

                
                self.sample_num += 1

            # 1.3 最优方案记录，得到TP Tile内的评估结果    
            tp_sp_fitness[best_tp_name] = best_tp_fitness
            best_Nchip_partition[tp_id] = copy.deepcopy(best_Nchip)
            
            tp_sp_fitness_comm[best_tp_name_comm] = best_tp_fitness_comm
            best_Nchip_partition_comm[tp_id] = copy.deepcopy(best_Nchip_comm)

        # 2. calSP, 对多个TP Tile进行合并评估
        schedule_name, total_fitness = self.calSP(tp_sp_fitness, None)
        
        schedule_name_comm, total_fitness_comm = self.calSP(tp_sp_fitness_comm, None)

        if debug_in_evaluation_tp_sp:
            print('schedule_name = ', schedule_name)
            print('fitness = ', total_fitness)
        
        #exit()
        return schedule_name, total_fitness, best_Nchip_partition, schedule_name_comm, total_fitness_comm, best_Nchip_partition_comm

    # --------------------------------------------------------------------------------
    # decodeCode: workload_code进行解码，解码成每个[tp_id, sp_id]内包含哪些workload的形式
    # --------------------------------------------------------------------------------
    def decodeCode(self, code):
        sp_max_dict = {}
        for w_name, sp_tp_id in code.items():
            sp_id = sp_tp_id[1]
            tp_id = sp_tp_id[0]
            if tp_id not in sp_max_dict:
                sp_max_dict[tp_id] = 0
            if sp_id > sp_max_dict[tp_id]:
                sp_max_dict[tp_id] = sp_id
        
        tp_sp_space = {}
        for tp_id in range(len(sp_max_dict)):
            tp_sp_space[tp_id] = {}
            sp_max = sp_max_dict[tp_id]
            for sp_id in range(sp_max+1):
                tp_sp_space[tp_id][sp_id] = []

        for w_name, sp_tp_id in code.items():
            sp_id = sp_tp_id[1]
            tp_id = sp_tp_id[0]
            #if tp_id not in tp_sp_space:
            #    tp_sp_space[tp_id] = {}
            #if sp_id not in tp_sp_space[tp_id]:
            #    tp_sp_space[tp_id][sp_id] = []
            tp_sp_space[tp_id][sp_id].append(w_name)
        return tp_sp_space

    # --------------------------------------------------------------------------------
    # getWorkloadScheduleCodeList_random: workload的调度，随机获得workload的tp_sp_id映射
    # --- workload_position_list: 任务调度方案列表, [workload_name][tp_id, sp_id]
    # --- sp_TH: chiplet拆分的最多块数
    # --------------------------------------------------------------------------------
    def getWorkloadScheduleCodeList_random(self, tp_ratio, sp_ratio, TH):
        
        def get_sp_id_dict_1(tp_TH, nn_sp_partitions, nn_num):
            sp_id_dict = {}
            for tp_id in range(tp_TH):
                random.shuffle(nn_sp_partitions)
                partition = nn_sp_partitions[0]
                sp_list = [0 for _ in range(nn_num)] # sp_list: key(app_id), value(sp_id)
                for i, ss in enumerate(partition):
                    for s in ss:
                        sp_list[s] = i
                sp_id_dict[tp_id] = sp_list
            return sp_id_dict

        def get_tp_id_dict(workload_dict, tp_max):
            tp_id_dict = {}
            tp_exist = [0 for _ in range(tp_max)]
            # random generate tp id
            for nn_name, workload in workload_dict.items():
                w_num = len(workload)
                tp_list = []
                for _ in range(w_num):
                    tp_id = random.randint(0, tp_max-1)
                    tp_list.append(tp_id)
                    tp_exist[tp_id] = 1
                tp_list.sort()
                tp_id_dict[nn_name] = tp_list
                
            # 删除没有任务的TP
            exchange = []
            id = 0
            for i in tp_exist:
                exchange.append(id)
                id += i  
            for nn_name, tp_list in tp_id_dict.items():
                for i, tp in enumerate(tp_list):
                    tp_id_dict[nn_name][i] = exchange[tp]
                    
            return tp_id_dict

        def get_sp_id_dict(tp_id_dict, nn_sp_partitions):
            sp_id_dict = {}
            tp_workloads = {}
            for nn_name, tp_list in tp_id_dict.items():
                tp_set = set(tp_list)
                for tp in tp_set:
                    if tp not in tp_workloads:
                        tp_workloads[tp] = []
                    tp_workloads[tp].append(nn_name)
            
            for tp, workloads in tp_workloads.items():
                sp_id_dict[tp] = {}
                random.shuffle(nn_sp_partitions[len(workloads)])
                partition = nn_sp_partitions[len(workloads)][0]
                for i, group in enumerate(partition):
                    for n in group:
                        sp_id_dict[tp][workloads[n]] = i
            return sp_id_dict
                    
        # --- 设置SP和TP的阈值 ---
        app_num = len(self.workload_dict)
        avg_w_num = len(self.workload_list)
        sp_min = 1
        if sp_ratio == 0:
            sp_TH = 1
        else:
            sp_TH = math.ceil(app_num * sp_ratio)
        if tp_ratio == 0:
            tp_TH = 1
            if app_num == 4:
                sp_min = app_num - 1
            else:
                sp_min = app_num
        else:
            tp_TH = math.ceil(tp_ratio * avg_w_num)

        workload_position_list = []

        nn_sp_partitions = get_total_group(app_num, sp_TH, sp_min)
    
        max_iter = 1000
        iter_num = 0
        while iter_num < max_iter:
            iter_num += 1
            # 1. 对网络内的任务进行时间上的调度
            tp_id_dict = get_tp_id_dict(self.workload_dict, tp_TH)
            
            # 2. 获得模型在每个时间点下的sp id，以模型为部署粒度
            # sp_id_dict: key(tp_id), value(sp_list), 获得每个timestamp下的网络模型的sp id
            sp_id_dict = get_sp_id_dict(tp_id_dict, nn_sp_partitions)

            # 3. 获得每个workload的tp与sp位置信息
            workload_position = {}
            for nn_name, workloads in self.workload_dict.items():
                for w_id, w_name in enumerate(workloads):
                    tp_id = tp_id_dict[nn_name][w_id]
                    sp_id = sp_id_dict[tp_id][nn_name]
                    workload_position[w_name] = [tp_id, sp_id]
            
            # 4. 若方案唯一则记录
            if workload_position not in workload_position_list:
                iter_num = 0 
                workload_position_list.append(workload_position)
                
            if len(workload_position_list) > TH:
                break
        return workload_position_list, sp_TH

    #########################################################
    # Layout mapping and bandwidth reallocate 
    #########################################################
    def layout_mapping(self):
        if debug_in_BWR_simple:
            print(str(sys._getframe().f_lineno) + ': self.tp_sp_space_best = ', self.tp_sp_space_best)
            print(str(sys._getframe().f_lineno) + ': self.Nchip_partition_best = ', self.Nchip_partition_best)
            print(str(sys._getframe().f_lineno) + ': self.final_merge_workload_fitness_dict = ', self.final_merge_workload_fitness_dict)
            print(str(sys._getframe().f_lineno) + ': self.final_workload_BWNeeded_dict = ', self.final_workload_BWNeeded_dict)
            
        sp_workload_fitness_bw = {}
        
        for tp_id, sp_space in self.tp_sp_space_best.items():
            sp_workload_fitness = {}
            sp_workload_fitness_bw[tp_id] = {}

            for sp_id in sp_space:
                sp_workload_fitness[sp_id] = {}
                Nchip = self.Nchip_partition_best[tp_id][sp_id]
                for workload in sp_space[sp_id]:
                    if debug_in_BWR:
                        print('workload = ', workload)
                        print('Nchip = ', Nchip)
                        print('self.final_merge_workload_fitness_dict[workload]= ', self.final_merge_workload_fitness_dict[workload])
                    fitness = self.final_merge_workload_fitness_dict[workload][Nchip]
                    sp_workload_fitness[sp_id][workload] = fitness
            if debug_in_BWR:
                print(str(sys._getframe().f_lineno) + ': sp_workload_fitness = ', sp_workload_fitness)
            if len(sp_workload_fitness.keys()) == 1:
                for sp_id, workload_fitness in sp_workload_fitness.items():
                    sp_workload_fitness_bw[tp_id][sp_id] = {}
                    for workload, fitness in workload_fitness.items():
                        sp_workload_fitness_bw[tp_id][sp_id][workload] = fitness
            else:
                '''
                sp_workload_fitness =  
                {
                    0: {
                        'resnet18': [1793044.2722481424, 514782.6666666666, 3483109258.2400002], 
                        'resnet50': [9630241.645624578, 1326631.466666666, 7259168719.872001]
                    }, 
                    1: {'resnet50same': [15876227.827040443, 1926204.2666666668, 8242234793.984004]}
                }
                sp_Nchip =  {1: 6, 0: 10}
                '''
                #########################################################
                # init
                #########################################################
                delay_min_dict = {}
                degrad_ratio_dict = {}
                dram_to_L2_min_dict = {}
                L2_to_DRAM_min_dict = {}
                total_cycle_dict = {}
                total_cycle_list = []
                eva_nn_chiplet_num_dict = {}

                for sp_id, workload_fitness in sp_workload_fitness.items():
                    sp_workload_fitness_bw[tp_id][sp_id] = {}
                    Nchip = self.Nchip_partition_best[tp_id][sp_id]
                    if debug_in_BWR:
                        print(str(sys._getframe().f_lineno) + ': workload_fitness.keys() = ', workload_fitness.keys())
                        print(str(sys._getframe().f_lineno) + ': Nchip = ', Nchip)
                    delay_min_dict[sp_id] = {}
                    degrad_ratio_dict[sp_id] = {}
                    L2_to_DRAM_min_dict[sp_id] = {}
                    dram_to_L2_min_dict[sp_id] = {}
                    eva_nn_chiplet_num_dict[sp_id] = Nchip
                    
                    for workload, fitness in workload_fitness.items():
                        sp_workload_fitness_bw[tp_id][sp_id][workload] = fitness
                        workload_BW_list = self.final_merge_workload_BWNeeded_dict[workload][Nchip]
                        
                        if debug_in_BWR:
                            print(str(sys._getframe().f_lineno) + ': workload_BW_list = ', workload_BW_list)
                        
                        for layer_id, BW_list in workload_BW_list.items():
                            # latency = BW_list[0]
                            # NoC_NR = BW_list[1]
                            # L2_to_DRAM_NR = BW_list[2]
                            # DRAM_to_L2_NR = BW_list[3]
                            delay_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[0]
                            degrad_ratio_dict[sp_id][workload + '_' + str(layer_id)] = max(BW_list[1], BW_list[2], BW_list[3])
                            L2_to_DRAM_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[2]
                            dram_to_L2_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[3]

                if debug_in_BWR:
                    print(str(sys._getframe().f_lineno) + ': delay_min_dict = ', delay_min_dict)
                    print(str(sys._getframe().f_lineno) + ': degrad_ratio_dict = ', degrad_ratio_dict)
                    print(str(sys._getframe().f_lineno) + ': dram_to_L2_min_dict = ', dram_to_L2_min_dict)
                    print(str(sys._getframe().f_lineno) + ': L2_to_DRAM_min_dict = ', L2_to_DRAM_min_dict)
                    print(str(sys._getframe().f_lineno) + ': eva_nn_chiplet_num_dict = ', eva_nn_chiplet_num_dict)

                
                for nn_name in sp_workload_fitness.keys():
                    total_cycle_dict[nn_name] = {}
                    cycle = 0
                    for layer, item in delay_min_dict[nn_name].items():
                        cycle += item
                        total_cycle_dict[nn_name][layer] = int(cycle)
                        total_cycle_list.append(int(cycle))
                # total_cycle = max(total_cycle_dict.values())
                total_cycle_list.sort()
                if debug_in_BWR:
                    print(str(sys._getframe().f_lineno) + ': total_cycle_dict = ', total_cycle_dict)
                    print(str(sys._getframe().f_lineno) + ': total_cycle_list = ', total_cycle_list)
                #########################################################
                # event
                #########################################################
                event_dict = {}
                old_tick = 0
                index = 0
                while index < len(total_cycle_list):
                    nn_name_list = list(total_cycle_dict.keys())
                    tick = total_cycle_list[index]
                    state = {}
                    for nn_name in total_cycle_dict.keys():
                        layer_list = list(total_cycle_dict[nn_name].keys())
                        now_layer = layer_list[0]
                        for i, layer in enumerate(layer_list):
                            if total_cycle_dict[nn_name][layer] == tick:
                                now_layer = layer
                            elif total_cycle_dict[nn_name][layer] < tick and i < len(layer_list) - 1:
                                now_layer = layer_list[i + 1]
                            else:
                                break
                        state[nn_name] = now_layer
                    event_dict[tick] = state

                    if debug_in_BWR:
                        print(str(sys._getframe().f_lineno) + ': ', tick, state)
                    #########################################################
                    # layout mapping
                    #########################################################
                    degrad_ratio_list = []
                    network_list = []
                    for nn_name in state.keys():
                        degrad_ratio_list = degrad_ratio_list + [degrad_ratio_dict[nn_name][state[nn_name]]] * eva_nn_chiplet_num_dict[nn_name]
                        network_list = network_list + [nn_name] * eva_nn_chiplet_num_dict[nn_name]

                    degrad_ratio_list = np.array(degrad_ratio_list)
                    network_list = np.array(network_list)

                    if self.layout_mapping_method == 'balance':
                        degrad_ratio_list = degrad_ratio_list.reshape([self.cluster_size, self.cluster_num]).T
                        network_list = network_list.reshape([self.cluster_size, self.cluster_num]).T
                    elif self.layout_mapping_method == 'concentrate':
                        degrad_ratio_list = degrad_ratio_list.reshape([self.cluster_num, self.cluster_size])
                        network_list = network_list.reshape([self.cluster_num, self.cluster_size])
                    elif self.layout_mapping_method == 'random':
                        np.random.shuffle(network_list)
                        degrad_ratio_list = []
                        for nn_name in network_list:
                            degrad_ratio_list = degrad_ratio_list + [degrad_ratio_dict[nn_name][state[nn_name]]]
                        
                        degrad_ratio_list = np.array(degrad_ratio_list)
                        network_list = np.array(network_list)
                        degrad_ratio_list = degrad_ratio_list.reshape([self.cluster_num, self.cluster_size])
                        network_list = network_list.reshape([self.cluster_num, self.cluster_size])
                    else:
                        raise NotImplementedError

                    if debug_in_BWR:
                        print('degrad_ratio_list = ', degrad_ratio_list)
                        print('network_list = ', network_list)
                    #########################################################
                    # bandwidth allocation 
                    #########################################################
                    for nn_name in nn_name_list:
                        if debug_in_BWR:
                            print(str(sys._getframe().f_lineno) + ': nn_name = ', nn_name)

                        if degrad_ratio_dict[nn_name][state[nn_name]] > 1:
                            flag = True
                            min_degrade = float('inf')

                            for i in range(self.cluster_num):
                                if nn_name in network_list[i]:
                                    if sum(degrad_ratio_list[i] < 1) >= 1:
                                        # can be shared bandwidth
                                        degrade = sum((1 - degrad_ratio_list[i]) * (degrad_ratio_list[i] < 1))
                                        if degrade < min_degrade:
                                            min_degrade = degrade
                                    else:
                                        # can not be shared bandwidth for this network
                                        flag = False
                                        break
                                else:
                                    continue

                            if flag:
                                #########################################################
                                # improve_ratio
                                #########################################################
                                if degrad_ratio_dict[nn_name][state[nn_name]] - min_degrade < 1:
                                    improve_ratio = (degrad_ratio_dict[nn_name][state[nn_name]] - 1) / degrad_ratio_dict[nn_name][state[nn_name]]
                                else:
                                    improve_ratio = min_degrade / degrad_ratio_dict[nn_name][state[nn_name]]
                                if debug_in_BWR:
                                    print(str(sys._getframe().f_lineno) + ': nn_name = ', nn_name)
                                    print(str(sys._getframe().f_lineno) + ': improve_ratio = ', improve_ratio)
                                    print(str(sys._getframe().f_lineno) + ': tick - old_tick = ', tick - old_tick)
                                    print(str(sys._getframe().f_lineno) + ': int((tick - old_tick) * improve_ratio) = ', int((tick - old_tick) * improve_ratio))
                                #########################################################
                                # update
                                #########################################################
                                for layer in total_cycle_dict[nn_name].keys():
                                    if total_cycle_dict[nn_name][layer] >= tick:
                                        total_cycle_dict[nn_name][layer] -= int((tick - old_tick) * improve_ratio)

                                total_cycle_list = []
                                for nn_name in eva_nn_chiplet_num_dict.keys():
                                    cycle = 0
                                    for layer in total_cycle_dict[nn_name].keys():
                                        total_cycle_list.append(total_cycle_dict[nn_name][layer])
                                total_cycle_list.sort()

                                tick = total_cycle_list[index]
                                if debug_in_BWR:
                                    print('--------------')
                                    print(str(sys._getframe().f_lineno) + ': index = ', index)
                                    print(str(sys._getframe().f_lineno) + ': tick = ', tick)
                                    print(str(sys._getframe().f_lineno) + ': new_total_cycle_dict = ', total_cycle_dict)
                                    print(str(sys._getframe().f_lineno) + ': total_cycle_list = ', total_cycle_list)
                                break

                    old_tick = tick
                    index += 1
                #########################################################
                # output
                #########################################################
                for sp_id, workload_fitness in sp_workload_fitness.items():
                    Nchip = self.Nchip_partition_best[tp_id][sp_id]
                    for workload, fitness in workload_fitness.items():
                        workload_BW_list = self.final_merge_workload_BWNeeded_dict[workload][Nchip]
                        layer_id = list(workload_BW_list.keys())[-1]
                        latency_sp = total_cycle_dict[sp_id][workload + '_' + str(layer_id)]
                        energy_sp = sp_workload_fitness_bw[tp_id][sp_id][workload][2]
                        edp_sp = latency_sp * energy_sp / PE_Frequency
                        sp_workload_fitness_bw[tp_id][sp_id][workload] = [edp_sp, latency_sp, energy_sp]
        
        tp_sp_fitness = {}
        for tp_id, sp_space in sp_workload_fitness_bw.items():
            sp_fitness_dict = {}
            for sp_id, workload_fitness in sp_space.items():
                sp_name, fitness = self.calSP(workload_fitness, sp_id)
                sp_fitness_dict[sp_name] = fitness
            tp_name, tp_fitness = self.calTP(sp_fitness_dict, tp_id)
            tp_sp_fitness[tp_name] = tp_fitness
        schedule_name, total_fitness = self.calSP(tp_sp_fitness, None)
        if debug_in_BWR_simple:
            print('sp_workload_fitness_bw = ', sp_workload_fitness_bw)
            print('schedule_name = ', schedule_name)
            print('total_fitness = ', total_fitness)
        return total_fitness
 
    # -----------------------------------------------
    # evoluation_temporal_spatial: 探索函数，主运行函数
    # -----------------------------------------------
    def evoluation_temporal_spatial(self):
        self.initialize()

        # 1. workload schedule：任务调度方案获取 
        # ---目前方法：随机
        workload_code_list, max_sp_num = self.getWorkloadScheduleCodeList_random(self.space_tp_ratio, self.space_sp_ratio, self.workload_schedule_TH)
        
        self.chiplet_partition_dict = self.getChipletPartionDict(max_sp_num)

        if debug_in_evoluation_temporal_spatial_simple:
            print("DEBUG IN evoluation_temporal_spatial()------------------------------")
            print("---chiplet_partition_dict: ", self.chiplet_partition_dict)
            print("---Start---------------------------------------------")
        
        # 2. 迭代进化：遍历所有的workload调度的方案
        # --- 进行硬件资源分配和评估
        for workload_code in workload_code_list:
            if debug_in_evoluation_temporal_spatial_simple:
                print("-----Start a new workload mapping----------")
                print("-----workload_code = ", workload_code)

            # 2.1: Workload Schedule Method Decode
            # --- Decode and Workload Merge
            tp_sp_space = self.decodeCode(workload_code)
            tp_sp_idealParam = self.calIdealParam(tp_sp_space)
            merge_tp_sp_space = self.mergeWorkload(tp_sp_space)

            if debug_in_evoluation_temporal_spatial_simple:
                print("------tp_sp_space = ", tp_sp_space)
                print("-----merge_tp_sp_space = ", merge_tp_sp_space)
                print("-----merge_workload_dict = ", self.merge_workload_dict)
            
            tp_sp_space = copy.deepcopy(merge_tp_sp_space)

            # 2.2 : Evaluation
            # --- Chiplet Allocate + Method Evaluation
            schedule_name, fitness, Nchip_partition, schedule_name_comm, fitness_comm, Nchip_partition_comm = self.evaluation_tp_sp(tp_sp_space, tp_sp_idealParam)

            # 2.3 : 迭代比较
            ob_id = Optimization_Objective_index[self.Optimization_Objective]
            if self.fitness_best == None or fitness[ob_id] < self.fitness_best[ob_id]:
                self.fitness_best = fitness
                self.schedule_best = schedule_name
                self.tp_sp_space_best = tp_sp_space
                self.Nchip_partition_best = Nchip_partition
                self.merge_workload_dict_best = self.merge_workload_dict
                self.final_merge_workload_fitness_dict = self.merge_workload_fitness_dict
                self.final_merge_workload_NeuNeeded_dict = self.merge_workload_NeuNeeded_dict
                
            if self.fitness_best_comm == None or fitness_comm[ob_id] < self.fitness_best_comm[ob_id]:
                self.fitness_best_comm = fitness_comm
                self.schedule_best_comm = schedule_name_comm
                self.tp_sp_space_best_comm = tp_sp_space
                self.Nchip_partition_best_comm = Nchip_partition_comm
                self.merge_workload_dict_best_comm = self.merge_workload_dict
                self.final_merge_workload_fitness_dict_comm = self.merge_workload_fitness_dict
                self.final_merge_workload_NeuNeeded_dict_comm = self.merge_workload_NeuNeeded_dict
            
            if debug_in_record_fitness_iter:
                print("--------------------------------------------------------")
                print("now_schedule_name: ", schedule_name)
                print("now_fitess: ", fitness)
                print("best_schedule_name: ", self.schedule_best)
                print("best_fitess: ", self.fitness_best)
                print("tp_sp_space_best: ", self.tp_sp_space_best)
                print("merge_workload_dict_best: ", self.merge_workload_dict_best)
                print("final_merge_workload_fitness_dict: ", self.final_merge_workload_fitness_dict)
                        
            self.fitness_record.append(fitness)
            self.schedule_record.append(schedule_name)
            self.tp_sp_space_record.append(tp_sp_space)

            if debug_in_evoluation_temporal_spatial:
                print("-----best_fitess: ", self.fitness_best)
                print("-----best_schedule_name: ", self.schedule_best)
                print("-----best_Nchip_partition: ", self.Nchip_partition_best)
        print("-----wwww best_fitess: ", self.fitness_best)
        mesh_f, cmesh_f, ours_f, mesh_latency_tp_sp, cmesh_latency_tp_sp, ours_latency_tp_sp = self.finalFitness()
        mesh_f_comm, cmesh_f_comm, ours_f_comm, mesh_latency_tp_sp_comm, cmesh_latency_tp_sp_comm, ours_latency_tp_sp_comm = self.finalFitness_comm()
        return  mesh_f, cmesh_f, ours_f, mesh_latency_tp_sp, cmesh_latency_tp_sp, ours_latency_tp_sp, \
            mesh_f_comm, cmesh_f_comm, ours_f_comm, mesh_latency_tp_sp_comm, cmesh_latency_tp_sp_comm, ours_latency_tp_sp_comm
    
    
    def finalFitness(self):
        mesh_latency_tp_sp = {}
        cmesh_latency_tp_sp = {}
        ours_latency_tp_sp = {}
        for tp_id, sp_space in self.tp_sp_space_best.items():
            mesh_latency_tp_sp[tp_id] = {}
            cmesh_latency_tp_sp[tp_id] = {}
            ours_latency_tp_sp[tp_id] = {}
            for sp_id, workloads in sp_space.items():
                mesh_latency_tp_sp[tp_id][sp_id] = [0, 0, 0] # total, noc, nop
                cmesh_latency_tp_sp[tp_id][sp_id] = [0, 0, 0] # total, noc, nop
                ours_latency_tp_sp[tp_id][sp_id] = [0, 0, 0] # total, noc, nop
                Nchip = self.Nchip_partition_best[tp_id][sp_id]
                for workload in workloads:
                    for layer in self.final_merge_workload_fitness_dict[workload][Nchip]:
                        latency = self.final_merge_workload_fitness_dict[workload][Nchip][layer]['L']
                        comm_num = self.final_merge_workload_NeuNeeded_dict[workload][Nchip][layer]
                        ifmapD = comm_num['ifmap_DRAM']
                        weightD = comm_num['weight_DRAM']
                        ofmapD = comm_num['ofmap_DRAM']
                        parallel = comm_num['parallel']
                        commPerf_mesh = comm_performance(ifmapD, weightD, ofmapD, parallel, 'mesh', self.route_table_mesh, self.linkBW_mesh, self.links_mesh)
                        commPerf_cmesh = comm_performance(ifmapD, weightD, ofmapD, parallel, 'cmesh', self.route_table_cmesh, self.linkBW_cmesh, self.links_cmesh)
                        commPerf_ours = comm_performance(ifmapD, weightD, ofmapD, parallel, 'ours', None, None, None)
                        mesh_latency_tp_sp[tp_id][sp_id][0] += latency + commPerf_mesh
                        cmesh_latency_tp_sp[tp_id][sp_id][0] += latency + commPerf_cmesh
                        ours_latency_tp_sp[tp_id][sp_id][0] += latency + commPerf_ours
                        mesh_latency_tp_sp[tp_id][sp_id][1] += latency
                        cmesh_latency_tp_sp[tp_id][sp_id][1] += latency
                        ours_latency_tp_sp[tp_id][sp_id][1] += latency
                        mesh_latency_tp_sp[tp_id][sp_id][2] += commPerf_mesh
                        cmesh_latency_tp_sp[tp_id][sp_id][2] += commPerf_cmesh
                        ours_latency_tp_sp[tp_id][sp_id][2] += commPerf_ours
        mesh_fitness = [0, 0, 0]
        for tp_id, sp_space in self.tp_sp_space_best.items():
            latency_max = 0
            best_fitness = []
            for sp_id in sp_space:
                if mesh_latency_tp_sp[tp_id][sp_id][0] > latency_max:
                    latency_max = mesh_latency_tp_sp[tp_id][sp_id][0]
                    best_fitness = mesh_latency_tp_sp[tp_id][sp_id]
            
            mesh_fitness[0] += best_fitness[0]
            mesh_fitness[1] += best_fitness[1]
            mesh_fitness[2] += best_fitness[2]
        
        cmesh_fitness = [0, 0, 0]
        for tp_id, sp_space in self.tp_sp_space_best.items():
            latency_max = 0
            best_fitness = []
            for sp_id in sp_space:
                if cmesh_latency_tp_sp[tp_id][sp_id][0] > latency_max:
                    latency_max = cmesh_latency_tp_sp[tp_id][sp_id][0]
                    best_fitness = cmesh_latency_tp_sp[tp_id][sp_id]
            
            cmesh_fitness[0] += best_fitness[0]
            cmesh_fitness[1] += best_fitness[1]
            cmesh_fitness[2] += best_fitness[2]
            
        ours_fitness = [0, 0, 0]
        for tp_id, sp_space in self.tp_sp_space_best.items():
            latency_max = 0
            best_fitness = []
            for sp_id in sp_space:
                if ours_latency_tp_sp[tp_id][sp_id][0] > latency_max:
                    latency_max = ours_latency_tp_sp[tp_id][sp_id][0]
                    best_fitness = ours_latency_tp_sp[tp_id][sp_id]
            
            ours_fitness[0] += best_fitness[0]
            ours_fitness[1] += best_fitness[1]
            ours_fitness[2] += best_fitness[2]
        
        return mesh_fitness, cmesh_fitness, ours_fitness, mesh_latency_tp_sp, cmesh_latency_tp_sp, ours_latency_tp_sp
    
    def finalFitness_comm(self):
        mesh_latency_tp_sp = {}
        cmesh_latency_tp_sp = {}
        ours_latency_tp_sp = {}
        for tp_id, sp_space in self.tp_sp_space_best_comm.items():
            mesh_latency_tp_sp[tp_id] = {}
            cmesh_latency_tp_sp[tp_id] = {}
            ours_latency_tp_sp[tp_id] = {}
            for sp_id, workloads in sp_space.items():
                mesh_latency_tp_sp[tp_id][sp_id] = [0, 0, 0] # total, noc, nop
                cmesh_latency_tp_sp[tp_id][sp_id] = [0, 0, 0] # total, noc, nop
                ours_latency_tp_sp[tp_id][sp_id] = [0, 0, 0] # total, noc, nop
                Nchip = self.Nchip_partition_best_comm[tp_id][sp_id]
                for workload in workloads:
                    for layer in self.final_merge_workload_fitness_dict_comm[workload][Nchip]:
                        latency = self.final_merge_workload_fitness_dict_comm[workload][Nchip][layer]['L']
                        comm_num = self.final_merge_workload_NeuNeeded_dict_comm[workload][Nchip][layer]
                        ifmapD = comm_num['ifmap_DRAM']
                        weightD = comm_num['weight_DRAM']
                        ofmapD = comm_num['ofmap_DRAM']
                        parallel = comm_num['parallel']
                        commPerf_mesh = comm_performance(ifmapD, weightD, ofmapD, parallel, 'mesh', self.route_table_mesh, self.linkBW_mesh, self.links_mesh)
                        commPerf_cmesh = comm_performance(ifmapD, weightD, ofmapD, parallel, 'cmesh', self.route_table_cmesh, self.linkBW_cmesh, self.links_cmesh)
                        commPerf_ours = comm_performance(ifmapD, weightD, ofmapD, parallel, 'ours', None, None, None)
                        mesh_latency_tp_sp[tp_id][sp_id][0] += latency + commPerf_mesh
                        cmesh_latency_tp_sp[tp_id][sp_id][0] += latency + commPerf_cmesh
                        ours_latency_tp_sp[tp_id][sp_id][0] += latency + commPerf_ours
                        mesh_latency_tp_sp[tp_id][sp_id][1] += latency
                        cmesh_latency_tp_sp[tp_id][sp_id][1] += latency
                        ours_latency_tp_sp[tp_id][sp_id][1] += latency
                        mesh_latency_tp_sp[tp_id][sp_id][2] += commPerf_mesh
                        cmesh_latency_tp_sp[tp_id][sp_id][2] += commPerf_cmesh
                        ours_latency_tp_sp[tp_id][sp_id][2] += commPerf_ours
        mesh_fitness = [0, 0, 0]
        for tp_id, sp_space in self.tp_sp_space_best_comm.items():
            latency_max = 0
            best_fitness = []
            for sp_id in sp_space:
                if mesh_latency_tp_sp[tp_id][sp_id][0] > latency_max:
                    latency_max = mesh_latency_tp_sp[tp_id][sp_id][0]
                    best_fitness = mesh_latency_tp_sp[tp_id][sp_id]
            
            mesh_fitness[0] += best_fitness[0]
            mesh_fitness[1] += best_fitness[1]
            mesh_fitness[2] += best_fitness[2]
        
        cmesh_fitness = [0, 0, 0]
        for tp_id, sp_space in self.tp_sp_space_best_comm.items():
            latency_max = 0
            best_fitness = []
            for sp_id in sp_space:
                if cmesh_latency_tp_sp[tp_id][sp_id][0] > latency_max:
                    latency_max = cmesh_latency_tp_sp[tp_id][sp_id][0]
                    best_fitness = cmesh_latency_tp_sp[tp_id][sp_id]
            
            cmesh_fitness[0] += best_fitness[0]
            cmesh_fitness[1] += best_fitness[1]
            cmesh_fitness[2] += best_fitness[2]
            
        ours_fitness = [0, 0, 0]
        for tp_id, sp_space in self.tp_sp_space_best_comm.items():
            latency_max = 0
            best_fitness = []
            for sp_id in sp_space:
                if ours_latency_tp_sp[tp_id][sp_id][0] > latency_max:
                    latency_max = ours_latency_tp_sp[tp_id][sp_id][0]
                    best_fitness = ours_latency_tp_sp[tp_id][sp_id]
            
            ours_fitness[0] += best_fitness[0]
            ours_fitness[1] += best_fitness[1]
            ours_fitness[2] += best_fitness[2]
        
        return mesh_fitness, cmesh_fitness, ours_fitness, mesh_latency_tp_sp, cmesh_latency_tp_sp, ours_latency_tp_sp
        
    def nopComm(self):
        for w_name, items in self.merge_workload_fitness_dict.items():
            for c_num, layers in items.items():
                for layer_name, perf in layers.items():
                    comm_num = self.merge_workload_NeuNeeded_dict[w_name][c_num][layer_name]
                    ifmapD = comm_num['ifmap_DRAM']
                    weightD = comm_num['weight_DRAM']
                    ofmapD = comm_num['ofmap_DRAM']
                    parallel = comm_num['parallel']
                    commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, self.topology, self.route_table, self.linkBW, self.links)
                    # commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, self.topology)
                    self.merge_workload_fitness_dict[w_name][c_num][layer_name]['L'] = max(commPerf, perf['L'])
                    if commPerf > perf['L']:
                        print(1)
def plot(nn_name_list, architecture):
    id = 1
    row = len(nn_name_list)
    plt.figure("Fitness per chiplet num")
    for nn_name in nn_name_list:
        plt.subplot(row, 1, id)
        id += 1
        file = "{}/SE_result/{}_{}.txt".format(cur_dir, architecture, nn_name)
        f = open(file)
        lines = f.readlines()
        x = []
        y = []
        for line in lines:
            if line.startswith("chiplet"):
                line_items = line.replace("\n","").split("\t")
                chiplet_num = int(line_items[1])
                fitness = float(line_items[3])
                x.append(chiplet_num)
                y.append(fitness)
        if id > row:
            plt.xlabel("Chiplet Num", fontsize = 10)
        plt.ylabel("Fitness", fontsize = 10)
        plt.bar(x, y, width=0.5,color='rosybrown')
        plt.plot(x,y,color='brown')
        plt.tick_params(labelsize=8)
        for i in range(len(x)):
            plt.scatter(x[i],y[i],s=8,color='brown')
            #xy = (x[i], round(y[i]))
            #plt.annotate("%s" % round(y[i]), xy=xy, xytext=(-20, 10), textcoords='offset points')
        plt.title(nn_name, fontsize = 12, color='brown')
    plt.tight_layout(pad=1.1)
    plt.savefig(cur_dir + "/SE_result/fitness_change_per_Nchiplet_line.png", bbox_inches = 'tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default="ours", help='architecture')
    parser.add_argument('--nn_list_type', type=str, default="vision", help='neural network, using + as split')
    parser.add_argument('--chiplet_num', type=int, default=16, help='Compute Chiplet Num')
    parser.add_argument('--mem_num', type=int, default=4, help='Memory Chiplet Num')
    parser.add_argument('--Optimization_Objective', type=str, default="latency", help='Optimization Objective: edp, energy, latency')
    parser.add_argument('--alg', type=str, default='GA', help='use layer fuse or not')
    parser.add_argument('--encode_type', type=str, default='index', help='encode type')
    parser.add_argument('--topology', type=str, default='mesh', help='topology')
    parser.add_argument('--BW_Reallocator_tag', type=int, default=1, help='use BW Reallocator or not')
    parser.add_argument('--layout_mapping_method', type=str, default='balance', help='balance, concentrate or random')
    parser.add_argument('--tp_TH', type=float, default=1, help='tp ratio max num')
    parser.add_argument('--sp_TH', type=float, default=0.8, help='sp ratio max num')
    parser.add_argument('--partition_ratio', type=float, default=0.4, help='partition ratio')
    
    opt = parser.parse_args()
    architecture = opt.architecture
    chiplet_num = opt.chiplet_num
    mem_num = opt.mem_num
    Optimization_Objective = opt.Optimization_Objective
    
    type = opt.nn_list_type
    if type == 'vision':
        nn_name_list = ['resnet50', 'darknet19', 'VGG16']
    elif type == 'nlp':
        nn_name_list = ['GNMT', 'BERT', 'ncf']
    else:
        nn_name_list = ['BERT', 'GNMT', 'resnet18', 'vit']

    # 读取任务列表  
    # workload_dict = {"resnet18":{"":[1,17]}, "resnet50":{"w1":[1,21], "w2":[22,50]}, "VGG16":{"":[1,13]}}
    abs_path = os.path.dirname(os.path.abspath(__file__))
    SE_abs_path = os.path.join(abs_path, "../nnparser_SE_hetero_iodie")

    # DSE
    MNN_Engine = multi_network_DSE(architecture, chiplet_num, mem_num, nn_name_list, Optimization_Objective, tp_TH=opt.tp_TH, sp_TH=opt.sp_TH, topology=opt.topology, partition_ratio=opt.partition_ratio, BW_tag=opt.BW_Reallocator_tag, layout_mapping_method=opt.layout_mapping_method)
    start_time = datetime.datetime.now()
    mesh_f, cmesh_f, ours_f, mesh_latency_tp_sp, cmesh_latency_tp_sp, ours_latency_tp_sp, \
    mesh_f_comm, cmesh_f_comm, ours_f_comm, mesh_latency_tp_sp_comm, cmesh_latency_tp_sp_comm, ours_latency_tp_sp_comm \
    = MNN_Engine.evoluation_temporal_spatial()
    
    
    #########################################################
    # output
    #########################################################
    #MNN_Engine.getFinalBWFN()
    #total_fitness = MNN_Engine.layout_mapping()
    #print('total_fitness = ', total_fitness)

    # 控制台输出
    print("Sim END----------------------------------------------------------")
    print("best schedule name : ", MNN_Engine.schedule_best)
    print("best Nchip Partition : ", MNN_Engine.Nchip_partition_best)
    print("best fitness : ", MNN_Engine.fitness_best)
    print("")

    # 文本输出
    file_path = cur_dir + "/multi_nn_result/explore_result_total.txt"
    if os.path.exists(file_path) == False:
        result_out_file = open(cur_dir + "/multi_nn_result/explore_result_total.txt", 'w')
    else:
        result_out_file = open(cur_dir + "/multi_nn_result/explore_result_total.txt", 'a')
    end_time = datetime.datetime.now()
    sim_time = end_time - start_time

    print("{:-^120}".format(" SIM TIME "), file = result_out_file)
    print("start_time = {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')), file = result_out_file)
    print("end_time = {}".format(end_time.strftime('%Y-%m-%d %H:%M:%S')), file = result_out_file)
    print("sim_time = {}".format(sim_time), file = result_out_file)

    print("{:-^120}".format(" SETTING "), file = result_out_file)
    print("Chiplet Allocate Add", file = result_out_file)
    print("nn_list = {}".format(nn_name_list), file = result_out_file)
    print("workload = {}".format(MNN_Engine.workload_dict), file = result_out_file)
    print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
    print("architecture = {}".format(architecture), file = result_out_file)
    print("alg = {}".format(opt.alg), file = result_out_file)
    print("encode_type = {}".format(opt.encode_type), file = result_out_file)
    print("BW_Reallocator_tag = {} (0: without BWR;  1: with BWR)".format(opt.BW_Reallocator_tag), file = result_out_file)
    print("tp_TH = {}".format(opt.tp_TH), file = result_out_file)
    print("sp_TH = {}".format(opt.sp_TH), file = result_out_file)

    print("{:-^120}".format(" RESULT "), file = result_out_file)
    print("total_sample_num = {}".format(MNN_Engine.sample_num), file = result_out_file)
    print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
    print("merge_workload = {}".format(MNN_Engine.merge_workload_dict_best), file = result_out_file)
    print("{:-<100}".format("schedule space result "), file = result_out_file)
    for tp_id, sp_space in MNN_Engine.tp_sp_space_best.items():
        line = "\ttp_id({})\t".format(tp_id)
        for sp_id in sp_space:
            sp_item = "sp_id({}): ".format(sp_id)
            for workload_name in sp_space[sp_id]:
                sp_item += workload_name + "+"
            sp_item = sp_item[:-1]
            sp_item += "; " + str(MNN_Engine.Nchip_partition_best[tp_id][sp_id])
            line += "{:30}\t\t".format(sp_item)
        print(line, file = result_out_file)
    print("{:-<100}".format(""), file = result_out_file)
    print("fitness_result: latency({}), energy({})".format(MNN_Engine.fitness_best['L'], MNN_Engine.fitness_best['E']), file = result_out_file)
    print("{:-^120}".format(" topology (total, noc, nop) "), file = result_out_file)
    print("{:-^40}".format(" comm ignore "), file = result_out_file)
    
    print("mesh latency:\t{}\t{}\t{}".format(mesh_f[0], mesh_f[1], mesh_f[2])) 
    print("cmesh latency:\t{}\t{}\t{}".format(cmesh_f[0], cmesh_f[1], cmesh_f[2])) 
    print("ours latency:\t{}\t{}\t{}".format(ours_f[0], ours_f[1], ours_f[2])) 
    print("mesh latency:\t{}\t{}\t{}".format(mesh_f[0], mesh_f[1], mesh_f[2]), file = result_out_file) 
    print("cmesh latency:\t{}\t{}\t{}".format(cmesh_f[0], cmesh_f[1], cmesh_f[2]), file = result_out_file) 
    print("ours latency:\t{}\t{}\t{}".format(ours_f[0], ours_f[1], ours_f[2]), file = result_out_file) 
    print("mesh_latency_tp_sp:\t{}\t".format(mesh_latency_tp_sp), file = result_out_file) 
    print("cmesh_latency_tp_sp:\t{}\t".format(cmesh_latency_tp_sp), file = result_out_file) 
    print("ours_latency_tp_sp:\t{}\t".format(ours_latency_tp_sp), file = result_out_file) 
    print("{:-^40}".format(" comm consideration "), file = result_out_file)
    print("mesh latency:\t{}\t{}\t{}".format(mesh_f_comm[0], mesh_f_comm[1], mesh_f_comm[2])) 
    print("cmesh latency:\t{}\t{}\t{}".format(cmesh_f_comm[0], cmesh_f_comm[1], cmesh_f_comm[2])) 
    print("ours latency:\t{}\t{}\t{}".format(ours_f_comm[0], ours_f_comm[1], ours_f_comm[2])) 
    print("mesh latency:\t{}\t{}\t{}".format(mesh_f_comm[0], mesh_f_comm[1], mesh_f_comm[2]), file = result_out_file) 
    print("cmesh latency:\t{}\t{}\t{}".format(cmesh_f_comm[0], cmesh_f_comm[1], cmesh_f_comm[2]), file = result_out_file) 
    print("ours latency:\t{}\t{}\t{}".format(ours_f_comm[0], ours_f_comm[1], ours_f_comm[2]), file = result_out_file) 
    print("mesh_latency_tp_sp:\t{}\t".format(mesh_latency_tp_sp_comm), file = result_out_file) 
    print("cmesh_latency_tp_sp:\t{}\t".format(cmesh_latency_tp_sp_comm), file = result_out_file) 
    print("ours_latency_tp_sp:\t{}\t".format(ours_latency_tp_sp_comm), file = result_out_file) 
    print("{:-^120}".format(" END "), file = result_out_file)
    print("", file = result_out_file)
    print("", file = result_out_file)
    print("", file = result_out_file)
    result_out_file.close()

    # 文本输出：迭代样本完整记录
    sample_record = False
    if sample_record:
        result_out_file = open(cur_dir + "/multi_nn_result/explore_result_sample_record.txt", 'w')
        print("{:-^120}".format(" SIM TIME "), file = result_out_file)
        print("start_time = {}".format(start_time), file = result_out_file)
        print("end_time = {}".format(end_time), file = result_out_file)
        print("sim_time = {}".format(sim_time), file = result_out_file)
        print("", file = result_out_file)

        print("{:-^120}".format(" SETTING "), file = result_out_file)
        print("nn_list = {}".format(nn_name_list), file = result_out_file)
        print("workload = {}".format(MNN_Engine.workload_dict), file = result_out_file)
        print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
        print("Optimization_Objective = {}".format(Optimization_Objective), file = result_out_file)
        print("BW_Reallocator_tag = {} (0: without BWR;  1: with BWR)".format(opt.BW_Reallocator_tag), file = result_out_file)
        print("", file = result_out_file)

        print("{:-^120}".format(" RESULT "), file = result_out_file)
        print("total_sample_num = {}".format(MNN_Engine.sample_num), file = result_out_file)
        print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
        print("{:-<100}".format("schedule space result "), file = result_out_file)
        for tp_id, sp_space in MNN_Engine.tp_sp_space_best.items():
            line = "\ttp_id({})\t".format(tp_id)
            for sp_id in sp_space:
                sp_item = "sp_id({}): ".format(sp_id)
                for workload_name in sp_space[sp_id]:
                    sp_item += workload_name + "+"
                sp_item = sp_item[:-1]
                sp_item += "; " + str(MNN_Engine.Nchip_partition_best[tp_id][sp_id])
                line += "{:30}\t\t".format(sp_item)
            print(line, file = result_out_file)
        print("{:-<100}".format(""), file = result_out_file)
        print("fitness_result: latency({}), energy({})".format(MNN_Engine.fitness_best['L'], MNN_Engine.fitness_best['E'], MNN_Engine.fitness_best[2]), file = result_out_file)
        print("", file = result_out_file)

        print("{:-^120}".format(" SAMPLE RECORD "), file = result_out_file)
        print("", file = result_out_file)
        print("{:-^120}".format(" END "), file = result_out_file)

        id = 0
        for sample in MNN_Engine.schedule_record:
            fitness = str(MNN_Engine.fitness_record[id])
            Nchip = str(MNN_Engine.Nchip_par_record[id])
            id += 1
            print("sample={:100}\t, ---Nchip_par:{:60}\t, ---fitness:\t{:15}\t".format(str(sample), Nchip, str(fitness)), file = result_out_file)

    #MNN_Engine.plot()