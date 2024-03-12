import math
import os
from platform import architecture
import sys
import random
import copy
import argparse
from matplotlib import pyplot as plt
import shutil
import openpyxl

# Parameter
chiplet_parallel_list = ["P_stable", "PK_stable", "K_stable"]
PE_Frequency = 1000 * 1000 * 1000
ratio = 1 / 1024 # 当Chiplet变多时需要需要一些额外的开销，通过这个ratio来反映

# degub signal
debug_in_workload_extract = 0
debug_in_layer_fuse = 0
debug_in_main = 1
debug_in_txt_extract = 0
debug_in_workload_partition = 0

# getLayerParam: 根据优化目标objective，输出对应的层的理论评估结果
def getLayerParam(app_name, objective):
    SE_path_dir = "/home/wangxy/workspace/chiplet/wxy_chiplet/DSE/SE_DSE/nn_file"
    layer_num = 0
    layer_dict = {}
    layer_id_list = []
    layer_name_list = []
    f = open(SE_path_dir + "/" + app_name + ".txt")

    lines = f.readlines()
    for line in lines:
        if line.startswith("#") or line.startswith("*"):
            pass
        elif line != "\n":
            line_item = line.split(" ")
            layer_num += 1
            H = int(line_item[1])
            M = int(line_item[2])
            C = int(line_item[3])
            R = int(line_item[4])
            S = int(line_item[4])
            stride = int(line_item[5])
            padding = int(line_item[6])
            K = int(line_item[7])
            P = int(line_item[8])
            Q = int(line_item[9])

            iact_num = H * M * C
            oact_num = P * Q * K
            weight_num = R * S * K * C
            compute_num = P * Q * K * R * S * C
            iparam_num = iact_num + weight_num
            param_num = iparam_num + oact_num

            iparam_vs_compute = iparam_num / compute_num
            param_vs_compute = param_num / compute_num

            iparam_mux_compute = iparam_num * compute_num
            param_mux_compute = param_num * compute_num

            layer_id_list.append(layer_num)
            layer_name = "layer" + str(layer_num)
            layer_name_list.append(layer_name)
            layer_dict[layer_name] = {"iact_num":iact_num, "oact_num":oact_num, "weight_num":weight_num, \
                                    "compute_num":compute_num, "iparam_num":iparam_num, "param_num":param_num, \
                                    "iparam_vs_compute":iparam_vs_compute, "param_vs_compute":param_vs_compute, \
                                    "iparam_mux_compute":iparam_mux_compute, "param_mux_compute":param_mux_compute}
    f.close()

    if objective == "noPar" or objective == "totalPar":
        return layer_id_list, layer_name_list

    result_dict = {}
    for layer_name, result_items in layer_dict.items():
        result = result_items[objective]
        result_dict[layer_name] = result
    return result_dict

# partition: 获得评估目标下的最优划分方案
# --- 方案迭代方法：随机，partition_num为方案数目
# --- objective_tag：评估目标（min、max、balance），todo：min和max的考量没有想的很清楚
# --- method_best：输出为划分点的层id，相当于每个workload的第一层的id
def partition(fitness_list, partition_num, objective_tag="balance"):
    def partition_max(list, partition_num):
        list_order = sorted(list, key = lambda x: x, reverse=True)
        TH = list_order[partition_num-1]
        select_index_list = []
        TH_fitness_list = []
        rest_partiton_num = partition_num
        for index, fitness in enumerate(list):
            if fitness > TH:
                select_index_list.append(index)
                rest_partiton_num -= 1
            elif fitness == TH:
                TH_fitness_list.append(index)
            else:
                pass
        
        TH_num = len(TH_fitness_list)
        assert(TH_num >= rest_partiton_num)
        TH_list = list(range(TH_num))
        random.shuffle(TH_list)
        for i in range(rest_partiton_num):
            select_index_list.append(TH_list[i])    
        select_index_list.sort()
        return select_index_list
    
    def partition_min(list, partition_num):
        list_order = sorted(list, key = lambda x: x, reverse=False)
        TH = list_order[partition_num-1]
        select_index_list = []
        TH_fitness_list = []
        rest_partiton_num = partition_num
        for index, fitness in enumerate(list):
            if fitness < TH:
                select_index_list.append(index)
                rest_partiton_num -= 1
            elif fitness == TH:
                TH_fitness_list.append(index)
            else:
                pass
        
        TH_num = len(TH_fitness_list)
        assert(TH_num >= rest_partiton_num)
        TH_list = list(range(TH_num))
        random.shuffle(TH_list)
        for i in range(rest_partiton_num):
            select_index_list.append(TH_list[i])
        select_index_list.sort()
        return select_index_list
        
    def partition_balance(fitness_list, partition_num, method_num):
        # cal_unbalance: 评估方案的不平衡度，不平衡度越高，方案越不好
        def cal_unbalance(list):
            list.sort()
            min = list[0]
            max = list[-1]
            unbalance = 1 - min/max
            return unbalance

        # get_method: 获取方案，随机获取
        def get_method(layer_num, partition_num, method_num):
            index_list = list(range(layer_num))
            method_list = []
            num = 0
            unchange_times = 0
            while num < method_num:
                random.shuffle(index_list)
                select_index_list = index_list[0: partition_num-1]
                assert(len(select_index_list) == partition_num-1)
                select_index_list.sort()
                if select_index_list not in method_list:
                    method_list.append(select_index_list)
                    num += 1
                    unchange_times = 0
                else:
                    unchange_times += 1
                
                if unchange_times > 300:
                    break
            return method_list
        layer_num = len(fitness_list)
        
        if partition_num == layer_num:
            return list(range(partition_num))
        merge_fitness_list = [fitness_list[0]]
        merge_ids = [0]
        for i, f in enumerate(fitness_list):
            if f == merge_fitness_list[-1]:
                merge_ids[-1] = i
            else:
                merge_fitness_list.append(f)
                merge_ids.append(i)
        
        if len(merge_fitness_list) <= partition_num:
            return merge_ids
        
        method_list = get_method(layer_num, partition_num, method_num)
        unbalance_best = None
        method_best = None
        for method in method_list:
            unbalance = 0
            index_pre = 0
            for index in method:
                unbalance += cal_unbalance(merge_fitness_list[index_pre: index+1])
            index_pre = index
            if unbalance_best == None or unbalance < unbalance_best:
                unbalance_best = unbalance
                method_best = method
        method = []
        for i in method_best:
            method.append(merge_ids[i])
        
        return method

    if objective_tag == "balance":
        method_num = 1000
        select_index_list = partition_balance(fitness_list, partition_num, method_num)
    elif objective_tag == "max":
        select_index_list = partition_max(fitness_list, partition_num)

    elif objective_tag == "min":
        select_index_list = partition_min(fitness_list, partition_num)

    return select_index_list

# getWorkloadPartition: 任务划分
# --- objective: "noPar"(以模型为粒度)、"totalPar"(以层为粒度)、"iact_num"、"oact_num"...(省略的详见代码75行layer_dict内容)
def getWorkloadPartition(app_name, objective, partition_ratio):
    if objective == "noPar":
        workload_dict = {}
        layer_id_list, layer_name_list = getLayerParam(app_name, objective)
        workload_dict[app_name+"w0"] = {"layer_id_list":layer_id_list, "layer_name_list":layer_name_list, "workload_name": "w0"}
        return workload_dict
    elif objective == "totalPar":
        workload_dict = {}
        layer_id_list, layer_name_list = getLayerParam(app_name, objective)
        for id in range(len(layer_id_list)):
            w_name = app_name + "w" + str(id)
            layer_id = layer_id_list[id]
            layer_name = layer_name_list[id]
            workload_name = "w" + str(id)
            workload_dict[w_name] = {"layer_id_list": [layer_id], "layer_name_list": [layer_name], "workload_name": workload_name}
        return workload_dict
    
    # 1. 获得各层的优化对象objective的理论评估结果
    result_dict = getLayerParam(app_name, objective)
    result_list = list(result_dict.values())
    partition_num =  math.ceil(partition_ratio * len(result_list))
    if debug_in_workload_partition == 1:
        print("DEBUG IN WORKLOAD PARTITION")
        print("---objective:{}, partition_num:{}".format(objective, partition_num))
        print("---result_dict: ", result_dict)

    # 2. 根据理论评估结果，进行任务划分探索
    # --- select_index_list: 划分点的层id
    select_index_list = partition(result_list, partition_num, objective_tag="balance")

    # 3. 根据划分点，进行任务列表的整合
    workload_dict = {}
    workload_id = 0
    layer_index = 0
    for layer_name, result in result_dict.items():
        if layer_index != 0 and layer_index-1 in select_index_list:
            workload_id += 1
        else:
            pass
        workload_name = app_name + "w" + str(workload_id)
        if workload_name not in workload_dict:
            workload_dict[workload_name] = {"layer_id_list":[], "layer_name_list":[], "workload_name": "w{}".format(workload_id)}
        
        layer_id = layer_index + 1
        workload_dict[workload_name]["layer_id_list"].append(layer_id)
        workload_dict[workload_name]["layer_name_list"].append(layer_name)
        assert(layer_name == "layer"+str(layer_id))
        layer_index += 1
    
    if debug_in_workload_partition == 1:
        print("---workload_dict: ", workload_dict)
    return workload_dict

# getLayerList：获得每一层的对应的单网络仿真的layer id
def getLayerList(layer_name_dict_line):
    #layer_name_dict:  {'layer1': 'layer1', 'layer2': 'layer2'
    line = layer_name_dict_line.replace("layer_name_dict:  ", "")
    line = line.replace("{","")
    line = line.replace("}","")
    line = line.replace("\n","")
    line = line.replace("\'","")
    line = line.replace(":","")
    line = line.replace(", ",",")
    line_list = line.split(",")
    layer_dict = {}
    for item in line_list:
        if item.startswith("layer"):
            item_list = item.split(" ")
            layer_init = item_list[0]
            layer_real = item_list[1]
            layer_dict[layer_init] = layer_real
    return layer_dict

def fitness_line_parse(line, layer_name_dict):
    line = (line.split('{'))[1]
    line = line.replace('{', '')
    line = line.replace('}', '')
    line = line.replace('\n', '')
    line = line.replace('\'', '')
    line = line.replace(' ', '')
    items = line.split(',')
    fitnesses = {}
    for item in items:
        details = item.split(':')
        layer_name = details[0]
        fitness = float(details[1])
        fitnesses[layer_name] = fitness
    return fitnesses

def extand_fitnesses(fitnesses, layer_name_dict):
    fitness_o = {}
    for layer_name, refer_name in layer_name_dict.items():
        fitness_o[layer_name] = fitnesses[refer_name]
    return fitness_o

# extract_fitness: 提取每一层的参数，并进行扩展
def extract_fitness(architecture, nn_name, objective):
    obj = 'latency'
    dir = "/home/wangxy/workspace/chiplet/wxy_chiplet/DSE/SE_DSE/result/{}_{}".format(architecture, nn_name)
    fitnesses_o = {}
    unique_fitnesses_o = {}
    for c_num in range(1, 17, 1):
        file1_latency = "{}/{}/{}_final_result_record.txt".format(dir, obj, c_num)
        file1_energy = "{}/energy/{}_final_result_record.txt".format(dir, c_num)
        file1_edp = "{}/edp/{}_final_result_record.txt".format(dir, c_num)
        
        f1 = open(file1_latency, 'r')
        lines = f1.readlines()
        f1.close()
        layer_name_dict = getLayerList(lines[4])
        latency = fitness_line_parse(lines[2], layer_name_dict)
        
        f1 = open(file1_latency, 'r')
        lines = f1.readlines()
        f1.close()
        energy = fitness_line_parse(lines[1], layer_name_dict)
        
        f1 = open(file1_latency, 'r')
        lines = f1.readlines()
        f1.close()
        edp = fitness_line_parse(lines[0], layer_name_dict)
        
        file2 = "{}/{}/{}_pkt_neu_needed.txt".format(dir, obj, c_num)
        f2 = open(file2, 'r')
        lines = f2.readlines()
        f2.close()
        neu_needed = {}
        chiplet_parallel = {}
        for line in lines:
            if line.startswith("workload"):
                items = line.split(' ')
                layer_name = items[-1].replace('\n', '')
            elif line.startswith('pkt_needed'):
                line = line.replace("pkt_needed:  {", '')
                line = line.replace("}", '')
                line = line.replace(" ", '')
                line = line.replace("\n", '')
                ii = line.split('\'chiplet_para')
                items = ii[0].split(',')
                things = {}
                for item in items[0:-1]:
                    list = item.split(':')
                    key = list[0].replace('\'', '')
                    value = int(float(list[1]))
                    things[key] = value
                line1 = ii[1].split(':')[-1]
                line1 = line1.split('[')[-1]
                line1 = line1.split(']')[0]
                items = line1.split(',')
                chiplet_parallel[layer_name] = [int(i) for i in items]
                    
                neu_needed[layer_name] = {'ifmap_DRAM': 0, 'weight_DRAM': 0, 'ofmap_DRAM': 0, 'ofmap_store_onchip_tag': 0}
                neu_needed[layer_name]["ifmap_DRAM"] = things['input_DRAM']*32
                neu_needed[layer_name]["weight_DRAM"] = things['weight_DRAM']*32
                if things['output_wr'] == 0:
                    neu_needed[layer_name]["ofmap_DRAM"] = things['output_wr_L1'] * 16*32
                    neu_needed[layer_name]["ofmap_store_onchip_tag"] = 1
                else:
                    neu_needed[layer_name]["ofmap_DRAM"] = things['output_wr']*32
                
                assert(things['output_rd'] == 0)
                assert(things['output_rd_L1'] == 0)
        TH = 100
        fitnesses= {}
        # print("neu_needed: ", neu_needed)
        for layer_name in energy.keys():
            fitnesses[layer_name] = {}
            fitnesses[layer_name]["energy"]     = energy[layer_name]
            fitnesses[layer_name]["latency"]    = latency[layer_name]*(1+c_num*c_num*ratio) / 2 # 2GHz
            fitnesses[layer_name]["edp"]        = edp[layer_name]
            fitnesses[layer_name]["neu_needed"]         = neu_needed[layer_name]
            fitnesses[layer_name]["chiplet_parallel"]   = chiplet_parallel[layer_name]
        
        unique_fitnesses_o[c_num] = fitnesses
        fitnesses_o[c_num] = extand_fitnesses(fitnesses, layer_name_dict)
    return fitnesses_o, unique_fitnesses_o

def fitness_merge(fitness_1, fitness_2):
    fitness_o = fitness_1
    fitness_o["energy"] += fitness_2["energy"]
    fitness_o["latency"] += fitness_2["latency"]
    fitness_o["edp"] += fitness_2["edp"]
    fitness_o["neu_needed"]["ofmap_DRAM"] = fitness_2["neu_needed"]["ofmap_DRAM"]
    fitness_o["neu_needed"]["ofmap_store_onchip_tag"] = fitness_2["neu_needed"]["ofmap_store_onchip_tag"]
    return fitness_o

# merge_layer: 融合一些可以直接合并的层
def merge_layer(workload_dict, fitnesses):
    workload_fitnesses = {}
    for w_name, items in workload_dict.items():
        workload_fitnesses[w_name] = {}
        nn_name = w_name.split('w')[0]
        layer_names = items["layer_name_list"]
        for c_num in fitnesses[nn_name]:
            workload_fitnesses[w_name][c_num] = {}
            for i in range(len(layer_names)):
                layer_name = layer_names[i]
                if i == 0:
                    workload_fitnesses[w_name][c_num][layer_name] = fitnesses[nn_name][c_num][layer_name]
                else:
                    if workload_fitnesses[w_name][c_num][name_pre]["neu_needed"]["ofmap_store_onchip_tag"] == 1 and workload_fitnesses[w_name][c_num][name_pre]["chiplet_parallel"] == fitnesses[nn_name][c_num][layer_name]["chiplet_parallel"]:
                        f_o = fitness_merge(workload_fitnesses[w_name][c_num].pop(name_pre), fitnesses[nn_name][c_num][layer_name])
                        layer_name = '{}_{}'.format(name_pre,layer_name)
                        workload_fitnesses[w_name][c_num][layer_name] = f_o
                    else:
                        workload_fitnesses[w_name][c_num][layer_name] = fitnesses[nn_name][c_num][layer_name]
                name_pre = layer_name    
    return workload_fitnesses

def fitness_plot(fitnesses, architecture="simba"):
    energy_dict = {}
    latency_dict = {}
    edp_dict = {}
    layer_name_dict = {}
    x = [i for i in range(4, 17, 4)]
    for nn_name, items in fitnesses.items():
        energy_dict[nn_name]        = {}
        latency_dict[nn_name]       = {}
        edp_dict[nn_name]       = {}
        layer_name_dict[nn_name]    = []
        for c_num, item in items.items():
            for layer_name, fitness in item.items():
                if layer_name not in energy_dict[nn_name]:
                    energy_dict[nn_name][layer_name] = []
                    latency_dict[nn_name][layer_name] = []
                    edp_dict[nn_name][layer_name] = []
                    layer_name_dict[nn_name].append(layer_name)
                energy_dict[nn_name][layer_name].append(fitness["energy"] * (1+ c_num * c_num / 1024))
                latency_dict[nn_name][layer_name].append(fitness["latency"] * c_num / 16)
                edp_dict[nn_name][layer_name].append(fitness["edp"] * c_num / 16 * (1+ c_num * c_num / 1024))
    
    nn_num = len(fitnesses)    
    plt.figure("Fitness Result ", figsize=(12, 4))

    i = 0
    for nn_name in energy_dict:
        layers = layer_name_dict[nn_name]
        plt.subplot(nn_num, 3, i*3+1)
        plt.title("{} Energy Result".format(nn_name), fontsize=10)
        plt.xlabel("chiplet_num", fontsize=9)
        plt.ylabel("energy", fontsize=9)
        for layer in layers:
            plt.plot(x,energy_dict[nn_name][layer], marker='o', markersize=5)
        #plt.legend(layers)
        
        plt.subplot(nn_num, 3, i*3+2)
        plt.title("{} Latency Result".format(nn_name), fontsize=10)
        plt.xlabel("chiplet_num", fontsize=9)
        plt.ylabel("latency", fontsize=9)
        for layer in layers:
            plt.plot(x,latency_dict[nn_name][layer], marker='x', markersize=5)
        #plt.legend(layers)
        
        plt.subplot(nn_num, 3, i*3+3)
        plt.title("{} EDP Result".format(nn_name), fontsize=10)
        plt.xlabel("chiplet_num", fontsize=9)
        plt.ylabel("EDP", fontsize=9)
        for layer in layers:
            plt.plot(x,edp_dict[nn_name][layer], marker='*', markersize=5)
        #plt.legend(layers)
        plt.legend(layers, bbox_to_anchor=(1.1,0.77))
        i += 1
        
    plt.tight_layout(pad=1.1)
    plt_file = "./result.png"
    plt.savefig(plt_file, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    architecture = 'simba'
    nn_names = ['vit']
    fitnesses = {}
    unique_fitnesses = {}
    workload_fitnesses = {}
    for nn_name in nn_names:
        print("------------------ {} ------------------".format(nn_name))
        workload_dict = getWorkloadPartition(nn_name, 'param_mux_compute', 0.4)
        print("workload_dict: ", workload_dict)
        fitnesses[nn_name], unique_fitnesses[nn_name] = extract_fitness(architecture, nn_name, 'latency')
        print("fitness: ", fitnesses[nn_name])
        workload_fitnesses[nn_name] = merge_layer(workload_dict, fitnesses)
        print("workload_fitnesses: ", workload_fitnesses[nn_name])
        print("------------------ {} ------------------".format('end'))
        print("")
    
    fitness_plot(unique_fitnesses)
    
 