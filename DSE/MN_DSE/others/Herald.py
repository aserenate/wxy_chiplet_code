from simplified_multi_network_DSE import *
arch_tag = "ours"

def merge_workload(MNN_Engine):
    fitness = {}
    neu_Needed = {}
    for workload, cc in MNN_Engine.workload_fitness_dict.items():
        fitness[workload] = {}
        neu_Needed[workload] = {}
        for c_num, ccc in cc.items():
            fitness[workload][c_num] = {'t':0, 'noc':0, 'nop':0}
            neu_Needed[workload][c_num] = {'ifmap_DRAM': 0, 'weight_DRAM': 0, 'ofmap_DRAM': 0}
            for layer, perf in ccc.items():
                comm_num = MNN_Engine.workload_NeuNeeded_dict[workload][c_num][layer]
            
                neu_Needed[workload][c_num]['ifmap_DRAM'] += comm_num['ifmap_DRAM']
                neu_Needed[workload][c_num]['weight_DRAM'] += comm_num['weight_DRAM']
                neu_Needed[workload][c_num]['ofmap_DRAM'] += comm_num['ofmap_DRAM']
                neu_Needed[workload][c_num]['parallel'] = comm_num['parallel']
                ifmapD = comm_num['ifmap_DRAM']
                weightD = comm_num['weight_DRAM']
                ofmapD = comm_num['ofmap_DRAM']
                parallel = comm_num['parallel']
                if arch_tag == "ours":
                    commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, 'ours', None, None, None)
                else:
                    commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, arch_tag, MNN_Engine.route_table, MNN_Engine.linkBW, MNN_Engine.links)
                fitness[workload][c_num]['t'] += perf['L'] + commPerf
                fitness[workload][c_num]['noc'] += perf['L']
                fitness[workload][c_num]['nop'] += commPerf
    return fitness, neu_Needed
    
def getAccIndex(workload_fitness_dict, workload, sub_acc):
    min_latency = float('inf')
    min_index = 0
    noc_latency = 0
    nop_latency = 0

    for index, sub in enumerate(sub_acc):
        latency = workload_fitness_dict[workload][sub]['t']
        if latency < min_latency:
            min_index = index
            min_latency = latency
            noc_latency = workload_fitness_dict[workload][sub]['noc']
            nop_latency = workload_fitness_dict[workload][sub]['nop']
    return min_index, min_latency, noc_latency, nop_latency

def argmin_list(lst):
    return min(range(len(lst)), key=lst.__getitem__)

def herald_scheduling(MNN_Engine, workload_fitness_dict, LbF = 3):
    
    min_total_latency = float('inf')
    min_acc = None
    total_nop = None
    total_stall = None
    total_noc = None

    chiplet_partition_dict = MNN_Engine.getChipletPartionDict(chiplet_num)
    chiplet_partition_dict.pop(1)

    for sub in chiplet_partition_dict.values():
        for sub_acc in sub:
            # sub_acc = [2, 1, 1]
            total_latency_list = [0] * len(sub_acc)
            noc_latency_list = [0] * len(sub_acc)
            nop_latency_list = [0] * len(sub_acc)
            stall_latency_list = [0] * len(sub_acc)

            workload_list = []
            for i in range(len(sub_acc)):
                workload_list.append([])

            # iterate over the DNN models in the order specified by workload_fitness_dict
            for nn in workload_fitness_dict.keys():
                last_index = None

                for workload in workload_fitness_dict[nn].keys():
                    index, latency, noc_l, nop_l = getAccIndex(workload_fitness_dict[nn], workload, sub_acc)
                    
                    # Users can specify the maximum allowed load-unbalancing factor, the largest latency across subaccelerators divided by the smallest one. Our scheduler detects an unbalanced load based on the factor.
                    load_balance_cond = ((max(total_latency_list) / (min(total_latency_list) + 0.001)) <= LbF)

                    if load_balance_cond:
                        total_latency_list[index] = total_latency_list[index] + latency
                        if last_index != None and index != last_index:
                            print(last_index, index)
                            stall_latency_list[index] += (max(total_latency_list[last_index], total_latency_list[index]) - total_latency_list[index])
                            total_latency_list[index] = max(total_latency_list[last_index], total_latency_list[index]) + latency
                        else:
                            total_latency_list[index] = total_latency_list[index] + latency
                    else:
                        index = argmin_list(total_latency_list)
                        latency = workload_fitness_dict[nn][workload][sub_acc[index]]['t']
                        total_latency_list[index] = total_latency_list[index] + latency
                        if last_index != None and index != last_index:
                            stall_latency_list[index] += (max(total_latency_list[last_index], total_latency_list[index]) - total_latency_list[index])
                            total_latency_list[index] = max(total_latency_list[last_index], total_latency_list[index]) + latency
                        else:
                            total_latency_list[index] = total_latency_list[index] + latency
                    
                    noc_latency_list[index] += noc_l
                    nop_latency_list[index] += nop_l
                    
                    workload_list[index].append(workload)
                    last_index = index

                    print(sub_acc, index, workload, latency, total_latency_list)

            if max(total_latency_list) < min_total_latency:
                min_total_latency = max(total_latency_list)
                min_acc = sub_acc
                min_workload_list = workload_list
                total_nop = nop_latency_list
                total_stall = stall_latency_list
                total_noc = noc_latency_list
        
    return min_total_latency, min_acc, min_workload_list, total_nop, total_noc, total_stall

if __name__ == '__main__':
    architecture = 'simba'
    chiplet_num = 16
    mem_num = 4
    Optimization_Objective = 'latency'
    tp_TH = 4
    sp_TH = 4

    type = 'mix'
    if type == 'vision':
        workload_dict = ['resnet50', 'darknet19', 'VGG16']
    elif type == 'nlp':
        workload_dict = ['GNMT', 'BERT', 'ncf']
    else:
        workload_dict = ['BERT', 'GNMT', 'resnet18', 'vit']

    MNN_Engine = multi_network_DSE(architecture, chiplet_num, mem_num, workload_dict, Optimization_Objective, tp_TH=0.2, sp_TH=1,  topology='mesh',partition_ratio=1, BW_tag=False, layout_mapping_method=None)
    MNN_Engine.initialize()
    MNN_Engine.setTotalWorkloadFitness()
    
    fitness, neuNeeded = merge_workload(MNN_Engine)

    workload_fitness_dict = {}
    workload_neuNeeded_dict = {}
    for nn in workload_dict:
        workload_fitness_dict[nn] = {}
        workload_neuNeeded_dict[nn] = {}
    
    for key in fitness.keys():
        for nn in workload_dict:
            if nn in key:
                workload_fitness_dict[nn][key] = fitness[key]
                workload_neuNeeded_dict[nn][key] = neuNeeded[key]
                break

    total_latency, sub_acc, workload_list, total_nop, total_noc, total_stall = herald_scheduling(MNN_Engine, workload_fitness_dict)

    print('total_latency = ', total_latency)
    print('sub_acc = ', sub_acc)
    print('workload_list = ', workload_list)
    
    line = '------------- {} -----------\n'.format('Herald')
    line += 'architecture : {}\n'.format(architecture)
    line += 'nn_name : {}\n'.format(workload_dict)
    line += 'interconnection : {}\n'.format(arch_tag)
    line += 'total_latency :\t{}\t\n'.format(total_latency)
    line += 'total_nop :\t{}\t\n'.format(total_nop)
    line += 'total_noc :\t{}\t\n'.format(total_noc)
    line += 'total_stall :\t{}\t\n'.format(total_stall)
    line += 'workload_list :\t{}\t\n'.format(workload_list)
    line += 'sub_acc :\t{}\t\n'.format(sub_acc)
    line += '\n'
    
    file_n = './Herald.txt'
    if os.path.exists(file_n) == False:
        f = open(file_n, 'w')
    else:
        f = open(file_n, 'a')
    
    print(line, file=f)
    f.close()
