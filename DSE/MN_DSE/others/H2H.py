from simplified_multi_network_DSE import *

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
                commPerf_mesh = comm_performance(ifmapD, weightD, ofmapD, parallel, 'mesh', MNN_Engine.route_table, MNN_Engine.linkBW, MNN_Engine.links)
                fitness[workload][c_num]['t'] += perf['L'] + commPerf_mesh
                fitness[workload][c_num]['noc'] += perf['L']
                fitness[workload][c_num]['nop'] += commPerf_mesh
    
    return fitness, neu_Needed

def getAccIndex(workload_fitness_dict, workload, nn_name_list, sub_acc):
    noc_l = 0
    nop_l = 0
    for index, nn in enumerate(nn_name_list):
        index = index % len(sub_acc)
        if nn in workload:
            latency = workload_fitness_dict[workload][sub_acc[index]]['t']
            print(workload, index, sub_acc[index])
            noc_l = workload_fitness_dict[workload][sub_acc[index]]['noc']
            nop_l = workload_fitness_dict[workload][sub_acc[index]]['nop']
            return index, latency, noc_l, nop_l

def h2h_scheduling(MNN_Engine, workload_fitness_dict, nn_name_list):

    min_total_latency = float('inf')
    min_acc = None
    min_noc = None
    min_nop = None

    chiplet_partition_dict = MNN_Engine.getChipletPartionDict(chiplet_num)
    chiplet_partition_dict.pop(1)
    print('chiplet_partition_dict = ', chiplet_partition_dict)

    for sub in chiplet_partition_dict.values():
        for sub_acc in sub:
            sub_acc.sort(reverse = True)

            workload_list = []
            for i in range(len(sub_acc)):
                workload_list.append([])

            total_latency_list = [0] * len(sub_acc)
            noc_latency_list = [0] * len(sub_acc)
            nop_latency_list = [0] * len(sub_acc)
            for workload in workload_fitness_dict.keys():
                index, latency, noc_l, nop_l = getAccIndex(workload_fitness_dict, workload, nn_name_list, sub_acc)
                total_latency_list[index] = total_latency_list[index] + latency
                noc_latency_list[index] += noc_l
                nop_latency_list[index] += nop_l
                workload_list[index].append(workload)

            if max(total_latency_list) < min_total_latency:
                min_total_latency = max(total_latency_list)
                min_acc = sub_acc
                min_noc = noc_latency_list
                min_nop = nop_latency_list
                min_workload_list = workload_list

            print(sub_acc, workload_list, total_latency_list)

    return min_total_latency, min_acc, min_workload_list, min_noc, min_nop

if __name__ == '__main__':
    architecture = 'simba'
    chiplet_num = 16
    mem_num = 4
    Optimization_Objective = 'latency'
    tp_TH = 4
    sp_TH = 4

    workload_dict = ['resnet18', 'darknet19', 'vit', 'Unet']
    # workload_dict = ['GNMT', 'BERT', 'ncf']
    # workload_dict = ['BERT', 'GNMT', 'resnet18', 'VGG16']


    MNN_Engine = multi_network_DSE(architecture, chiplet_num, mem_num, workload_dict, Optimization_Objective, tp_TH=0.2, sp_TH=1,  topology='mesh',partition_ratio=1, BW_tag=False, layout_mapping_method=None)
    MNN_Engine.initialize()
    MNN_Engine.setTotalWorkloadFitness()
    
    fitness, neuNeeded = merge_workload(MNN_Engine)

    total_latency, sub_acc, workload_list, min_noc, min_nop = h2h_scheduling(MNN_Engine, fitness, workload_dict)


    print('total_latency = ', total_latency)
    print('sub_acc = ', sub_acc)
    print('workload_list = ', workload_list)
    
    line = '------------- {} -----------\n'.format('H2H')
    line += 'nn_name : {}\n'.format(workload_dict)
    line += 'total_latency :\t{}\t\n'.format(total_latency)
    line += 'total_noc :\t{}\t\n'.format(min_noc)
    line += 'total_nop :\t{}\t\n'.format(min_nop)
    line += 'workload_list :\t{}\t\n'.format(workload_list)
    line += 'sub_acc :\t{}\t\n'.format(sub_acc)
    line += '\n'
    
    file_n = './H2H.txt'
    if os.path.exists(file_n) == False:
        f = open(file_n, 'w')
    else:
        f = open(file_n, 'a')
    
    print(line, file=f)
    f.close()
