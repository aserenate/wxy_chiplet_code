from simplified_multi_network_DSE import *
import os

arch_tag = 'ours'
def merge_workload(MNN_Engine, nn_name_list):
    fitness = {}
    neu_Needed = {}
    for workload, cc in MNN_Engine.workload_fitness_dict.items():
        for nn in nn_name_list:
            if nn in workload:
                break
            nn = None
        if nn == None:
            exit()
        if nn not in fitness:
            fitness[nn] = {}
            neu_Needed[nn] = {}
        for c_num, ccc in cc.items():
            if c_num not in fitness[nn]:
                fitness[nn][c_num] = {'t':0, 'noc':0, 'nop':0}
                neu_Needed[nn][c_num] = {'ifmap_DRAM': 0, 'weight_DRAM': 0, 'ofmap_DRAM': 0}
            for layer, perf in ccc.items():
                comm_num = MNN_Engine.workload_NeuNeeded_dict[workload][c_num][layer]
            
                neu_Needed[nn][c_num]['ifmap_DRAM'] += comm_num['ifmap_DRAM']
                neu_Needed[nn][c_num]['weight_DRAM'] += comm_num['weight_DRAM']
                neu_Needed[nn][c_num]['ofmap_DRAM'] += comm_num['ofmap_DRAM']
                neu_Needed[nn][c_num]['parallel'] = comm_num['parallel']
                ifmapD = comm_num['ifmap_DRAM']
                weightD = comm_num['weight_DRAM']
                ofmapD = comm_num['ofmap_DRAM']
                parallel = comm_num['parallel']
                if arch_tag == "ours":
                    commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, 'ours', None, None, None)
                else:
                    commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, arch_tag, MNN_Engine.route_table, MNN_Engine.linkBW, MNN_Engine.links)
                fitness[nn][c_num]['t'] += perf['L'] + commPerf
                fitness[nn][c_num]['noc'] += perf['L']
                fitness[nn][c_num]['nop'] += commPerf
    return fitness, neu_Needed
    
def get_perf(fitness, chiplets, nn_list):
    nn2id = {}
    total_latency = []
    nops = []
    nocs = []
    for id, nn in enumerate(nn_list):
        nn2id[nn] = id
    for nn in fitness.keys():
        id = nn2id[nn]
        chiplet = chiplets[id]
        latency = fitness[nn][chiplet]['t']
        latency_noc = fitness[nn][chiplet]['noc'] 
        latency_nop = fitness[nn][chiplet]['nop'] 
            
        total_latency.append(latency)
        nops.append(latency_nop)
        nocs.append(latency_noc)
    return total_latency, nops, nocs

def planria(fitness, nn_list):
    latency_best = float('inf')
    latency_nop = 0
    latency_noc = 0
    chiplet_best = None
    nn_num = len(nn_list)
    if nn_num == 4:
        chiplet_partition_dict = MNN_Engine.getChipletPartionDict(4, 2)
    else:
        chiplet_partition_dict = MNN_Engine.getChipletPartionDict(nn_num, nn_num-1)
    # iterate over the DNN models in the order specified by workload_fitness_dict
    for chiplets in chiplet_partition_dict[nn_num]:
        for i in range(10):
            random.shuffle(chiplets)
            total_latency, nops, nocs = get_perf(fitness, chiplets, nn_list)
            if max(total_latency) < latency_best:
                latency_best = max(total_latency)
                latency_nop = nops
                latency_noc = nocs
                chiplet_best = chiplets

    return latency_best, latency_nop, latency_noc, chiplet_best
    

if __name__ == '__main__':
    architecture = 'simba'
    chiplet_num = 16
    mem_num = 4
    Optimization_Objective = 'latency'
    tp_TH = 4
    sp_TH = 4

    type = 'vision'
    if type == 'vision':
        workload_dict = ['resnet50', 'darknet19', 'VGG16']
    elif type == 'nlp':
        workload_dict = ['GNMT', 'BERT', 'ncf']
    else:
        workload_dict = ['BERT', 'GNMT', 'resnet18', 'vit']

    MNN_Engine = multi_network_DSE(architecture, chiplet_num, mem_num, workload_dict, Optimization_Objective, tp_TH=0.2, sp_TH=1,  topology='cmesh',partition_ratio=1, BW_tag=False, layout_mapping_method=None)
    MNN_Engine.initialize()
    MNN_Engine.setTotalWorkloadFitness()
    
    fitness, neuNeeded = merge_workload(MNN_Engine, workload_dict)

    total_latency, latency_nop, latency_noc, chiplet_best = planria(fitness, workload_dict)

    print('total_latency = ', total_latency)
    print('chiplet_best = ', chiplet_best)
    print('workload_dict = ', workload_dict)
    
    line = '------------- {} -----------\n'.format('Planria')
    line += 'architecture : {}\n'.format(architecture)
    line += 'nn_name : {}\n'.format(workload_dict)
    line += 'interconnection : {}\n'.format(arch_tag)
    line += 'total_latency :\t{}\t\n'.format(total_latency)
    line += 'latency_nop :\t{}\t\n'.format(latency_nop)
    line += 'latency_noc :\t{}\t\n'.format(latency_noc)
    line += 'workload_dict :\t{}\t\n'.format(workload_dict)
    line += 'chiplet_best :\t{}\t\n'.format(chiplet_best)
    line += '\n'
    
    file_n = './Planria.txt'
    if os.path.exists(file_n) == False:
        f = open(file_n, 'w')
    else:
        f = open(file_n, 'a')
    
    print(line, file=f)
    f.close()
