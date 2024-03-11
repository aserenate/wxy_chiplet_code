from simplified_multi_network_DSE import *
import os
arch_tag = "ours"
def fcfs_scheduling(MNN_Engine):
    total_latency = 0
    latency_nop = 0
    latency_noc = 0
    chiplet_num = MNN_Engine.chiplet_num
    
    # iterate over the DNN models in the order specified by workload_fitness_dict
    for workload in MNN_Engine.workload_fitness_dict.keys():
        latency = 0
        for layer, perf in MNN_Engine.workload_fitness_dict[workload][chiplet_num].items():
            comm_num = MNN_Engine.workload_NeuNeeded_dict[workload][chiplet_num][layer]
            ifmapD = int(comm_num['ifmap_DRAM'])
            weightD = int(comm_num['weight_DRAM'])
            ofmapD = int(comm_num['ofmap_DRAM'])
            parallel = comm_num['parallel']
            if arch_tag == "ours":
                commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, 'ours', None, None, None)
            else:
                commPerf = comm_performance(ifmapD, weightD, ofmapD, parallel, arch_tag, MNN_Engine.route_table, MNN_Engine.linkBW, MNN_Engine.links)
                
            latency_noc +=  perf['L']
            latency += perf['L'] + commPerf * 1.15
            latency_nop += commPerf * 1.15
            
        total_latency += latency
    return total_latency, latency_nop, latency_noc
    

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
        
    MNN_Engine = multi_network_DSE(architecture, chiplet_num, mem_num, workload_dict, Optimization_Objective, tp_TH=0.2, sp_TH=0.1,  topology='cmesh',partition_ratio=1, BW_tag=False, layout_mapping_method=None)
    MNN_Engine.initialize()
    MNN_Engine.setTotalWorkloadFitness()

    total_latency, latency_nop, latency_noc = fcfs_scheduling(MNN_Engine)
    
    print('latency_noc = ', latency_noc)
    print('latency_nop = ', latency_nop)
    print('total_latency = ', total_latency)
    line = '------------- {} -----------\n'.format('PRAMA')
    line += 'architecture : {}\n'.format(architecture)
    line += 'nn_name : {}\n'.format(workload_dict)
    line += 'interconnection : {}\n'.format(arch_tag)
    line += 'latency :\t{}\t{}\t{}\t\n'.format(total_latency, latency_noc, latency_nop)
    line += '\n'
    
    file_n = './PREMA.txt'
    if os.path.exists(file_n) == False:
        f = open(file_n, 'w')
    else:
        f = open(file_n, 'a')
    
    print(line, file=f)
    f.close()