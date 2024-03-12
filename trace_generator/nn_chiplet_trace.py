import math
import random
import os
# config parameter
store_mode = 'pix' # pix or c
other_mode = 'c'
link_width = 4 * 4 # 4Byte, latency needs to multiple 4
packet_size = link_width * 4

scaleSize = 10

in_path = '/home/wangxy/workspace/chiplet/wxy_chiplet/DSE/SE_DSE/nn_file'
out_path = "/home/wangxy/workspace/chiplet/simulator_gem5/task_file/"

def read_network(nn_name):
    path = in_path
    f = open('{}/{}.txt'.format(path, nn_name), 'r')

    print("network model ----- " + nn_name + " -------------")

    lines = f.readlines()
    layer_dict = {}
    layer_name_list = []
    layer_num = 0
    for line in lines:
        if line.startswith("#") or line.startswith("*"):
            pass
        else:
            line = line.replace("\n","")
            line_item = line.split(" ")
            layer_name = line_item[0]
            H = int(line_item[1])
            M = int(line_item[2])
            P = int(line_item[8])
            Q = int(line_item[9])
            C = int(line_item[3])
            K = int(line_item[7])
            R = int(line_item[4])
            S = int(line_item[4])
            stride = int(line_item[5])
            padding = int(line_item[6])
            layer = {'H':H, 'M':M, "P":P,"Q":Q,"C":C,"K":K,"R":R,"S":S, "stride":stride, "padding":padding}
            layer_dict[layer_num] = layer
            layer_name_list.append(layer_name)
            layer_num += 1
    
    layer_list = []
    layer_dict_unique = {}
    layer_name_dict = {}
    layer_num -= 1
    for i, layer in layer_dict.items():
        layer_name = layer_name_list[i]

        if layer not in layer_list:
            layer_dict_unique[layer_name] = layer
            layer_list.append(layer)
            layer_name_dict[layer_name] = layer_name
        else:
            for layer_name_1, layer_1 in layer_dict_unique.items():
                if layer == layer_1:
                    layer_name_same = layer_name_1
                    break
            layer_name_dict[layer_name] = layer_name_same

    f.close()
    
    return layer_dict_unique, layer_name_dict

def get_node(size_row, size_col):
    mem_nodes = []
    compute_nodes = []
    dummy_nodes = []
    compute_2_mem = {}
    for row in range(size_row):
        id1 = row * size_col
        id2 = row * size_col + size_col - 1
        if row % 2 == 0:
            # dummy node
            dummy_nodes.append(id1)
            dummy_nodes.append(id2)
            mem_id1 = id1 + size_col
            mem_id2 = id2 + size_col
        else:
            mem_nodes.append(id1)
            mem_nodes.append(id2)
        for col in range(1, size_col-1):
            id = row * size_col + col
            compute_nodes.append(id)
            
            if col < size_col//2:
                compute_2_mem[id] = mem_id1
            else:
                compute_2_mem[id] = mem_id2
    return mem_nodes, compute_nodes, dummy_nodes, compute_2_mem

def generate_trace(size_row, size_col, mem_num, act_size={'pix': 16, 'c':32}, wgt_size={ 'ci':32, 'co':16, 'pix': 9}, out_size={'pix': 16, 'c':32}, mapping_pattern={'pix':4, 'c':4}, app_name='resnet18', layer_name='layer1'):
    cal_cycle = 1
    # 1 chiplet needed data calculation
    chiplet_needed_act = {}
    chiplet_needed_wgt = {}
    chiplet_needed_out = {}
    
    # act
    if store_mode == 'pix':
        pix_partition = max(mem_num, mapping_pattern['pix'])
        num = math.ceil(mem_num / mapping_pattern['pix'])
        act_size_mem = math.ceil(math.ceil(act_size['pix'] / pix_partition) * act_size['c'] // packet_size / scaleSize)
        for i in range(mapping_pattern['pix']):
            for j in range(mapping_pattern['c']):
                chiplet_id = i * mapping_pattern['c'] + j
                act_mem_nodes = []
                for m in range(num):
                    act_mem_nodes.append(i * num + m)
                chiplet_needed_act[chiplet_id] = act_mem_nodes
    else:
        # store c
        act_size_mem = math.ceil(math.ceil(act_size['c'] / mem_num) * act_size['pix'] // packet_size / scaleSize)
        size = min(act_size['c'], mem_num)
        for i in range(mapping_pattern['pix']):
            for j in range(mapping_pattern['c']):
                chiplet_id = i * mapping_pattern['c'] + j
                act_mem_nodes = []
                for m in range(size):
                    act_mem_nodes.append(m)
                chiplet_needed_act[chiplet_id] = act_mem_nodes
    # out
    partition_size = max(mem_num, mapping_pattern[store_mode])
    out_size_mem = math.ceil(math.ceil(out_size[store_mode] / partition_size) * out_size[other_mode] // packet_size / scaleSize)
    num = math.ceil(mem_num / mapping_pattern[store_mode])
    for i in range(mapping_pattern['pix']):
        for j in range(mapping_pattern['c']):
            chiplet_id = i * mapping_pattern['c'] + j
            out_mem_nodes = []
            for m in range(num):
                if store_mode == 'pix':
                    mem_id = i * num + m
                else:
                    mem_id = j * num + m
                out_mem_nodes.append(mem_id)
            chiplet_needed_out[chiplet_id] = out_mem_nodes
    # wgt
    partition_size = max(mem_num, mapping_pattern['c'])
    wgt_size_mem = math.ceil(math.ceil(wgt_size['co'] / partition_size) * wgt_size['ci'] * wgt_size['pix'] // packet_size / scaleSize)
    num = math.ceil(mem_num / mapping_pattern['c'])
    for i in range(mapping_pattern['pix']):
        for j in range(mapping_pattern['c']):
            chiplet_id = i * mapping_pattern['c'] + j
            wgt_mem_nodes = []
            for m in range(num):
                mem_id = j * num + m
                wgt_mem_nodes.append(mem_id)
            chiplet_needed_wgt[chiplet_id] = wgt_mem_nodes
            
    # 2 trace generate
    mem_nodes, compute_nodes, dummy_nodes, compute_2_mem = get_node(size_row, size_col)
    inst_dict_base = {}
    inst_dict_ours = {}
    wait_num = {}
    
    for node in mem_nodes:
        wait_num[node] = 0
        inst_dict_base[node] = []
        inst_dict_ours[node] = []
    
    for node in compute_nodes:
        wait_num[node] = {'a':0, 'w':0}
        inst_dict_base[node] = []
        inst_dict_ours[node] = []
    
    for node in dummy_nodes:
        inst_dict_base[node] = ["finish\n"]
        inst_dict_ours[node] = []
    
    if act_size_mem == 0:
        pass
    else:
        for c_node_id, m_nodes in chiplet_needed_act.items():
            c_node = compute_nodes[c_node_id]
            for m_node_id in m_nodes:
                m_node = mem_nodes[m_node_id]
                wait_num[c_node]['a'] += act_size_mem
                inst_dict_base[m_node].append("send {} {} {} // act\n".format(c_node, act_size_mem, 1001))
    if wgt_size_mem == 0:
        pass
    else:
        for c_node_id, m_nodes in chiplet_needed_wgt.items():
            c_node = compute_nodes[c_node_id]
            for m_node_id in m_nodes:
                m_node = mem_nodes[m_node_id]
                wait_num[c_node]['w'] += wgt_size_mem
                inst_dict_base[m_node].append("send {} {} {} // wgt\n".format(c_node, wgt_size_mem, 1002))
    
    if out_size_mem == 0:
        pass
    else:
        for c_node_id, m_nodes in chiplet_needed_out.items():
            c_node = compute_nodes[c_node_id]
            for m_node_id in m_nodes:
                m_node = mem_nodes[m_node_id]
                wait_num[m_node] += out_size_mem
                inst_dict_base[c_node].append("send {} {} {} // out\n".format(m_node, out_size_mem, 1003))
                inst_dict_ours[c_node].append("send {} {} {} // out\n".format(m_node, out_size_mem, 1003))
            
    for m_node in mem_nodes:
        random.shuffle(inst_dict_base[m_node])
    
    for c_node in compute_nodes:
        random.shuffle(inst_dict_base[c_node])
        random.shuffle(inst_dict_ours[c_node])
    
    for c_node in compute_nodes:
        inst_dict_base[c_node].append("cal {}\n".format(cal_cycle))
        inst_dict_ours[c_node].append("cal {}\n".format(cal_cycle))
        if wait_num[c_node]['a'] > 0:
            inst_dict_base[c_node].append("wait {} {} // act\n".format(wait_num[c_node]['a'], 1001))
        if wait_num[c_node]['w'] > 0:
            inst_dict_base[c_node].append("wait {} {} // wgt\n".format(wait_num[c_node]['w'], 1002))
        inst_dict_base[c_node].append("finish\n")
        inst_dict_ours[c_node].append("finish\n")
        
    act_num = math.ceil(act_size['pix'] * act_size['c'] // mem_num // packet_size  / scaleSize)
    wgt_num = math.ceil(wgt_size['ci'] * wgt_size['co'] * wgt_size['pix'] // mem_num // packet_size  / scaleSize)
    for i in range(mem_num):
        dummy_node = dummy_nodes[i]
        mem_node = mem_nodes[i]
        inst_dict_ours[mem_node].append("send {} {} {} // act\n".format(dummy_node, act_num, 1001))
        inst_dict_ours[mem_node].append("send {} {} {} // wgt\n".format(dummy_node, wgt_num, 1002))
        inst_dict_ours[dummy_node].append("wait {} {} // act\n".format(act_num, 1001))
        inst_dict_ours[dummy_node].append("wait {} {} // wgt\n".format(wgt_num, 1002))
        inst_dict_ours[dummy_node].append("finish\n")
        inst_dict_ours[dummy_node].append("finish\n")
    
    for m_node in mem_nodes:
        if wait_num[m_node]> 0:
            inst_dict_base[m_node].append("wait {} {} // out\n".format(wait_num[m_node], 1003))
            inst_dict_ours[m_node].append("wait {} {} // out\n".format(wait_num[m_node], 1003))
        inst_dict_base[m_node].append("finish\n")
        inst_dict_ours[m_node].append("finish\n")
    
    
    base_inst_line_dict = {}
    ours_inst_line_dict = {}
    for node_id, i_list in inst_dict_base.items():
        line = ""
        for i in i_list:
            line += i
        base_inst_line_dict[node_id] = line    
    
    for node_id, i_list in inst_dict_ours.items():
        line = ""
        for i in i_list:
            line += i
        ours_inst_line_dict[node_id] = line 
    
    out_Dir = out_path
    if os.path.exists(out_Dir) == False:
        os.mkdir(out_Dir)
    out_Dir = out_path + "{}/".format(app_name)
    if os.path.exists(out_Dir) == False:
        os.mkdir(out_Dir)
    
    out_Dir_base = "{}base/".format(out_Dir)
    out_Dir_ours = "{}ours/".format(out_Dir)
    if os.path.exists(out_Dir_base) == False:
        os.mkdir(out_Dir_base)
    if os.path.exists(out_Dir_ours) == False:
        os.mkdir(out_Dir_ours)
    out_Dir_base = "{}{}/".format(out_Dir_base,layer_name)
    out_Dir_ours = "{}{}/".format(out_Dir_ours,layer_name)
    if os.path.exists(out_Dir_base) == False:
        os.mkdir(out_Dir_base)
    if os.path.exists(out_Dir_ours) == False:
        os.mkdir(out_Dir_ours)

    for node_id, inst in base_inst_line_dict.items():
        file = "{}{}.txt".format(out_Dir_base, node_id)
        ff = open(file, 'w')
        print(inst, file=ff)
        ff.close()
    for node_id, inst in ours_inst_line_dict.items():
        file = "{}{}.txt".format(out_Dir_ours, node_id)
        ff = open(file, 'w')
        print(inst, file=ff)
        ff.close()
    
if __name__ == '__main__':
    size_row = 4
    size_col = 6
    mem_num = 4
    mapping_pattern = {'pix':4, 'c':4}
    nns = ['resnet18', 'resnet50', 'VGG16', 'vit']
    
    for name in nns:
        layer_dict, layer_name_dict = read_network(name)
        for layer_name, layer in layer_dict.items():
            act_size = {'pix': layer['H']*layer['M'], 'c': layer['C']}
            wgt_size = {'ci': layer['C'], 'co': layer['K'], 'pix': layer['R']*layer['S']}
            out_size = {'pix': layer['P']*layer['Q'], 'c': layer['K']}
            generate_trace(size_row, size_col, mem_num, act_size, wgt_size, out_size, mapping_pattern, app_name=name, layer_name=layer_name)
        file = out_path + "{}/{}.txt".format(name, "layer_name_dict")
        f = open(file, 'w')
        print("layer_dict:--------", file=f)
        print(layer_dict, file=f)
        print("-------------------", file=f)
        f.close()