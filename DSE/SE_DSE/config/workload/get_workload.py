
def getLayerParam(nn_name):
    path = '/home/wangxy/workspace/chiplet/wxy_chiplet/DSE/SE_DSE/nn_file'
    f = open('{}/{}.txt'.format(path, nn_name), 'r')

    print("network model ----- " + nn_name + " -------------")

    lines = f.readlines()
    layer_dict = {}
    layer_name_list = []
    layer_num = 0
    compute_num = 0
    mem_num = 0
    i_num = 0
    o_num = 0
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
            layer = {"P":P,"Q":Q,"C":C,"K":K,"R":R,"S":S, "stride":stride, "padding":padding}
            layer_dict[layer_num] = layer
            layer_name_list.append(layer_name)
            layer_num += 1
            compute_num += P*Q*K*C*stride*stride
            mem_num += P*Q*K + H*M*C + K*C*stride*stride
            i_num += K*C*stride*stride + H*M*C
            o_num += P*Q*K
    
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
    
    return layer_dict_unique, layer_name_dict, compute_num, mem_num, i_num, o_num

def get_workload(nn_name):
    layer_dict, layer_name_dict, compute_num, mem_num, i_num, o_num = getLayerParam(nn_name)
    
    line = ''
    for layer_name, layer in layer_dict.items():
        line += '{}:\n'.format(layer_name)
        for dim, num in layer.items():
            line += '  {}: {}\n'.format(dim, num)
    line += '{}:\n'.format("layer_name_dict")
    for layer_name, layer_refer in layer_name_dict.items():
        line += '  {}: {}\n'.format(layer_name, layer_refer)
    
    f = open("{}.yaml".format(nn_name), 'w')
    print(line, file = f)
    f.close()
    return compute_num, mem_num, i_num, o_num

if __name__ == "__main__":
    name_list = ["resnet18", "resnet50", "alexnet", "VGG16",'BERT', 'darknet19', 'GNMT', 'ncf', 'Unet', 'vit', 'lenet']
    lines = "nn_name\tcompute\tmem\t\n"
    for name in name_list:
        compute_num, mem_num, i_num, o_num = get_workload(name)
        lines += "{}\t{}\t{}\t{}\t{}\t\n".format(name, compute_num, mem_num, i_num, o_num)
    file = open('workload_param.txt', 'w')
    print(lines, file=file)
    file.close()
        