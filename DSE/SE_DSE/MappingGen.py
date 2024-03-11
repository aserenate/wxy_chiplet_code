
import math
import random
import copy

act_tag = 1001
wgt_tag = 1002
out_tag_m2c = 1003
out_tag_c2m = 1004

# 质因数分解
def getZhiyinShu(num, list):
	isZhishu = True
	i = 2
	square = int(math.sqrt(num)) + 1
	while i <= square:
		if num % i == 0:
			list.append(i)
			isZhishu = False
			getZhiyinShu(num / i , list)
			i += 1
			break
		i += 1
	if isZhishu:
		list.append(int(num))

# 将一个数拆成三个整数相乘
def setPartition_1(num, dim):
	par_list = []
	getZhiyinShu(num, par_list)
	par_dim = [1, 1, 1]
	for i in par_list:
		ran = random.randint(1, dim)
		par_dim[ran-1] *= i
	return par_dim

# 将num拆分为dim个整数相乘（大于等于）
def setPartition(num, dim):
	list = []
	num_i = num
	for i in range(0, dim - 1):
		ran = random.randint(1, math.ceil(num_i))
		list.append(ran)
		num_i = num_i / ran

	list.append(math.ceil(num_i))
	random.shuffle(list)
	return list

# Single Chiplet + Single Network
class MappingGenerator:
    def __init__(self, temporal_level, HW_param, dimensions_tag):
        self.temporal_level = temporal_level
        self.Chiplet_size   = HW_param["Chiplet"]
        self.PE_size        = HW_param["PE"]
        self.MAC_size       = HW_param["MAC"]
        self.dimensions_tag = dimensions_tag
        self.nn_param       = None
        self.t_dimensions   = None
        self.correlations   = None
        self.samples        = None
        
        self.s_parallel     = None
        self.t_partition    = None
        self.orders         = None
        self.s_type         = None
        
        self.samples        = None
    
    # ------------------------
    # set_layer
    # ------------------------
    def set_layer(self, layer):
        self.nn_param = layer
    
    # ------------------------------------------------------
    # set s_parallel (PE Level): s_parallel = {'P': , 'K': }
    # ------------------------------------------------------
    def set_spatial_parallel(self, sp=None, Chiplet_type='PK', PE_type='PK', MAC_type='KC'):
        def get_parallel_from_type(type, size):
            parallel = {}
            if type[0] == type[1]:
                parallel[type[0]] = size['x'] * size['y']
            else:
                parallel[type[0]] = size['x']
                parallel[type[1]] = size['y']
            return parallel
        if sp != None:
            self.s_parallel = sp
        else:
            self.s_parallel = {}
            self.s_parallel["Chiplet"] = copy.deepcopy(get_parallel_from_type(Chiplet_type, self.Chiplet_size))
            self.s_parallel["PE"] = copy.deepcopy(get_parallel_from_type(PE_type, self.PE_size))
            self.s_parallel["MAC"] = copy.deepcopy(get_parallel_from_type(MAC_type, self.MAC_size))
        self.s_type = {"Chiplet": Chiplet_type, "PE": PE_type, "MAC": MAC_type}
    
    # -----------------------------------------------------------
    # set temporal dimensions : temporal_dims = {'P': , 'Q': ...}
    # set correlations : correlations = {'P' : [1, 0, 1] ...}
    # -----------------------------------------------------------
    def set_temporal_dimensions(self):
        temporal_dims = []
        self.correlations = {}
        self.t_dimensions = {}
        for dim, tag in self.dimensions_tag.items():
            if len(tag) > 2 and tag[2] != 'spatial':
                temporal_dims.append(dim)
                self.correlations[dim] = tag[3]['correlation']
                
        for dim in temporal_dims:
            self.t_dimensions[dim] = self.nn_param[dim]
        
        for arch, parallels in self.s_parallel.items():
            for dim, p_num in parallels.items():
                self.t_dimensions[dim] = math.ceil(self.t_dimensions[dim] / p_num)
    
    # ------------------------------------------------------------
    # get temporal partitions randomlly : t_partition
    #   t_partition : {'P' : [xx, xx] ...}
    #   partition_tag : random_equal, random_not_smaller or single
    # ------------------------------------------------------------
    def get_temporal_partitions(self):
        self.t_partition = {}
        for dim, num in self.t_dimensions.items():
            partition = [1 for _ in range(self.temporal_level)]
            partition_tag = self.dimensions_tag[dim][0]
            
            if partition_tag == 'random_equal':
                partition = setPartition_1(num, self.temporal_level)
            elif partition_tag == 'random_not_smaller':
                partition = setPartition(num, self.temporal_level)
            elif partition_tag == 'single':
                partition[0] = num
            else:
                print("error partition tag : ", partition_tag)
                exit()
            
            self.t_partition[dim] = partition
    
    # ---------------------------------------
    # get orders
    #   order_tag : random, bottom, top
    #   orders : [{'P': 0.3, 'Q': 0.2 ... }]        
    # ---------------------------------------
    def get_orders(self):
        self.orders = [{} for _ in range(self.temporal_level)]
        for dim in self.t_dimensions:
            order_tag = self.dimensions_tag[dim][1]
            
            if order_tag == 'random':
                for i in range(self.temporal_level):
                    order = random.random()
                    while order == 0:
                        order = random.random()
                    if self.t_partition[dim][i] != 1:
                        self.orders[i][dim] = order
            elif order_tag == 'bottom':
                for i in range(self.temporal_level):
                    if self.t_partition[dim][i] != 1:
                        self.orders[i][dim] = 0
            elif order_tag == 'top':
                for i in range(self.temporal_level):
                    if self.t_partition[dim][i] != 1:
                        self.orders[i][dim] = 1
            else:
                print("error order tag : ", order_tag)
                exit()
    
    # ---------------------------------------
    # parse
    # ---------------------------------------
    def parse(self):
        dataflow = []
        all_param = []
        ol1_ratio = []
        al1_ratio = []
        wl1_ratio = []
        out_final_dim = 0
        for t, order in enumerate(self.orders):
            s_order = sorted(list(order.items()), key = lambda x: x[1])
            for (dim, _) in s_order:
                name = '{}{}'.format(dim, t)
                num = self.t_partition[dim][t]
                assert(num > 1)
                if num > 1:
                    dataflow.append(name)
                    all_param.append(num)
                    correlation = self.correlations[dim]
                    if correlation[0]:
                        al1_ratio.append(num)
                    else:
                        al1_ratio.append(1)
                    
                    if correlation[1]:
                        wl1_ratio.append(num)
                    else:
                        wl1_ratio.append(1)
                        
                    if correlation[2]:
                        ol1_ratio.append(num)
                    else:
                        ol1_ratio.append(1)
                        if num > 1:
                            out_final_dim = len(dataflow)
                    
        out_final = []
        for i in range(len(ol1_ratio)):
            if i < out_final_dim:
                out_final.append(0)
            else:
                out_final.append(1)
        
        dataflow.append("top")
        ol1_ratio.append(1)
        al1_ratio.append(1)
        wl1_ratio.append(1)
        all_param.append(1)
        out_final.append(1)
        
        for_list = {}
        for_list[0] = dataflow
        for_list[1] = ol1_ratio
        for_list[2] = al1_ratio
        for_list[3] = wl1_ratio
        for_list[4] = all_param
        for_list[5] = out_final
        for_list[6] = None
        for_list[7] = None
        for_list[8] = None
        for_list[9] = None
        
        dim2id = {'P':0, 'Q':1, 'C':2, 'K':3}
        parallel_dim_list = {0: [1, 1, 1, 1], 1: [1, 1, 1, 1]}
        for dim, num in self.s_parallel['PE'].items():
            dim_id = dim2id[dim]
            parallel_dim_list[0][dim_id] = num
        for dim, num in self.s_parallel['Chiplet'].items():
            dim_id = dim2id[dim]
            parallel_dim_list[1][dim_id] = num
        
        partition_list = {}
        for dim in dim2id:
            partition_list[dim] = [1 for _ in range(3)]
            partition_list[dim][0:self.temporal_level] = self.t_partition[dim][0:self.temporal_level]
        

        def get_comm_nodes(correlation, dx, dy, idx, size_x, size_y):
            nodes = {}
            if correlation[dx][idx] == 1 and correlation[dy][idx] == 1:
                for y in range(size_y):
                    for x in range(size_x):
                        node = x + y * size_y
                        nodes[node] = [node]
            elif correlation[dx][idx] == 0 and correlation[dy][idx] == 0:
                nodes = {0:[]}
                for y in range(size_y):
                    for x in range(size_x):
                        node = x + y * size_y
                        nodes[0].append(node)
            elif correlation[dx][idx] == 1:
                for y in range(size_y):
                    for x in range(size_x):
                        node = x + y * size_y
                        if x not in nodes:
                            nodes[x] = []
                        nodes[x].append(node)
            elif correlation[dy][idx] == 1:
                for y in range(size_y):
                    nodes[y] = []
                    for x in range(size_x):
                        node = x + y * size_y
                        nodes[y].append(node)
            return nodes
        
        comm_patterns = {act_tag: {}, wgt_tag: {}, out_tag_m2c: {}, out_tag_c2m: {}}
        dim_x = self.s_type["PE"][0]
        dim_y = self.s_type["PE"][1]
        act_nodes = get_comm_nodes(self.correlations, dim_x, dim_y, 0, self.PE_size['x'], self.PE_size['y'])
        wgt_nodes = get_comm_nodes(self.correlations, dim_x, dim_y, 1, self.PE_size['x'], self.PE_size['y'])
        out_nodes = get_comm_nodes(self.correlations, dim_x, dim_y, 2, self.PE_size['x'], self.PE_size['y'])
        self.Mesh_nodes()
        # act
        comm_patterns[act_tag][self.mem_nodes['a']] = {}
        for i, nodes in act_nodes.items():
            comm_patterns[act_tag][self.mem_nodes['a']][i] = []
            for node in nodes:
                c_node = self.compute_nodes[node]
                comm_patterns[act_tag][self.mem_nodes['a']][i].append(c_node)
        
        # wgt
        comm_patterns[wgt_tag][self.mem_nodes['w']] = {}
        for i, nodes in wgt_nodes.items():
            comm_patterns[wgt_tag][self.mem_nodes['w']][i] = []
            for node in nodes:
                c_node = self.compute_nodes[node]
                comm_patterns[wgt_tag][self.mem_nodes['w']][i].append(c_node)
       
        # out
        comm_patterns[out_tag_m2c][self.mem_nodes['o1']] = {}
        comm_patterns[out_tag_m2c][self.mem_nodes['o2']] = {}
        comm_patterns[out_tag_c2m] = {}
        for i, nodes in out_nodes.items():
            if i < len(out_nodes) // 2:
                mem_node = self.mem_nodes['o1']
                packet_id = i
            else:
                mem_node = self.mem_nodes['o2']
                packet_id = i - len(out_nodes) // 2
                
            comm_patterns[out_tag_m2c][mem_node][packet_id] = []
            for node in nodes:
                c_node = self.compute_nodes[node]
                comm_patterns[out_tag_m2c][mem_node][packet_id].append(c_node)
                comm_patterns[out_tag_c2m][c_node] = {0: [mem_node]}
        
        return for_list, comm_patterns, parallel_dim_list, partition_list
    
    # ---------------------------
    # Mesh_nodes : get the nodes
    # ---------------------------
    def Mesh_nodes(self):
        mem_nodes = []
        self.compute_nodes = []
        self.total_nodes = []
        for y in range(self.PE_size['y']):
            for x in range(self.PE_size['x'] + 1):
                node = x + y * self.PE_size['y']
                if x == 0:
                    mem_nodes.append(node)
                else:
                    self.compute_nodes.append(node)
                self.total_nodes.append(node)
        mem_num = len(mem_nodes)
        self.mem_nodes = {'a': mem_nodes[1%mem_num], 'w': mem_nodes[2%mem_num], 'o1':mem_nodes[0%mem_num], 'o2':mem_nodes[3%mem_num]}
    
    # -----------------------------
    # fresh : generate the sample
    # -----------------------------   
    def fresh(self):
        self.get_temporal_partitions()
        self.get_orders()
        return self.parse()

'''
INPUT: temporal_level, HW_param, nn_param, dimensions_tag
    temporal_level: int
    HW_param: {"PE": {'x': , 'y': }, "MAC": {'dim1': , 'dim2': }}
    nn_param: {'P': int, 'Q': int}
    dimensions_tag: {"dim": ['random', 'random', 'both', {'correlation': [1, 0, 1]} ],  xxx }
'''

'''
pre version of act_wgt_dict
        # act
        act_core = {}
        act_core['send'] = {0:[self.mem_nodes['a']]}
        act_core['recv'] = {}
        for i, nodes in act_nodes:
            act_core['recv'][i] = []
            for node in nodes:
                c_node = self.compute_nodes[node]
                act_core['recv'][i].append(c_node)
        act_wgt_dict["act_core"].append(act_core)
        
        # wgt
        wgt_core = {}
        wgt_core['send'] = {0:[self.mem_nodes['w']]}
        wgt_core['recv'] = {}
        for i, nodes in wgt_nodes:
            wgt_core['recv'][i] = []
            for node in nodes:
                c_node = self.compute_nodes[node]
                wgt_core['recv'][i].append(c_node)
        act_wgt_dict["wgt_core"].append(wgt_core)
       
        # out
        out_dict = {"rd_core":[], "rd_chip":None}
        out_core1 = {}
        out_core1['send'] = {0:[self.mem_nodes['o1']]}
        out_core1['recv'] = {}
        out_core2 = {}
        out_core2['send'] = {0:[self.mem_nodes['o2']]}
        out_core2['recv'] = {}
        for i, nodes in out_nodes:
            if i < len(out_nodes) // 2:
                out_core1['recv'][i] = []
            else:
                out_core2['recv'][i] = []
            for node in nodes:
                c_node = self.compute_nodes[node]
                if i < len(out_nodes) // 2:
                    out_core1['recv'][i].append(c_node)
                else:
                    out_core2['recv'][i].append(c_node)
        out_dict["rd_core"].append(out_core1)
        out_dict["rd_core"].append(out_core2)
        
'''