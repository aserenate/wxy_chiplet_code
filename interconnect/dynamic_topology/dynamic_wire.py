from ast import main
import numpy as np
import random
import yaml
import os

## RX, TX, IS
## --- 0: OFF
## --- 1: ON
## direction
## --- TIR: TX IS RX, TX IS RX
## --- RIT : RX IS TX, RX IS TX
class DYWire:
    def __init__(self, name, node_num, direction='TIR', bandwidth=1):
        self.name = name
        self.node_num = node_num
        self.node_list = [i for i in range(node_num)]
        self.direction = direction
        self.bandwidth = bandwidth
        self.TX_status = [0 for _ in range(node_num)]
        self.RX_status = [1 for _ in range(node_num)]
        self.IS_status = [0 for _ in range(node_num)]
        self.wire_status = [0 for _ in range(2*node_num)]
        
        self.routing_table = None
    
    def set_node_list(self, node_list):
        self.node_list = node_list
    
    def random_wire(self):
        # 1. 随机IS的开闭状态
        self.IS_status = list(np.random.randint(0, 2, (self.node_num)))
        print("IS_status: ", self.IS_status)

        # 2. 分段
        start_id = -1
        section_list = []
        for i, status in enumerate(self.IS_status):
            if status == 0:
                section_list.append([start_id, i])
                start_id = i
            else:
                pass
        section_list.append([start_id, self.node_num])
        print("section_list: ", section_list)
        
        # 3. 选择每段的TX节点
        for [start_id, end_id] in section_list:
            if self.direction == 'TIR':
                i = random.randint(start_id+1, end_id)
            else:
                i = random.randint(start_id, end_id-1)
            if i >= 0 and i < self.node_num:
                self.TX_status[i] = 1
        print("TX_status: ", self.TX_status)
        
        # 4. 确定RX的开关情况
        for i in range(self.node_num):
            if self.IS_status[i] == 1 and self.TX_status[i] == 1:
                self.RX_status[i] = 0
        print("RX_status: ", self.RX_status)
    
    def wire_status_load(self, name):
        file = './wire_record/{}.yaml'.format(name)
        with open(file, encoding='utf-8') as rstream:
            data = yaml.load(rstream, yaml.SafeLoader)
        
        for type, item in data.items():
            if type == "TX_status":
                self.TX_status = item
            elif type == "RX_status":
                self.RX_status = item
            elif type == "IS_status":
                self.IS_status = item
            elif type == "direction":
                self.direction = item
            else:
                print("Error type : {}".format(type))
                exit()
    
    def Mesh_status(self):
        self.TX_status = [1 for _ in range(self.node_num)]
        self.RX_status = [1 for _ in range(self.node_num)]
        self.IS_status = [0 for _ in range(self.node_num)]
    
    def Bus_status(self, main_node):
        self.TX_status = [0 for _ in range(self.node_num)]
        self.TX_status[main_node] = 1
        self.RX_status = [1 for _ in range(self.node_num)]
        self.RX_status[main_node] = 0
        self.IS_status = [1 for _ in range(self.node_num)]
            
    def record_status(self, outDir=''):
        line = "TX_status: \n"
        for status in self.TX_status:
            line += "  - {}\n".format(status)
        line += "RX_status: \n"
        for status in self.RX_status:
            line += "  - {}\n".format(status)
        line += "IS_status: \n"
        for status in self.IS_status:
            line += "  - {}\n".format(status)
        line += "direction: {}\n".format(self.direction)
        line += self.wire_print(0)
        out_dir = './wire_record/{}/'.format(outDir)
        file = '{}{}.yaml'.format(out_dir, self.name)
        if os.path.exists(out_dir) == False:
            os.mkdir(out_dir)
        r_f = open(file, 'w')
        print(line, file=r_f)
        r_f.close()
    
    def wire_print(self, print_tag=1):
        line = "#\t"
        # --- line 1
        for i in range(self.node_num):
            line += "{:^10s}".format(str(self.node_list[i]))
        line += "\n#\t"
        # --- line 2
        for i in range(self.node_num):
            line += "{:^10s}".format("________")
        line += "\n#\t"
        # --- line 3
        for i in range(self.node_num):
            if self.TX_status[i] == 1:
                tx = "{:^5s}".format('|')
            else:
                tx = "{:^5s}".format('')
            if self.RX_status[i] == 1:
                rx = "{:^5s}".format('^')
            else:
                rx = "{:^5s}".format('')
                
            if self.direction == 'TIR':
                line += tx + rx
            else:
                line += rx + tx
        line += "\n#\t"
        # --- line 4
        for i in range(self.node_num):
            if self.IS_status[i]==1:
                line += "{:_^10s}".format("")
            else:
                line += "{:_^10s}".format("    ")
        line += "\n"
        
        if print_tag == 1:
            print(line)
        else:
            return line
        
    def generate_pair(self):
        # print("RX_status: ", self.RX_status)
        # print("TX_status: ", self.TX_status)
        # print("IS_status: ", self.IS_status)
        
        # 1. 生成节点顺序
        switch_status = []
        switch_type = []
        for i in range(self.node_num):
            RX = self.RX_status[i]
            TX = self.TX_status[i]
            IS = self.IS_status[i]
            if self.direction == 'TIR':
                switch_status += [TX, IS, RX]
                switch_type += ['TX', 'IS', 'RX']
            else:
                switch_status += [RX, IS, TX]
                switch_type += ['RX', 'IS', 'TX']
        # print("switch_type: ", switch_type)
        # print("switch_status: ", switch_status)
        
        # 2. 生成互连关系
        pairs = {}
        dst_list = []
        src = None
        for i in range(3*self.node_num):
            type = switch_type[i]
            status = switch_status[i]
            if type == 'RX' and status == 1:
                dst_list.append(self.node_list[i//3])
            if type == 'TX' and status == 1:
                src = self.node_list[i//3]
            if type == 'IS' and status == 0:
                if src != None and len(dst_list) != 0:
                    pairs[src] = dst_list
                src = None
                dst_list = []
        if src != None and src not in pairs and len(dst_list) != 0:
            pairs[src] = dst_list
        
        return pairs

class DYTopology:
    def __init__(self, size_w, size_h, edge_wire_num, BW):
        self.name = 'w{}_h{}_n{}_b{}'.format(size_w, size_h, edge_wire_num, BW)
        self.size_w = size_w
        self.size_h = size_h
        self.edge_wire_num = edge_wire_num
        self.bandwidth = BW
        
        self.horizon_node_dict = {}
        self.vertical_node_dict = {}
        self.node_list = []
        
        self.pairs = {}
    
    def get_node_list(self):
        for i in range(self.size_w):
            self.vertical_node_dict[i] = []
        for j in range(self.size_h):
            self.horizon_node_dict[j] = []
        for i in range(self.size_w):
            for j in range(self.size_h):
                id = i*self.size_h + j
                self.node_list.append(id)
                self.vertical_node_dict[i].append(id)
                self.horizon_node_dict[j].append(id)
    
    def set_wire(self, name, size, direction, node_list):
        wire = DYWire(name, size, direction, self.bandwidth)
        wire.set_node_list(node_list)
        wire.random_wire()
        wire.record_status(self.name + '/')
        pairs = wire.generate_pair()
        return wire, pairs
    
    def set_topology(self):
        west_wire_list  = []
        east_wire_list  = []
        north_wire_list = []
        south_wire_list = []
        
        # West + East
        for i in range(self.size_h):
            for j in range(self.edge_wire_num):
                w_name = 'W{}_{}'.format(i, j)
                e_name = 'E{}_{}'.format(i, j)
                w_wire, w_pairs = self.set_wire(w_name, self.size_w, 'RIT', self.horizon_node_dict[i])
                e_wire, e_pairs = self.set_wire(e_name, self.size_w, 'TIR', self.horizon_node_dict[i])
                west_wire_list.append(w_wire)
                east_wire_list.append(e_wire)
            
        # South + North
        for i in range(self.size_w):
            for j in range(self.edge_wire_num):
                s_name = 'S{}_{}'.format(i, j)
                n_name = 'N{}_{}'.format(i, j)
                s_wire, s_pairs = self.set_wire(s_name, self.size_h, 'TIR', self.vertical_node_dict[i])
                n_wire, n_pairs = self.set_wire(n_name, self.size_h, 'RIT', self.vertical_node_dict[i])
                south_wire_list.append(s_wire)
                north_wire_list.append(n_wire)


if __name__ == '__main__':
    size_w = 6
    size_h = 4
    edge_wire_num = 4
    BW = 128
    DYT = DYTopology(size_w, size_h, edge_wire_num, BW)
    DYT.get_node_list()
    DYT.set_topology()
            
'''
if __name__ == '__main__':
    direction_list = ['RIT', 'TIR']
    i = random.randint(0, 1)
    print("direction: ", direction_list[i])
    wire = DYWire("Mesh_H1", 6, direction_list[i])
    # wire.random_wire()
    wire.Bus_status(2)
    wire.record_status()
    # wire.wire_status_load("H1")
    wire.wire_print()
    wire.generate_pair()
'''