from config import *
import yaml
import os

## 
# router {router_id: router_latency}
# n2r_connection = {node_id:[router_id, link_latency]}
# r2r_connection = {src_node_id:{dst0_node_id:[src_port_name, dst_port_name, link_latency, weight], dst1_node_id::[src_port_name, dst_port_name, link_latency, weight]}}

class OurInterposerConnection:
    def __init__(self, name):
        self.name           = name
        self.routers        = {}
        self.n2r_connection = {}
        self.r2r_connection = {}
        
        self.mesh_id_list = []
        self.bus_id_list = []
    
    def makeConnection(self):
        self.getMemYummyNode()
        self.getRouter()
        self.getN2RConnection()
        self.getR2RConnection()
        self.yamlOut()
    
    def getMemYummyNode(self):
        self.yummy_nodes = []
        self.mem_nodes = []
        for x in range(mesh_row_x):
            id1 = x * mesh_col_y
            id2 = id1 + mesh_col_y - 1
            if x % 2 == 0:
                self.yummy_nodes.append(id1)
                self.yummy_nodes.append(id2)
            else:
                self.mem_nodes.append(id1)
                self.mem_nodes.append(id2)
        ''' 
        for x in range(mesh_row_x//2):
            id1_1 = x * mesh_col_y * 2
            id1_2 = id1_1 + mesh_col_y - 1
            id2_1 = id1_1 + mesh_col_y
            id2_2 = id1_2 + mesh_col_y
            self.yummy_nodes.append(id1_1)
            self.yummy_nodes.append(id1_2)
            self.mem_nodes.append(id2_1)
            self.mem_nodes.append(id2_2)
        ''' 
    
    def getRouter(self):
        id = 0
        for size in mesh_size_list:
            for _ in range(size):
                if id in self.yummy_nodes:
                    self.routers[id] = mesh_router_latency
                else:
                    self.routers[id] = mesh_router_latency
                self.mesh_id_list.append(id)
                id += 1
        for size in bus_size_list:
            for _ in range(size):
                self.routers[id] = bus_router_latency
                self.bus_id_list.append(id)
                id += 1
    
    def getN2RConnection(self):
        for id in self.mesh_id_list:
            if id in self.yummy_nodes:
                self.n2r_connection[id] = [id, 0]
            else: 
                self.n2r_connection[id] = [id, mesh_link_latency]
        for id in self.bus_id_list:
            self.n2r_connection[id] = [id, bus_link_latency]
    
    def getR2RConnection(self):
        # create compute chiplet connection, 4x4 Mesh
        mesh_col_dict = {}
        mesh_row_dict = {}
        for i in range(mesh_row_x):
            mesh_row = []
            for j in range(mesh_col_y):
                id = i*mesh_col_y + j
                mesh_row.append(id)
                if j not in mesh_col_dict:
                    mesh_col_dict[j] = []
                mesh_col_dict[j].append(id)
            mesh_row_dict[i] = mesh_row
        
        # MESH: EAST WEST Connection
        for nodes in mesh_row_dict.values():
            n_max = len(nodes)
            for i, node in enumerate(nodes):
                node_1 = node
                if i+1 >= n_max:
                    pass
                else:
                    node_2 = nodes[i+1]
                    if node_2 in self.yummy_nodes or node_1 in self.yummy_nodes:
                        latency = 0
                    else:
                        latency = mesh_link_latency
                    if node_1 not in self.r2r_connection:
                        self.r2r_connection[node_1] = {}
                    self.r2r_connection[node_1][node_2] = ["East", "West", latency, mesh_weight_x]
                    
                    if node_2 not in self.r2r_connection:
                        self.r2r_connection[node_2] = {}
                    self.r2r_connection[node_2][node_1] = ["West", "East", latency, mesh_weight_x]
            
        # MESH: South North Connection
        for nodes in mesh_col_dict.values():
            n_max = len(nodes)
            for i, node in enumerate(nodes):
                node_1 = node
                if i+1 >= n_max:
                    pass
                else:
                    node_2 = nodes[i+1]
                    if node_1 not in self.r2r_connection:
                        self.r2r_connection[node_1] = {}
                    self.r2r_connection[node_1][node_2] = ["North", "South", mesh_link_latency, mesh_weight_y]
                    
                    if node_2 not in self.r2r_connection:
                        self.r2r_connection[node_2] = {}
                    self.r2r_connection[node_2][node_1] = ["South", "North", mesh_link_latency, mesh_weight_y]

        # MESH: MEM 2 COMPUTE
        '''
        MEM_nodes = self.mem_nodes
        for i, mem_node in enumerate(MEM_nodes):
            if i % 2 == 0:
                # left mem
                compute_id_1 = i * MESH_SIZE_X
                compute_id_2 = compute_id_1 + MESH_SIZE_X
                self.r2r_connection[mem_node] = {compute_id_1:["East", "West", mesh_link_latency, mesh_weight_x], compute_id_2:["North", "West", mesh_link_latency, mesh_weight_y]}
                self.r2r_connection[compute_id_1][mem_node] = ["West", "East", mesh_link_latency, mesh_weight_x]
                self.r2r_connection[compute_id_2][mem_node] = ["West", "North", mesh_link_latency, mesh_weight_x]
            else:
                # right mem
                compute_id_1 = i * MESH_SIZE_X - 1
                compute_id_2 = compute_id_1 + MESH_SIZE_X
                self.r2r_connection[mem_node] = {compute_id_1:["West", "East", mesh_link_latency, mesh_weight_x], compute_id_2:["North", "East", mesh_link_latency, mesh_weight_y]}
                self.r2r_connection[compute_id_1][mem_node] = ["East", "West", mesh_link_latency, mesh_weight_x]
                self.r2r_connection[compute_id_2][mem_node] = ["East", "North", mesh_link_latency, mesh_weight_x]
        '''
        
    def yamlOut(self):
        line = "routers: \n"
        for id, latency in self.routers.items():
            line += "  {}: {}\n".format(id, latency)
        
        line += "n2r_connections: \n"
        for node_id, list in self.n2r_connection.items():
            line += "  {}: \n".format(node_id)
            for item in list:
                line += "    - {}\n".format(item)
        
        line += "r2r_connection: \n"
        for src_node, r_dict in self.r2r_connection.items():
            line += "  {}: \n".format(src_node)
            for dst_node, r_list in r_dict.items():
                line += "    {}: \n".format(dst_node)
                for item in r_list:
                    line += "      - {}\n".format(item)
        
        out_dir = './topology_out/'
        file = './topology_out/{}.yaml'.format(self.name)
        if os.path.exists(out_dir) == False:
            os.mkdir(out_dir)
        r_f = open(file, 'w')
        print(line, file=r_f)
        r_f.close()
          
if __name__ == '__main__':
    IC = OurInterposerConnection("mesh_wxy")
    IC.makeConnection()
    