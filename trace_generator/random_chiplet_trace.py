import random
import os
import copy

size_x = 4
size_y = 4
size_y_extend = 5
route_TH = 2
# trace_num = random.randint(20, 40)

out_path = "/home/wangxy/workspace/chiplet/simulator_gem5/random_task_file/"
def get_nodes():
    nodes = []
    nodes_extend = []
    for y in range(size_y_extend):
        for x in range(size_x):
            node_id = y * size_x + x
            if y < size_y:
                nodes.append(node_id)
            nodes_extend.append(node_id)
    return nodes, nodes_extend

def random_trace(nodes):
    traces = {}
    for _ in range(trace_num):
        random.shuffle(nodes)
        src_node = nodes[0]
        dst_node = nodes[1]
        packet_num = random.randint(10, 300)
        if (src_node, dst_node) not in traces:
            traces[(src_node, dst_node)] = 0
        traces[(src_node, dst_node)] += packet_num
    return traces

def ours_topology(nodes, traces):
    
    def get_index(node_id):
        x = node_id % size_x
        y = node_id // size_x
        return x, y
    
    def get_node_id(x, y):
        id = x + y * size_x
        return id
    
    def get_route(src, dst, nodes_status):
        nodes_status_init = copy.deepcopy(nodes_status)
        # get index in x and y dimensions
        src_x, src_y = get_index(src)
        dst_x, dst_y = get_index(dst)
        
        # get direction in x and y dimensions
        if src_x < dst_x:
            dir_x = 'w'
        elif src_x > dst_x:
            dir_x = 'e'
        else:
            dir_x = (['w', 'e'])[random.randint(0, 1)]
        if src_y < dst_y:
            dir_y = 'n'
        elif src_y > dst_y:
            dir_y = 's'
        else:
            dir_y = (['n', 's'])[random.randint(0, 1)]

        cur_node = src
        find_tag = True
        route = []
        # nmr : non mininum route
        nmr_dis = route_TH
        while(cur_node != dst):
            cur_x, cur_y = get_index(cur_node)
            # horizon move
            nmr_tag_x = False
            if cur_x < dst_x:
                dir_x = 'w'
                horizon_node = get_node_id(cur_x+1, cur_y)
                h_status = nodes_status_init[horizon_node][dir_x]
            elif cur_x > dst_x:
                dir_x = 'e'
                horizon_node = get_node_id(cur_x-1, cur_y)
                h_status = nodes_status_init[horizon_node][dir_x]
            else:
                if dir_x == 'w':
                    horizon_node = get_node_id(cur_x+1, cur_y)
                    if cur_x + 1 >= size_x:
                        h_status = 0
                    else:
                        h_status = nodes_status_init[horizon_node][dir_x]
                else:
                    horizon_node = get_node_id(cur_x-1, cur_y)
                    if cur_x - 1 < 0:
                        h_status = 0
                    else:
                        h_status = nodes_status_init[horizon_node][dir_x]
                nmr_tag_x = True
            
            # vertical move
            nmr_tag_y = False
            if cur_y < dst_y:
                dir_y = 'n'
                vertical_node = get_node_id(cur_x, cur_y+1)
                v_status = nodes_status_init[vertical_node][dir_y]
            elif cur_y > dst_y:
                dir_y = 's'
                vertical_node = get_node_id(cur_x, cur_y-1)
                v_status = nodes_status_init[vertical_node][dir_y]
            else:
                if dir_y == 'n':
                    vertical_node = get_node_id(cur_x, cur_y+1)
                    if cur_y + 1 >= size_y:
                        v_status = 0
                    else:
                        v_status = nodes_status_init[vertical_node][dir_y]
                else:
                    vertical_node = get_node_id(cur_x, cur_y-1)
                    if cur_y - 1 < 0:
                        v_status = 0
                    else:
                        v_status = nodes_status_init[vertical_node][dir_y]
                nmr_tag_y = True

            # select
            if h_status == v_status and h_status == 0:
                find_tag = False
                break
            if nmr_tag_x == False and nmr_tag_y == False:
                if h_status >= v_status:
                    dst_node = horizon_node
                    dir  = dir_x
                else:
                    dst_node = vertical_node
                    dir = dir_y
            elif nmr_tag_y == True:
                assert(nmr_tag_x == False)
                if h_status > 0:
                    dst_node = horizon_node
                    dir = dir_x
                elif nmr_dis > 0:
                    dst_node = vertical_node
                    dir = dir_y
                    nmr_dis -= 1
                else:
                    find_tag = False
                    break
            else:
                assert(nmr_tag_y == False and nmr_tag_x == True)
                if v_status > 0:
                    dst_node = vertical_node
                    dir = dir_y
                elif nmr_dis > 0:
                    dst_node = horizon_node
                    dir = dir_x
                    nmr_dis -= 1
                else:
                    find_tag = False
                    break
            dir_col = {'w':'e', 'e':'w', 'n':'s', 's':'n'}
            dir_2 = dir_col[dir]
            nodes_status_init[dst_node][dir] -= 1
            nodes_status_init[cur_node][dir_2] -= 1
            route.append((cur_node, dst_node))
            cur_node = dst_node
        
        if find_tag == True:
            return route, nodes_status_init
        else:
            return None, nodes_status  
            
    nodes_status = {}
    for node in nodes:
        nodes_status[node] = {'s': 2, 'n': 2, 'e': 2, 'w': 2}

    sorted_traces = sorted(traces.items(), key = lambda x:x[1], reverse = True)
    traces_net_mesh = {}
    traces_net_sub = {}
    for ((src, dst), num) in sorted_traces:
        route, nodes_status = get_route(src, dst, nodes_status)
        if route == None:
            traces_net_mesh[(src, dst)] = num
        else:
            traces_net_sub[(src,dst)] = [num, route]
    max_num_sub = list(traces_net_sub.values())[0][0]
    traces_net_mesh[(size_x*size_y, size_x*size_y+1)] = max_num_sub
    return traces_net_mesh, traces_net_sub
    
def generate_task(nodes, traces, out_dir):
    
    nodes_inst = {}
    nodes_wait = {}
    for node in nodes:
        nodes_inst[node] = ''
        nodes_wait[node] = {1001:0, 1002:0, 1003:0}

    for (src_n, dst_n), num in traces.items():
        tag = random.randint(1001, 1003)
        nodes_inst[src_n] += "send {} {} {}\n".format(dst_n, num, tag)
        nodes_wait[dst_n][tag] += num
    for node, packet in nodes_wait.items():
        line = ''
        for tag, num in packet.items():
            if num == 0:
                pass
            else:
                line += "wait {} {} \n".format(num, tag)
        nodes_inst[node] += line
    
    for node in nodes_inst:
        nodes_inst[node] += 'finish\n'

    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)
    for node in nodes_inst:
        f = open('{}{}.txt'.format(out_dir, node), 'w')
        print(nodes_inst[node], file=f)
        f.close()

def custom_topology_trace_out(traces_mesh, traces, file_name):
    f = open(file_name, 'w')
    print("mesh trace -------------------", file=f)
    for (src, dst), num in traces_mesh.items():
        print("--- pairs {} : comm_num = {}".format((src, dst), num), file=f)
    print("custom topology trace --------", file=f)
    print("--- max packet num : ", list(traces.values())[0][0], file=f)
    for (src, dst), items in traces.items():
        num = items[0]
        route = items[1]
        print("--- pairs : {} ------- ".format((src, dst)), file=f)
        print("---    packet size = {}".format(num), file=f)
        print("---    route path = {}".format(route), file=f)
    f.close()
    
if __name__ == '__main__':
    sample_num = 100
    for trace_num in range(30, 39, 10):
        out_dir = out_path + 'trace_{}/'.format(trace_num)
        if os.path.exists(out_dir) == False:
            os.mkdir(out_dir)
        out_dir_base = out_dir + "base/"
        out_dir_ours = out_dir + "ours/"
        if os.path.exists(out_dir_base) == False:
            os.mkdir(out_dir_base)
        if os.path.exists(out_dir_ours) == False:
            os.mkdir(out_dir_ours)
        nodes, nodes_extand = get_nodes()
        for i in range(sample_num):
            task_name = 'T' + str(i+1)
            traces = random_trace(nodes)
            traces_mesh, traces_sub = ours_topology(nodes, traces)
            print("traces : ", traces)
            print("mesh traces : ", traces_mesh)
            print("custom traces : ", traces_sub)
            generate_task(nodes_extand, traces, '{}{}/'.format(out_dir_base, task_name))
            generate_task(nodes_extand, traces_mesh, '{}{}/'.format(out_dir_ours, task_name))
            custom_topology_trace_out(traces_mesh, traces_sub, '{}{}_costom_trace.txt'.format(out_dir, task_name))
    