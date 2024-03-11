import math
import copy
BWperLink = 8*2  # GB/s, 双向的, 一组线

def construct_Mesh(width, height, bw):
    linkBW = {}
    links = {}
    nodes = list(range(width*height))
    
    for h in range(height):
        for w in range(width):
            node = h*width + w
            east_n = h*width + w + 1
            north_n = (h+1)*width + w
            if h != height-1:
                # 不是最上面一排
                link1 = (node, north_n)
                link2 = (north_n, node)
                links[link1] = 0
                links[link2] = 0
                linkBW[link1] = bw
                linkBW[link2] = bw
            if w != width-1:
                # 不是最右边一列
                link1 = (node, east_n)
                link2 = (east_n, node)
                links[link1] = 0
                links[link2] = 0
                linkBW[link1] = bw
                linkBW[link2] = bw
    
    def route_XY(src, dst, W):
        src_x = src % W
        src_y = src // W
        dst_x = dst % W
        dst_y = dst // W
        
        diff_x = dst_x - src_x
        if diff_x == 0:
            step_x = 1
        else:
            step_x = diff_x // abs(diff_x)
        diff_y = dst_y - src_y
        if diff_y == 0:
            step_y = 1
        else:
            step_y = diff_y // abs(diff_y)
        path = []
        cur = src
        cur_x = src_x
        cur_y = src_y
        for i in range(0, diff_x, step_x):
            cur_x += step_x
            next = cur_y * W + cur_x
            path.append((cur, next))
            cur = next
        assert(cur_x == dst_x)
        cur_y = src_y
        cur_x = dst_x
        for i in range(0, diff_y, step_y):
            cur_y += step_y
            next = cur_y * W + cur_x
            path.append((cur, next))
            cur = next
        assert(cur == dst)
        return path
            
    route_table = {}
    for src_node in nodes:
        for dst_node in nodes:
            if src_node == dst_node:
                pass
            else:
                path = route_XY(src_node, dst_node, width)
                route_table[(src_node, dst_node)] = path
                
    return route_table, linkBW, links

def construct_CMesh(Mesh_w, Mesh_h, group_size, bw):
    group_num = Mesh_w * Mesh_h
    node_num = Mesh_w * Mesh_h * group_size
    total_nodes = list(range(node_num))
    groups = {}
    node_2_group = {}
    for g in range(group_num):
        central_node_id = g + node_num
        groups[central_node_id] = []
        for i in range(group_size):
            node_id = i + g * group_size
            groups[central_node_id].append(node_id)
            node_2_group[node_id] = central_node_id
    
    links = {}
    linkBW = {}
    for central_node_id, nodes in groups.items():
        for node in nodes:
            link1 = (central_node_id, node)
            link2 = (node, central_node_id)
            links[link1] = 0
            links[link2] = 0
            linkBW[link1] = bw
            linkBW[link2] = bw
    
    mesh_route_table, mesh_linkBW, mesh_links = construct_Mesh(Mesh_w, Mesh_h, bw)
    for (src, dst) in mesh_links:
        link = (src+node_num, dst+node_num)
        links[link] = 0
        linkBW[link] = bw
    
    route_table = {}
    for src in total_nodes:
        for dst in total_nodes:
            if src == dst:
                pass
            else:
                src_central_id = node_2_group[src]
                dst_central_id = node_2_group[dst]
                if src_central_id == dst_central_id:
                    path = [(src, dst_central_id), (dst_central_id, dst)]
                else:
                    mesh_path = mesh_route_table[(src_central_id-node_num,dst_central_id-node_num)]
                    path = [((src, src_central_id))]
                    for (m_src, m_dst) in mesh_path:
                        path.append((m_src+node_num, m_dst+node_num))
                    path.append((dst_central_id, dst))
                route_table[(src, dst)] = path
    return route_table, linkBW, links
       
def mesh_comm(pairs, route_table, linkBW, links):
    for src, packets in pairs.items():
        for pid, packet in packets.items():
            dstList = packet[0]
            commNum = packet[1]
            unique_path = []
            for dst in dstList:
                path = route_table[(src, dst)]
                for link in path:
                    if link in unique_path:
                        pass
                    else:
                        links[link] += commNum
                        # unique_path.append(link)
    
    max_utilization = 0
    max_link = None
    for link in links:
        links[link] /= linkBW[link]
        if links[link] > max_utilization:
            max_utilization = links[link]
            max_link = link
    
    return max_utilization, max_link

def Cmesh_comm(pairs, route_table, linkBW, links):
    for src, packets in pairs.items():
        for pid, packet in packets.items():
            dstList = packet[0]
            commNum = packet[1]
            # 考虑多播
            path_multicast = []
            for dst in dstList:
                path = route_table[(src, dst)]
                for link in path:
                    if link in path_multicast:
                        pass
                    else:
                        links[link] += commNum
                        path_multicast.append(link)
    
    max_utilization = 0
    max_link = None
    for link in links:
        links[link] /= linkBW[link]
        if links[link] > max_utilization:
            max_utilization = links[link]
            max_link = link
    
    return max_utilization, max_link

def our_comm(pairs, bw):
    data_num = {}
    datas = []
    for src, packets in pairs.items():
        for pid, packet in packets.items():
            dstList = packet[0]
            commNum = packet[1]
            tag = packet[2]
            if tag not in data_num:
                if tag == 'out':
                    data_num[tag] = commNum*2
                    datas.append(commNum*2)
                else:
                    dst_num = len(dstList)
                    if dst_num == 1:
                        data_num[tag] = commNum*2
                        datas.append(commNum*2)
                    else:
                        data_num[tag] = commNum
                        datas.append(commNum)
            else:
                pass
    
    assert (len(datas)==3)
    datas.sort()
    datas[-1] /= 2
    datas.sort()
    return datas[-1] // bw

def group_multicast(PPIX, PK, comm_num1, comm_num2, mem_num):
    ifmap_tile_mem = []
    weight_tile_mem = []
    ifmap_num = PPIX * comm_num1
    weight_num = PK * comm_num2
    # MEM放数
    mem_ifmap = {}
    mem_weight = {}
    for i in range(mem_num):
        mem_ifmap[i] = [math.ceil(ifmap_num/mem_num)*i, math.ceil(ifmap_num/mem_num)*(i+1)-1]
        mem_weight[i] = [math.ceil(weight_num/mem_num)*i, math.ceil(weight_num/mem_num)*(i+1)-1]
    # 寻找tile所在Mem
    def get_tile_mem(mem_partition, size, tile_num):
        tile_mems = {}
        mem_id = 0
        for i in range(tile_num):
            tile_mems[i] = {}
            t_begin = i * size
            t_end = i * size + size - 1
            cur = t_begin
            while (1):
                mem_end = mem_partition[mem_id][1]
                if mem_end > t_end:
                    tile_mems[i][mem_id] = t_end - cur + 1
                    break
                elif mem_end == t_end:
                    tile_mems[i][mem_id] = t_end - cur + 1
                    mem_id += 1
                    break
                else:
                    num = mem_end - cur + 1
                    tile_mems[i][mem_id] = num
                    mem_id += 1
                    cur = mem_partition[mem_id][0]        
        return tile_mems      

    ifmap_mems = get_tile_mem(mem_ifmap, comm_num1, PPIX)
    weight_mems = get_tile_mem(mem_weight, comm_num2, PK)
    
    ifmap_pairs = {}
    for tile in range(PPIX):
        dstList = []
        for j in range(PK):
            dstList.append(tile*PK+j)
        mems = ifmap_mems[tile]
        for mem, num in mems.items():
            if mem not in ifmap_pairs:
                ifmap_pairs[mem] = []
            ifmap_pairs[mem].append([dstList, num])
    
    weight_pairs = {}
    for tile in range(PK):
        dstList = []
        for j in range(PPIX):
            dstList.append(tile + j * PK)
        mems = weight_mems[tile]
        for mem, num in mems.items():
            if mem not in weight_pairs:
                weight_pairs[mem] = []
            weight_pairs[mem].append([dstList, num])
    
    return ifmap_pairs, weight_pairs
     
def gen_pairs(ifmapD, weightD, ofmapD, parallel, topology):
    PPIX = parallel[0] * parallel[1]
    PK = parallel[2]
    
    npus = PPIX * PK
    drams = math.ceil(npus / 4)
    
    mem_nodes = []
    npu_nodes = []
    
    # ifmap
    ifmap_pairs, weight_pairs = group_multicast(PPIX, PK, ifmapD, weightD, drams)
    if topology == "mesh" or topology == "ours":
        mem_nodes = [0, 5, 12, 17]
        npu_nodes = [1, 2, 7, 8, 3, 4, 9, 10, 13, 14, 19, 20, 15, 16, 21, 22]
        npu2mem = {1:0, 2:0, 7:0, 8:0, 3:5, 4:5, 9:5, 10:5, 13:12, 14:12, 19:12, 20:12, 15:17, 16:17, 21:17, 22:17}
    elif topology == 'cmesh':
        mem_nodes = [0, 5, 10, 15]
        npu_nodes = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
        npu2mem = {1:0, 2:0, 3:0, 4:0, 6:5, 7:5, 8:5, 9:5, 11:10, 12:10, 13:10, 14:10, 16:15, 17:15, 18:15, 19:15}    
    pairs = {}
    for mem_id, ipairs in ifmap_pairs.items():
        mem = mem_nodes[mem_id]
        pairs[mem] = {}
        pid = 0
        for pair in ipairs:
            dstList = pair[0]
            num = pair[1]
            dsts = []
            for dst in dstList:
                dsts.append(npu_nodes[dst])
            pairs[mem][pid] = [dsts, num, 'in']
            pid += 1
        wpairs = weight_pairs[mem_id]
        for pair in wpairs:
            dstList = pair[0]
            num = pair[1]
            dsts = []
            for dst in dstList:
                dsts.append(npu_nodes[dst])
            pairs[mem][pid] = [dsts, num, 'weight']
            pid += 1
    for n in range(npus):
        npu = npu_nodes[n]
        mem = npu2mem[npu]
        assert (npu not in pairs)
        pairs[npu] = {0:[[mem], ofmapD, 'out']}
    return pairs

def comm_performance(ifmapD, weightD, ofmapD, parallel, topology, route_table, linkBW, links):
    pairs = gen_pairs(ifmapD, weightD, ofmapD, parallel, topology)
    linkss = copy.deepcopy(links)
    if topology == 'mesh':
        perf, link = mesh_comm(pairs, route_table, linkBW, linkss)
    elif topology == 'ours':
        perf = our_comm(pairs, BWperLink)
    else:
        perf, link = Cmesh_comm(pairs, route_table, linkBW, linkss)
    
    return perf

if __name__ == '__main__':
    '''
    Mesh_w = 2
    Mesh_h = 2
    group_size = 5
    bw = 2
    route_table, linkBW, links = construct_CMesh(Mesh_w, Mesh_h, group_size, bw)
    print("route_table: ", route_table)
    print("linkBW: ", linkBW)
    '''
    ifmapD = 71456
    weightD = 25088
    ofmapD = 86016
    parallel = [1, 1, 12, 1]
    topology = 'mesh'
    route_table, linkBW, links = construct_Mesh(6, 4, BWperLink * 2)
    print('mesh perf ', comm_performance(ifmapD, weightD, ofmapD, parallel, topology, route_table, linkBW, links))
    topology = 'cmesh'
    route_table, linkBW, links = construct_CMesh(2, 2, 5, BWperLink * 2)
    print('cmesh perf ', comm_performance(ifmapD, weightD, ofmapD, parallel, topology, route_table, linkBW, links))
    topology = 'ours'
    print('ours perf ', comm_performance(ifmapD, weightD, ofmapD, parallel, topology, None, None, None))
    