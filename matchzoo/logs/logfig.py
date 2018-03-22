import glob
import os

import matplotlib.pyplot as plt
import sys
import re

seperator = ','  # '\t'

def reduce_result(x, interval):
    y = {}
    count = {}
    for iter, value in x:
        idx = iter / interval
        if idx not in y:
            y[idx] = 0.0
            count[idx] = 0
        y[idx] += value
        count[idx] += 1
    for idx in y:
        y[idx] /= count[idx]

    # convert dict to list
    idx = 0
    rtn = []
    while True:
        if idx in y:
            rtn.append(y[idx])
            idx += 1
        elif idx == 0:
            rtn.append(0)
            idx += 1
        else:
            break
    return rtn

def generate_log_data(filename, red_train = 1, red_valid = 1, red_test = 1):
    reduce_interval = {
        "train" : red_train,
        "valid" : red_valid,
        "test" : red_test
        }
    print filename
    
    # [Eval] @ test  epoch: 8, ndcg@3:0.303071  map:0.406644  ndcg@5:0.334777
    #pattern_raw = r"\[(.+?)\] \@ (.+?)  epoch: (.+?), (.+?)\:(.+?)  (.+?)\:(.+?)  (.+?)\:(.+?)  (.+?)\:(.+?)  (.+?)\:(.+?)\n"
    #[03-21-2018 04:12:39]   [Eval:test] Iter:96     ndcg@10=0.373602        ndcg@3=0.320856 map=0.403047    ndcg@5=0.329673
    pattern_raw = r"\[(.+?)\]\t\[(.+?):(.+?)\] Iter:(.+?)\t(.+?)=(.+?)\t(.+?)=(.+?)\t(.+?)=(.+?)\t(.+?)=(.+?)\n"
    #[03-21-2018 04:12:35]   [Train:train] Iter:96   loss=0.432692
    #pattern_raw = r"\[(.+?)\] \[(.+?):(.+?)\] Iter: (.+?), (.+?)=(.+?)\n"
    
    log_lines = {}
    for line in open(filename):
        m = re.match(pattern_raw, line)
        if m:
            tag = m.group(3)
            iter = int(m.group(4))
            node1 = m.group(5)
            value1 = float(m.group(6))
            node2 = m.group(7)
            value2 = float(m.group(8))
            node3 = m.group(9)
            value3 = float(m.group(10))
            node4 = m.group(11)
            value4 = float(m.group(12))
            if tag not in log_lines:
                log_lines[tag] = {}
            if node1 not in log_lines[tag]:
                log_lines[tag][node1] = []
            if node2 not in log_lines[tag]:
                log_lines[tag][node2] = []
            if node3 not in log_lines[tag]:
                log_lines[tag][node3] = []
            if node4 not in log_lines[tag]:
                log_lines[tag][node4] = []
            log_lines[tag][node1].append([iter, value1])
            log_lines[tag][node2].append([iter, value2])
            log_lines[tag][node3].append([iter, value3])
            log_lines[tag][node4].append([iter, value4])
            #print tag, iter, node4, value4

    for tag in log_lines:
        for node in log_lines[tag]:
            #print tag, node, len(log_lines[tag][node])
            log_lines[tag][node] = reduce_result(log_lines[tag][node], reduce_interval[tag])
            #print '11111', tag, node, len(log_lines[tag][node])
    
    return filename, log_lines

def draw_graph(fig_idx, log_data, tag, node, offset = 2):
    plt.figure(fig_idx, figsize=(20,10))
    plt.title('%s:%s' % (tag, node))
    plt.xlabel('iter')
    plt.ylabel(node)
    
    plot_list = []
    
    labels = []
    beg = 0
    end = -1
    if len(log_data) > 2:
        while(True):
            charbegin = ''
            idx = 1
            bTag = True
            for filename, log_data_one in log_data:
                if  idx == 1:
                    charbegin = filename[beg]
                elif charbegin != filename[beg]:
                    bTag = False
                    break
                idx += 1
            beg += 1
            if not bTag:
                break
        for filename, log_data_one in log_data:
            start = beg
            if beg != 0:
                start = beg - 1
            if end == -1:
                labels.append(filename[start:]+'$')
            else:
                labels.append(filename[start:end+1]+'$')
    else:
        for filename, log_data_one in log_data:
          labels.append(filename)


    idx = 0
    avg = 0
    for filename, log_data_one in log_data:
        p, = plt.plot(range(len(log_data_one[tag][node][offset:])), log_data_one[tag][node][offset:], 
                #label = labels[idx] + ' - ' + str(max(log_data_one[tag][node][offset:])))
                label = '.'.join(filename.split('/')[-1].split('.')[:-1]) + ' - ' + str(max(log_data_one[tag][node][offset:]) ))
        avg += max(log_data_one[tag][node][offset:])
        idx += 1
        plot_list.append(p)
    avg /= len(log_data)
    
    # plt.legend(plot_list, numpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.gca().get_yaxis().grid()
    plt.legend(plot_list, numpoints=1, bbox_to_anchor=(0, -0.18, 1., .102), loc=1,
           ncol=1, mode="expand", borderaxespad=0.)
    plt.show()
    print 'average : ',avg