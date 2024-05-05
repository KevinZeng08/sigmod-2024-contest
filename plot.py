import numpy as np
import matplotlib.pyplot as plt

nq = 1_000_000
topk = 100
ef = 800
bf_threshold = 0.05
category_index_threshold = 0.05
strategy = 'incr_build'

def get_query_stats():
    stats = []
    with open('query_stats.bin', 'rb') as f:
        for _ in range(nq):
            stats.append(np.frombuffer(f.read(4 * 3), dtype=np.float32))
    return stats

def get_recall():
    recall = []
    with open('recall1.bin', 'rb') as f:
        for _ in range(nq):
            recall.append(np.frombuffer(f.read(4), dtype=np.int32))
    
    return recall

stats = get_query_stats()
recall = get_recall()

def plot_selectivity_recall(stats, recall):
    x = []
    y = []
    for i in range(nq):
        x.append(stats[i][1])
        y.append(recall[i])
    plt.scatter(x, y, s=1)
    plt.xlabel('Selectivity')
    plt.ylabel('Recall@100')
    plt.title('Selectivity vs Recall@100')
    plt.ylim(80,100)
    plt.savefig(f'selectivity_recall_ef{ef}_sel{bf_threshold}_cat_{category_index_threshold}.png')
    plt.clf()

def plot_bruteforce_recall(stats, recall, threshold=bf_threshold):
    # 统计一下在selectivity<0.2情况下，每个recall（1，0.99）的query个数
    counter = {}
    for i in range(nq):
        if stats[i][1] < threshold:
            r = recall[i].item()
            if r in counter:
                counter[r] += 1
            else:
                counter[r] = 1
    # plot 
    x = list(counter.keys())
    y = list(counter.values())
    plt.bar(x, y)
    plt.xlim(90, 100)
    # 标出频数
    for a, b in zip(x, y):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Query Count')
    plt.title('Bruteforce Recall')
    plt.savefig(f'bruteforce_recall_sel{bf_threshold}_cat{category_index_threshold}.png')
    plt.clf()

def plot_graph_recall(stats, recall, threshold=bf_threshold):
    # 统计一下在selectivity<0.2情况下，每个recall（1，0.99）的query个数
    counter = {}
    for i in range(nq):
        if stats[i][1] >= threshold:
            r = recall[i].item()
            if r in counter:
                counter[r] += 1
            else:
                counter[r] = 1
    # plot 
    x = list(counter.keys())
    y = list(counter.values())
    plt.bar(x, y)
    plt.xlim(90, 100)
    for a,b in zip(x, y):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Query Count')
    plt.title('Graph Search Recall')
    plt.savefig(f'graph_recall_ef_{ef}_sel{bf_threshold}_cat_{category_index_threshold}.png')
    plt.clf()

def plot_selectivity_time(stats):
    x = []
    y = []
    for i in range(nq):
        x.append(stats[i][1])
        y.append(stats[i][2])
    plt.scatter(x, y, s=1)
    plt.xlabel('Selectivity')
    plt.ylabel('Time (ms)')
    plt.title('Selectivity vs Time')
    plt.savefig(f'selectivity_time_ef{ef}_sel{bf_threshold}_cat_{category_index_threshold}.png')
    plt.clf()

def plot_selectivity(stats):
    # plot distribution of selectivity
    # [0,0.1] [0.1,0.2] ... [0.9,1

    # 假设selectivity是一个包含选择性值的列表
    selectivity = [stat[1] for stat in stats]

    # 创建直方图
    plt.hist(selectivity, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # 设置标题和标签
    plt.title('Distribution of Selectivity')
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')
    # 显示纵坐标的值
    plt.yticks(range(0, 100000, 10000))

    # 显示图形
    plt.savefig('selectivity.png')
    plt.clf()

def plot_type_selectivity_frequency(stats):

    s1 = [stat[1] for i, stat in enumerate(stats) if stat[0] == 1]
    s2 = [stat[1] for i, stat in enumerate(stats) if stat[0] == 2]
    s3 = [stat[1] for i, stat in enumerate(stats) if stat[0] == 3]

    counts, bins, patches = plt.hist(s1, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], alpha=0.5, label='Type 1')
    plt.title('Distribution of Selectivity of Type 1')
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    # 给出每个直方的频数
    for i in range(len(counts)):
        plt.text(bins[i], counts[i], str(counts[i]), ha='center', va='bottom', fontsize=10)
    plt.savefig('type1_selectivity.png')
    plt.clf()

    counts, bins, patches = plt.hist(s2, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], alpha=0.5, label='Type 2')
    plt.title('Distribution of Selectivity of Type 2')
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    for i in range(len(counts)):
        plt.text(bins[i], counts[i], str(counts[i]), ha='center', va='bottom', fontsize=10)
    plt.savefig('type2_selectivity.png')
    plt.clf()

    counts, bins, patches = plt.hist(s3, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], alpha=0.5, label='Type 3')
    plt.title('Distribution of Selectivity of Type 3')
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    for i in range(len(counts)):
        plt.text(bins[i], counts[i], str(counts[i]), ha='center', va='bottom', fontsize=10)
    plt.savefig('type3_selectivity.png')
    plt.clf()

def plot_type3_selectivity_under_category():
    # 输出type3的query在对category首先partition后的selectivity
    s3 = [stat[1] for i, stat in enumerate(stats) if stat[0] == 3]

    counts, bins, patches = plt.hist(s3, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], alpha=0.5, label='Type 3')
    plt.title('Distribution of Selectivity of Type 3 under category')
    plt.xlabel('Selectivity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    for i in range(len(counts)):
        plt.text(bins[i], counts[i], str(counts[i]), ha='center', va='bottom', fontsize=10)
    plt.savefig('type3_selectivity_under_category.png')

def plot_query_type_selectivity_recall():
    # type = 0
    freq = {}
    for i in range(nq):
        if stats[i][0] == 0:
            r = recall[i].item()
            if r in freq:
                freq[r] += 1
            else:
                freq[r] = 1

    x = list(freq.keys())
    y = list(freq.values())
    plt.bar(x, y)
    plt.xlim(90, 100)
    for a,b in zip(x, y):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Query Count')
    plt.title('Type 0 Recall')
    plt.savefig(f'type0_recall_{strategy}_ef_{ef}_sel{bf_threshold}.png')
    plt.clf()

    # type = 1
    freq = {}
    for i in range(nq):
        if stats[i][0] == 1:
            r = recall[i].item()
            if r in freq:
                freq[r] += 1
            else:
                freq[r] = 1
    
    x = list(freq.keys())
    y = list(freq.values())
    plt.bar(x, y)
    plt.xlim(90, 100)
    for a,b in zip(x, y):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Query Count')
    plt.title('Type 1 Recall')
    plt.savefig(f'type1_recall_{strategy}_ef_{ef}_sel{bf_threshold}.png')
    plt.clf()

    # type = 2
    freq = {}
    for i in range(nq):
        if stats[i][0] == 2:
            r = recall[i].item()
            if r in freq:
                freq[r] += 1
            else:
                freq[r] = 1
                
    x = list(freq.keys())
    y = list(freq.values())
    plt.bar(x, y)
    plt.xlim(90, 100)
    for a,b in zip(x, y):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Query Count')
    plt.title('Type 2 Recall')
    plt.savefig(f'type2_recall_{strategy}_ef_{ef}_sel{bf_threshold}.png')
    plt.clf()

    # type = 3
    freq = {}
    for i in range(nq):
        if stats[i][0] == 3:
            r = recall[i].item()
            if r in freq:
                freq[r] += 1
            else:
                freq[r] = 1

    x = list(freq.keys())
    y = list(freq.values())
    plt.bar(x, y)
    plt.xlim(90, 100)
    for a,b in zip(x, y):
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Query Count')
    plt.title('Type 3 Recall')
    plt.savefig(f'type3_recall_{strategy}_ef_{ef}_sel{bf_threshold}.png')
    plt.clf()

def plot_range_filter_selectivity_recall():
    # 统计type=2时，在selectivity为[0, 0.1, 0.2, 0.3, ..., 1.0]下，recall大于等于0.95的query个数的百分比
    recall2 = [recall[i] for i, stat in enumerate(stats) if stat[0] == 2]
    selectivity2 = [stat[1] for i, stat in enumerate(stats) if stat[0] == 2]

    threshold = 95
    ratios = np.zeros(10, dtype=np.float32)
    for i in range(10):
        start = i * 0.1
        end = (i + 1) * 0.1
        sel_recall = [recall2[i].item() for i, sel in enumerate(selectivity2) if start <= sel < end and sel>=bf_threshold]
        ratios[i] = len([r for r in sel_recall if r >= threshold]) / len(sel_recall)

    plt.plot(np.arange(0, 1.0, 0.1), ratios)
    plt.xlabel('Selectivity')
    plt.ylabel(f'Recall Ratio(>{threshold})')
    plt.title(f'Range Filter Selectivity vs Recall Ratio (>{threshold})')
    for i in range(10):
        plt.text(i * 0.1, ratios[i], f'{ratios[i]:.4f}', ha='center', va='bottom', fontsize=10)
    plt.savefig(f'type2_sel_rec_{strategy}_ef{ef}_{threshold}.png')

    
    

# plot_selectivity_recall(stats, recall)
# plot_selectivity_time(stats)
# plot_selectivity(stats)
# plot_bruteforce_recall(stats, recall)
# plot_graph_recall(stats, recall)
# plot_type_selectivity_frequency(stats)
# plot_type3_selectivity_under_category()
plot_query_type_selectivity_recall()
plot_range_filter_selectivity_recall()