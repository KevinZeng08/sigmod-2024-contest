#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
using namespace std;

struct QueryStats {
    float type, selectivity, time;
};
constexpr int nq = 1000000;
void get_querydata(vector<QueryStats>&q)
{
    q.resize(nq);
    std::ifstream inFile("query_stats.bin", std::ios::binary); // 以二进制模式打开文件

    // 获取文件大小
    inFile.seekg(0, std::ios::end);
    std::streamsize size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    inFile.read(reinterpret_cast<char*>(buffer.data()), size); // 从文件中读取数据到缓冲区
    int i=0;
    for (const auto& value : buffer) {
        if(i%3==0)q[i/3].type=value;
        if(i%3==1)q[i/3].selectivity=value;
        if(i%3==2)q[i/3].time=value;
        i++;
    }

    inFile.close(); // 关闭文件
}

void get_recall(vector<int>& recall) {
    recall.resize(nq);

    std::ifstream in("recall1.bin", std::ios::binary); // 以二进制模式打开文件

    in.read(reinterpret_cast<char*>(recall.data()), nq * sizeof(int)); // 从文件中读取数据到缓冲区
}

void get_query_category(vector<int32_t>& categories) {
    categories.resize(nq);

    std::ifstream in("categories.bin", std::ios::binary); // 以二进制模式打开文件

    in.read(reinterpret_cast<char*>(categories.data()), nq * sizeof(int32_t)); // 从文件中读取数据到缓冲区
    in.close();
}

int main()
{
    vector<QueryStats> q;
    vector<int> recall;
    vector<int32_t> categories;
    get_querydata(q);
    get_recall(recall);
    get_query_category(categories);
    int bf_count = 0;
    int bf_error = 0;
    int graph_count = 0;
    int total_recall = 0;
    int cat_cnt = 0;
    vector<pair<int, int>> recall_count(4);
    unordered_map<int,pair<int, pair<float, int>>> timestamp_map; // timestamp -> (count, (time, recall))
    vector<pair<int, float>> type_time(4);
    for(int i=0;i<nq;i++){
        // if (q[i].selectivity < 0.2) {
            // double bf_time = 0.0;
            // if (recall[i] < 1) {
            //     bf_time += q[i].time / 0.001;
            //     cout << q[i].type << " " << q[i].selectivity << " " << q[i].time << endl;
            //     bf_error++;
            // }
            // bf_count++;
            // cout<<q[i].type<<" "<<q[i].selectivity<<" "<<q[i].time<<" "<<q[i].recall<<endl;
        // }
        if (q[i].selectivity >= 0.05) {
            graph_count++;
            total_recall += recall[i];
        }

        // 统计所有type的recall
        if(q[i].type==0) {
            recall_count[0].first++;
            recall_count[0].second+=recall[i];
        }
        if(q[i].type==1) {
            recall_count[1].first++;
            recall_count[1].second+=recall[i];
        }
        if(q[i].type==2) {
            recall_count[2].first++;
            recall_count[2].second+=recall[i];
        }
        if(q[i].type==3) {
            recall_count[3].first++;
            recall_count[3].second+=recall[i];
        }
        
        // 统计不同sel下的type2的time和recall
        // if (q[i].selectivity < 0.05) continue; // 只统计图搜
        // if (q[i].type == 2) {
        //     float sel = q[i].selectivity;
        //     int t = ceil(sel * 10);
        //     timestamp_map[t].first++;
        //     timestamp_map[t].second.first += q[i].time;
        //     timestamp_map[t].second.second += recall[i];
        // }

        // 统计type的平均时间
    }
    // for (int i = 1; i <= 10; ++i) {
    //     if (timestamp_map.find(i) != timestamp_map.end()) {
    //         printf("timestamp [%f, %f], count: %d, total time: %.4f, average time: %.4f, average recall: %.4f\n", (i - 1) * 0.1, i * 0.1, timestamp_map[i].first,
    //          timestamp_map[i].second.first, (float) timestamp_map[i].second.first / timestamp_map[i].first, (float)timestamp_map[i].second.second / timestamp_map[i].first);
    //     }
    // }
    // printf("type0 graph total time: %.4f, average time: %.4f\n", type_time[0].second, type_time[0].second / type_time[0].first);
    // printf("type1 graph total time: %.4f, average time: %.4f\n", type_time[1].second, type_time[1].second / type_time[1].first);
    // printf("type2 graph total time: %.4f, average time: %.4f\n", type_time[2].second, type_time[2].second / type_time[2].first);
    // printf("type3 graph total time: %.4f, average time: %.4f\n", type_time[3].second, type_time[3].second / type_time[3].first);
    for (int i = 0; i < 4; i++)
    {
        cout << (float)recall_count[i].second / recall_count[i].first << endl;
    }
    // cout << (float)total_recall / cat_cnt << endl;
}