#pragma once

#include <cmath>
#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
typedef char vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;
    unsigned int blocksize;
    unsigned int log_blocksize;
    unsigned int numblock;
    vl_type *changed;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }
    VisitedList(int numelements1,int log_blocksize) {
//        log_blocksize-=3;
        this->log_blocksize=log_blocksize;
        blocksize=1<<(log_blocksize-3);
        numblock=std::ceil((float)numelements1/blocksize);
        curV = -1;
        numelements = numblock*blocksize;
        mass = new vl_type[numelements];
        numblock=(numblock>>3)+1;
        changed = new vl_type[numblock];
        madvise(mass,numelements*sizeof(vl_type),MADV_HUGEPAGE);
//        madvise(changed,numblock*sizeof(vl_type),MADV_RANDOM);
    }

    inline void reset() {//this reset is NOT let someplace 0
        clear();
    }
//    inline void push_down(int upper_place) {
//        changed[upper_place>>3]|=(1<<(upper_place&7));
////        changed[upper_place]=1;
////        auto start = std::chrono::high_resolution_clock::now();
//        memset(mass+(upper_place*blocksize),0,blocksize);
////        auto end = std::chrono::high_resolution_clock::now();
////        memmasstime+=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//    }

    inline void clear() {
//        auto start = std::chrono::high_resolution_clock::now();
        memset(changed,0,numblock);
//        memset(mass,0,numelements);
//        auto end = std::chrono::high_resolution_clock::now();
//        memchangetime+=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    inline void set(int place) {
        int upper_place = place >> log_blocksize;
//        std::cout<<place<<' '<<(place>>3)<<' '<<upper_place<<std::endl;
        if(((changed[upper_place>>3]>>(upper_place&7))&1)==0){
//            push_down(upper_place);
            changed[upper_place>>3]|=(1<<(upper_place&7));
            memset(mass+(upper_place*blocksize),0,blocksize);
        }
        mass[place>>3]|=(1<<(place&7));
    }
    inline bool get(int place){
        int upper_place = place >> log_blocksize;
        if(((changed[upper_place>>3]>>(upper_place&7))&1)==0)return 0;
        return (mass[place>>3]>>(place&7))&1;
    }
    ~VisitedList() {
        delete[] mass;
        delete[] changed;
    }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;
    int logblocksize;
 public:
    VisitedListPool(int initmaxpools, int numelements1,int logblocksize1) {
        //todo:change block size here
//        logblocksize1=8;
        numelements1 = (numelements1>>3)+1;
        numelements = numelements1;
        logblocksize = logblocksize1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements,logblocksize1));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements,logblocksize);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
