# SIGMOD Programming Contest 2024: Hybrid Vector Search

This repository is an open-source code for the [SIGMOD 2024 Programming Contest](http://sigmodcontest2024.eastus.cloudapp.azure.com/index.shtml), which challenges participants to design and implement efficient and scalable algorithms for **Hybrid Vector Search** in high dimensional space.

# Getting Started

## Prerequirement

- CMake >= 3.16
- G++ >= 9.4.0
- OpenMP >= 4.0

## Quick start
We have provided a script (`run.sh`) for compiling and running.
```
sh ./run.sh
```

## Build
Clone this repository to your local computer:
```
git clone https://github.com/KevinZeng08/sigmod-2024-contest.git
cd sigmod-2024-contest
```
Create the build directory for compilation.
```
rm -rf build
mkdir build
cd build
```
The **dataset path** has been **hardcode** in the `baseline.cpp` for the contest as [example solution](http://sigmodcontest2024.eastus.cloudapp.azure.com/baseline/baseline.tar.gz).
```
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j${nproc}
```

## Run
```
cd ..
./build/hybrid_search
```

# Team: Alaya (Southern University of Science and Technology, Zhejiang University)
- Members:

| Name        |    Email    |  Institutions  |
| ----------- | ----------- | -------------- |
| Long Xiang   | xiangl3@mail.sustech.edu.cn | Southern University of Science and Technology |
| Bowen Zeng   | kevinzeng0808@gmail.com | Zhejiang University|
| Yu Lei | devleiyu@foxmail.com | Zhejiang University |
| Yujun He | heyj2022@mail.sustech.edu.cn | Southern University of Science and Technology |
| Weijian Chen | weijianchen666@gmail.com | Southern University of Science and Technology |
| Yitao Zheng | 12210830@mail.sustech.edu.cn | Southern University of Science and Technology |
| Yanqi Chen | 12011319@mail.sustech.edu.cn | Southern University of Science and Technology |

# Copyright

Some source code adapted from [pyglass](https://github.com/zilliztech/pyglass/tree/master)

pyglass is under the MIT-licensed.
