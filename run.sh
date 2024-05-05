rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j${nproc}
cd ..
./build/hybrid_search