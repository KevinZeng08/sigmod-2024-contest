
cmake_minimum_required (VERSION 3.12)
project (HybridSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Code intelligence use
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MKL is not allowed in the contest 
# #MKL Config
# if (MSVC)
#     # Only the DiskANN DLL and one of the tools need MKL libraries. Additionally, only a small part of MKL is used.
#     # Given that and given that MKL DLLs are huge, use static linking to end up with no MKL DLL dependencies and with
#     # significantly smaller disk footprint.
#     #
#     # The compile options are not modified as there's already an unconditional -DMKL_ILP64 define below
#     # for all architectures, which is all that's needed.
#     set(DISKANN_MKL_INCLUDE_DIRECTORIES "${DISKANN_MSVC_PACKAGES}/intelmkl.static.win-x64/lib/native/include")
#     set(DISKANN_MKL_LIB_PATH "${DISKANN_MSVC_PACKAGES}/intelmkl.static.win-x64/lib/native/win-x64")

#     set(DISKANN_MKL_LINK_LIBRARIES
#         "${DISKANN_MKL_LIB_PATH}/mkl_intel_ilp64.lib"
#         "${DISKANN_MKL_LIB_PATH}/mkl_core.lib"
#         "${DISKANN_MKL_LIB_PATH}/mkl_intel_thread.lib")
# else()
#     # expected path for manual intel mkl installs
#     set(POSSIBLE_OMP_PATHS "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so;/usr/lib/x86_64-linux-gnu/libiomp5.so;/opt/intel/lib/intel64_lin/libiomp5.so")
#     foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
#         if (EXISTS ${POSSIBLE_OMP_PATH})
#             get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
#         endif()
#     endforeach()

#     if(NOT OMP_PATH)
#         message(FATAL_ERROR "Could not find Intel OMP in standard locations; use -DOMP_PATH to specify the install location for your environment")
#     endif()
#     link_directories(${OMP_PATH})

#     set(POSSIBLE_MKL_LIB_PATHS "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;/opt/intel/mkl/lib/intel64/libmkl_core.so")
#     foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
#         if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
#             get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
#         endif()
#     endforeach()

#     set(POSSIBLE_MKL_INCLUDE_PATHS "/opt/intel/oneapi/mkl/latest/include;/usr/include/mkl;/opt/intel/mkl/include/;")
#     foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
#         if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH})
#             set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
#         endif()
#     endforeach()
#     if(NOT MKL_PATH)
#         message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_PATH to specify the install location for your environment")
#     elseif(NOT MKL_INCLUDE_PATH)
#         message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_INCLUDE_PATH to specify the install location for headers for your environment")
#     endif()
#     if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
#         set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
#     elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
#         set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
#     else()
#         message(FATAL_ERROR "Despite finding MKL, libmkl_def.so was not found in expected locations.")
#     endif()
#     link_directories(${MKL_PATH})
#     include_directories(${MKL_INCLUDE_PATH})

#     # compile flags and link libraries
#     add_compile_options(-m64 -Wl,--no-as-needed)
#     if (NOT PYBIND)
#         link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
#     else()
#         # static linking for python so as to minimize customer dependency issues
#         link_libraries(
#                 ${MKL_PATH}/libmkl_intel_ilp64.a
#                 ${MKL_PATH}/libmkl_intel_thread.a
#                 ${MKL_PATH}/libmkl_core.a
#                 ${MKL_DEF_SO}
#                 iomp5
#                 pthread
#                 m
#                 dl
#         )
#     endif()
# endif()

# add_definitions(-DMKL_ILP64)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -pthread -g -Wall -pthread -mavx2 -mfma")
else()
    # AVX2
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -march=native -mtune=native -O3 -mavx2 -pthread -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
    # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -std=c++17 -fPIC -mavx -mavx2 -mfma")
endif()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("pyglass")
add_subdirectory("pyglass")

add_executable(hybrid_search baseline.cpp)
#add_executable(hybrid_search testmap.cpp)
target_link_libraries(hybrid_search glass)

#add_executable(hybrid_graph heaptest.cpp)


# add_executable(test test_simd_distance.cpp)

# add_executable(count_selectivity count_selectivity.cpp hybrid_graph.cpp)
# target_link_libraries(count_selectivity glass)

#add_executable(hybrid_search generate_groundtruth.cpp)
#target_link_libraries(hybrid_search glass)