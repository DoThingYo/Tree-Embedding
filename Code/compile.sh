# These two programs are used to generate subset link prediction datasets for undirected and directed graphs, respectively
g++ -pthread -march=core2 -std=c++11 -O3 -o GEN_SUBSET_LP_DATA_U gen_subset_LP_data_u.cpp -I../include -g
g++ -pthread -march=core2 -std=c++11 -O3 -o GEN_SUBSET_LP_DATA_D gen_subset_LP_data_d.cpp -I../include -g





# These two programs are used to generate batch update edges
## Generate batch update Edges from intermediate snapshots for NC tasks
g++ -march=core2 -std=c++11 -O3 -o SPLIT_EDGES_NC split_edges_NC.cpp -I../include -g

## Generate batch Update update from intermediate snapshots for LP tasks
g++ -march=core2 -std=c++11 -O3 -o SPLIT_EDGES_LP split_edges_LP.cpp -I../include -g





# These two programs are used to generate batch update edges
g++ -pthread -march=core2 -std=c++11 -O3 -o RANDOM_SAMPLE_POINTS_NC Random_Sample_Points_NC.cpp -I../include -g

g++ -pthread -march=core2 -std=c++11 -O3 -o RANDOM_SAMPLE_POINTS_LP Random_Sample_Points_LP.cpp -I../include -g






gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_d_NC.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_D_NC -lstdc++ 

gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_u_NC.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_U_NC -lstdc++ 

gcc -O3 -m64 -I/usr/include/eigen3 -I../emhash/ -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_d_LP.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_D_LP -lstdc++

gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_u_LP.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_U_LP -lstdc++




