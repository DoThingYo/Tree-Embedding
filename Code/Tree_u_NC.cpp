extern "C"
{
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"
#include "string.h"
}

#undef max
#undef min

#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <deque>
#include <vector>
#include <unordered_map>
#include "Graph_dynamic.h"



#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>

#include<assert.h>

#include<queue>

#include "ppr_computation_store_dynamic.h"


#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<atomic>


using namespace Eigen;

using namespace std;









bool maxScoreCmpTriplet(const Triplet<double>& a, const Triplet<double>& b){
  return a.value() > b.value();
}

const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");











void sparse_sub_svd_function_with_norm_computation(int d, int pass, 
int update_j, 
vector<long long int>& record_submatrices_nnz,
d_row_tree_mkl* subset_tree,
int largest_level_start_index,
int current_out_iter,
int lazy_update_start_iter)
{


  Eigen::SparseMatrix<double, 0, int> &submatrix = subset_tree->mat_mapping[update_j];


  mat* matrix_vec_t = subset_tree->hierarchy_matrix_vec[largest_level_start_index + update_j];


  SparseMatrix<double, RowMajor, long> ppr_matrix_temp(submatrix.rows(), submatrix.cols());


  ppr_matrix_temp = submatrix;


  long long int nnz = record_submatrices_nnz[update_j];

  assert(nnz < INT_MAX);
  auto hash_coo_time = chrono::system_clock::now();


  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix.rows(), submatrix.cols(), nnz);
  ppr_matrix_coo->nnz = nnz;


  long nnz_iter=0;
  double ppr_norm =0;

  for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
    for (SparseMatrix<double, RowMajor, long int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){
      double value1 = it.value();
      if(value1 == 0){

      }
      else{
        ppr_matrix_coo->rows[nnz_iter] = it.row() + 1;
        ppr_matrix_coo->cols[nnz_iter] = it.col() + 1;
        ppr_matrix_coo->values[nnz_iter] = value1;
        ppr_norm += ppr_matrix_coo->values[nnz_iter]*ppr_matrix_coo->values[nnz_iter];
        nnz_iter ++;
      }
    }
  }



  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration_cast<std::chrono::seconds>(coo_csr_time- hash_coo_time);

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;


  mat *U = matrix_new(submatrix.rows(), d);
  mat *S = matrix_new(d, 1);

  mat *V = matrix_new(submatrix.cols(), d);


  frPCA(ppr_matrix, &U, &S, &V, d, pass);


  mat * S_full = matrix_new(d, d);
  for(int i = 0; i < d; i++){
    matrix_set_element(S_full, i, i, matrix_get_element(S, i, 0));
  }


  matrix_matrix_mult(U, S_full, matrix_vec_t);
  


  if(current_out_iter >= lazy_update_start_iter - 1){
    auto norm_start_time = chrono::system_clock::now();


    mat * V_transpose_matrix = matrix_new(d, submatrix.cols());


    matrix_build_transpose(V_transpose_matrix, V);


    mat * final_matrix_shape_for_frobenius = matrix_new(submatrix.rows(), submatrix.cols());


    matrix_matrix_mult(matrix_vec_t, V_transpose_matrix, final_matrix_shape_for_frobenius);


    matrix_delete(V_transpose_matrix);


    V_transpose_matrix = NULL;

    for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
      for (SparseMatrix<double, RowMajor, long int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){
        double value1 = it.value();
        if(value1 == 0){

        }
        else{
          double XY_value = matrix_get_element(final_matrix_shape_for_frobenius, it.row(), it.col());
          matrix_set_element(final_matrix_shape_for_frobenius, it.row(), it.col(), XY_value - value1);
        }
      }
    }


    subset_tree->norm_B_Bid_difference_vec[update_j] = get_matrix_frobenius_norm(final_matrix_shape_for_frobenius);


    matrix_delete(final_matrix_shape_for_frobenius);


    final_matrix_shape_for_frobenius = NULL;


    auto norm_end_time = chrono::system_clock::now();
    auto elapsed_norm_time = chrono::duration_cast<std::chrono::seconds>(norm_end_time - norm_start_time);
  }


  ppr_matrix_temp.resize(0,0);
  ppr_matrix_temp.data().squeeze();




  matrix_delete(U);
  matrix_delete(S);
  matrix_delete(V);
  matrix_delete(S_full);
  U = NULL;
  S = NULL;
  V = NULL;
  S_full = NULL;
  
  csr_matrix_delete(ppr_matrix);

  ppr_matrix = NULL;

}











void dense_sub_svd_function(int d, int pass, 
mat* submatrix, 
mat* matrix_vec_t ){

    mat *U = matrix_new(submatrix->nrows, d);

    mat * S_full = matrix_new(d, d);

    mat *Vt = matrix_new(d, submatrix->ncols);

    truncated_singular_value_decomposition(submatrix, U, S_full, Vt, d);
    
    matrix_matrix_mult(U, S_full, matrix_vec_t);

    matrix_delete(U);

    matrix_delete(S_full);

    matrix_delete(Vt);

    U = NULL;
    S_full = NULL;
    Vt = NULL;

}













void get_Eigen_matrix_from_mkl(mat* mkl, MatrixXd &Eig){
  Eig.resize(mkl->nrows, mkl->ncols);
  for(int i = 0; i < mkl->nrows; i++){
    for(int j = 0; j < mkl->ncols; j++){
      Eig(i, j) = matrix_get_element(mkl, i, j);
    }
  }
}












int main(int argc,  char **argv){
  auto start_time = std::chrono::system_clock::now();
  srand((unsigned)time(0));
  char *endptr;

  string queryname = argv[1];

  string EBpath = argv[2];


  clock_t start = clock();
  double alpha = strtod(argv[3], &endptr);
  int pass = strtod(argv[4], &endptr);
  double backward_theta = strtod(argv[5], &endptr);
  int NUMTHREAD = strtod(argv[6], &endptr);

  int nParts = strtod(argv[7], &endptr);
  
  int hierarchy_n = strtod(argv[8], &endptr);

  int vertex_number = strtod(argv[9], &endptr);

  int dynamic_ppr_start_iter = strtod(argv[10], &endptr);

  int lazy_update_start_iter = strtod(argv[11], &endptr);

  double delta = strtod(argv[12], &endptr);

  vector<int> indicator(vertex_number, -1);

  double residuemax = backward_theta; // PPR error up to residuemax

  double reservemin = backward_theta; // Retain approximate PPR larger than reservemin

  cout << "alpha: " << alpha << ", residuemax: " << residuemax << ", reservemin: " << reservemin <<endl;
  cout << "nParts: "<< nParts << ", hierarchy_n: "<< hierarchy_n << ", vertex_number: "<<vertex_number<<endl; 
  cout << "dynamic_ppr_start_iter: "<<dynamic_ppr_start_iter<<", lazy_update_start_iter: "<<lazy_update_start_iter<<endl;
  cout << "delta: "<<delta<<endl;
  
  


  string config_path =  "DY_Dataset/" + queryname + "/config.txt";
  ifstream infile3( config_path.c_str() );

  
  int snapshots_number = 0;

  vector<string> shots_address_vec;

  string s2;
  while(getline(infile3, s2)) 
  { 
    shots_address_vec.push_back(s2);
    snapshots_number++;
  }
  
  cout<<"snapshots_number = "<<snapshots_number<<endl;



  int d = 128;

  unordered_map<int, int> row_index_mapping;
  vector<int> labeled_node_vec;

  int count_labeled_node = 0;
  ifstream infile2(shots_address_vec[0].c_str());

  int node_number;

  while (infile2>>node_number){ 
    indicator[node_number] = 0;
    labeled_node_vec.push_back(node_number);
    row_index_mapping[node_number] = count_labeled_node;
    count_labeled_node++;
  }

  cout<<"count_labeled_node = "<<count_labeled_node<<endl;




  MatrixXd U_all_shots((snapshots_number-1) * count_labeled_node, d);

  UGraph* g = new UGraph();

  g->initializeDynamicUGraph(vertex_number);

  unordered_map<int, vector<int>> vec_mapping;

  vector<int> vertex_mapping;

  vector<int> inner_group_mapping;

  vertex_mapping.resize(vertex_number);
  inner_group_mapping.resize(vertex_number);



  int common_group_size;

  common_group_size = vertex_number / nParts;

  int final_group_size = vertex_number - (nParts - 1) * common_group_size;

  cout<<"common_group_size = "<<common_group_size<<endl;

  for(int t = 0; t < vertex_number; t++){
    int index = t / common_group_size;
    if(index != nParts){
      vertex_mapping[t] = index;

    }
    else{
      vertex_mapping[t] = index - 1;
    }
  }

  for(int i = 0; i < nParts; i++){
    if(i != nParts - 1){
      for(int j = i * common_group_size; j < (i+1) * common_group_size; j++){
        vec_mapping[i].push_back(j);
      }
    }
    else{
      for(int j = 0; j < final_group_size; j++){
        vec_mapping[i].push_back(common_group_size * (nParts - 1) + j);
      }
    }
  }

  for(int i = 0; i < vec_mapping.size(); i++){
    for(int j = 0; j < vec_mapping[i].size(); j++){
      inner_group_mapping[vec_mapping[i][j]] = j;
    }
  }





  d_row_tree_mkl* subset_tree;

  subset_tree = new d_row_tree_mkl(count_labeled_node, d, nParts, hierarchy_n, vec_mapping,
  0, count_labeled_node);      


  vector<int> update_mat_record;


  update_mat_record.resize(nParts);

  for(int i = 0; i < update_mat_record.size(); i++){
      update_mat_record[i] = -1;
  }


  int upper_nnz = ceil(1 / residuemax / alpha);

  Queue * queue_list = new Queue[count_labeled_node];


  float **residue = new float* [count_labeled_node];
  float **pi = new float* [count_labeled_node];
  
  // vector<vector<column_tuple*>> pi_transpose_storepush;
  // pi_transpose_storepush.resize(count_labeled_node);

  bool **flags = new bool* [count_labeled_node];

  for(int i = 0; i < count_labeled_node; i++){
    residue[i] = new float[vertex_number];

    pi[i] = new float[vertex_number];

    flags[i] = new bool[vertex_number];

  }

  for(int i = 0; i < count_labeled_node; i++){
    //Create Queue
    queue_list[i] = Queue
    {
      //.arr = 
      (int*)malloc( sizeof(int) * (upper_nnz + 2) * 2 ),
      //.capacity = 
      (upper_nnz + 2) * 2,
      //.front =
      0,
      //.rear = 
      0
    };
  }


  mat* U_cur_iter;

  U_cur_iter = matrix_new(count_labeled_node, d);


  mat* U_without_S;

  U_without_S = matrix_new(count_labeled_node, d);

  MatrixXd U_Eigen;


  vector<long long int> record_submatrices_nnz(nParts, 0);


  for(int iter = 1; iter < snapshots_number; iter++){
    auto iter_start_time = chrono::system_clock::now();    
    cout<<"Current Snapshot Number = "<<iter<<endl;

    vector<pair<int, int>> edge_vec;
    cout<<"shots_address_vec[iter] = "<<shots_address_vec[iter]<<endl;
    g->inputDynamicGraph(shots_address_vec[iter].c_str(), edge_vec);


    if(lazy_update_start_iter != 100 && iter < lazy_update_start_iter - 1){
      continue;
    }



    vector<long long int> all_count(NUMTHREAD);

    vector<thread> refresh_threads;

    auto ppr_refresh_time = std::chrono::system_clock::now();


    for (int t = 1; t <= NUMTHREAD; t++){

      int start = (t-1)*(labeled_node_vec.size()/NUMTHREAD);
      int end = 0;
      if (t == NUMTHREAD){
        end = labeled_node_vec.size();
      } else{
        end = t*(labeled_node_vec.size()/NUMTHREAD);
      }

        refresh_threads.push_back(thread(nodegree_DenseUndirected_Refresh_PPR_initialization, start, end, g, residuemax, 
        alpha, std::ref(labeled_node_vec), 
        residue, pi, flags, queue_list, iter, vertex_number,
        common_group_size,
        nParts,
        std::ref(inner_group_mapping),
        subset_tree,
        std::ref(row_index_mapping),
        std::ref(indicator),
        dynamic_ppr_start_iter

        ));


    }

    for (int t = 0; t < NUMTHREAD ; t++){
      refresh_threads[t].join();
    }
    vector<thread>().swap(refresh_threads);


    auto finish_ppr_refresh_time = chrono::system_clock::now();
    auto elapsed_ppr_refresh_time = chrono::duration_cast<std::chrono::seconds>(finish_ppr_refresh_time - ppr_refresh_time);
    cout<< "Iter = "<<iter << ", refresh ppr time: "<< elapsed_ppr_refresh_time.count() << endl;



    cout << "ppr computation " << endl;
    auto ppr_start_time = std::chrono::system_clock::now();
    vector<thread> threads;

    for (int t = 1; t <= NUMTHREAD; t++){

        int start = (t-1)*(labeled_node_vec.size()/NUMTHREAD);
        int end = 0;
        if (t == NUMTHREAD){
            end = labeled_node_vec.size();
        } else{
            end = t*(labeled_node_vec.size()/NUMTHREAD);
        }

        threads.push_back(thread(nodegree_DenseDynamicForwardPush, start, end, g, residuemax, 
            alpha, std::ref(labeled_node_vec), 
            residue, 
            pi,
            flags, 
            queue_list,
            std::ref(row_index_mapping),
            common_group_size,
            nParts,
            subset_tree,
            std::ref(inner_group_mapping),
            std::ref(indicator)
        ));

    }

    for (int t = 0; t < NUMTHREAD ; t++){
      threads[t].join();
    }
    vector<thread>().swap(threads);


    auto start_ppr_matrix_time = chrono::system_clock::now();
    auto elapsed_ppr_time = chrono::duration_cast<std::chrono::seconds>(start_ppr_matrix_time - ppr_start_time);
    cout<< "Iter = "<<iter << ", computing ppr time: "<< elapsed_ppr_time.count() << endl;


    if(iter >= lazy_update_start_iter){

        vector<thread> threads_top_list;

        for(int i = 0; i < vec_mapping.size(); i++){    

            threads_top_list.push_back(thread(Log_sparse_matrix_entries_with_norm_computation, 
            i,    
            reservemin, 
            subset_tree,
            std::ref(vec_mapping),
            std::ref(update_mat_record),
            iter,
            delta,
            count_labeled_node,
            d,
            std::ref(record_submatrices_nnz)
            ));


        }

        int all_blocks_number = nParts;
        for (int t = 0; t < all_blocks_number; t++){
        threads_top_list[t].join();
        }


        vector<thread>().swap(threads_top_list);

        cout<<"With norm computation!"<<endl;


    }
    else{

        vector<thread> threads_top_list;

        for(int i = 0; i < vec_mapping.size(); i++){
    
            update_mat_record[i] = iter;

            threads_top_list.push_back(thread(Log_sparse_matrix_entries, 
            i,    
            reservemin, 
            subset_tree,
            std::ref(vec_mapping),
            std::ref(record_submatrices_nnz)
            ));

        }

        int all_blocks_number = nParts;
        for (int t = 0; t < all_blocks_number; t++){
        threads_top_list[t].join();
        }


        vector<thread>().swap(threads_top_list);

        cout<<"Without norm computation!"<<endl;

    }
    int largest_level_start_index = subset_tree->largest_level_start_index;

    int count_current_threads_number = 0;

    unordered_set<int> record_next_level;

    int sparse_unique_update_times = 0;

    for(int update_j = 0; update_j < nParts; update_j++){
      if(update_mat_record[update_j] == iter){
        if(subset_tree->mat_mapping[update_j].nonZeros() == 0){
          update_mat_record[update_j] = -1;
          continue;
        }

        sparse_unique_update_times++;

        sparse_sub_svd_function_with_norm_computation(d, pass, 
        update_j, 
        std::ref(record_submatrices_nnz),
        subset_tree,
        largest_level_start_index,
        iter,
        lazy_update_start_iter);



        int son_index;
        if((largest_level_start_index + update_j) % hierarchy_n == 0){
          son_index = (largest_level_start_index + update_j) / hierarchy_n - 1;
        }
        else{
          son_index = (largest_level_start_index + update_j) / hierarchy_n;
        }
        count_current_threads_number++;
        record_next_level.insert(son_index);
      }


    }
    cout<<"sparse_unique_update_times = "<<sparse_unique_update_times<<endl;

    if(sparse_unique_update_times == 0){
      cout<<"Error bounded! No SVD update for iter = "<<iter<<"!"<<endl;

      int iter_beginning = (iter-1) * count_labeled_node;
      // Take U_Eigen from last snapshot
      U_all_shots.block(iter_beginning, 0, count_labeled_node, d) = U_Eigen;
  
      continue;
    }

    bool end_while = false;

    while(!end_while){

      if(*record_next_level.begin() == 0){

        int first_index = 0 * hierarchy_n + 1;

        subset_tree->near_n_matrix_vec[0] = matrix_new(subset_tree->row_dim, d);

        for(int i = 0; i < hierarchy_n; i++){

          if(i == 0){
            matrix_copy(subset_tree->near_n_matrix_vec[0], subset_tree->hierarchy_matrix_vec[first_index + i]);
            continue;
          }
          else{
            mat* less_near_n = matrix_new(subset_tree->near_n_matrix_vec[0]->nrows, 
                          subset_tree->near_n_matrix_vec[0]->ncols + d);
            append_matrices_horizontally( subset_tree->near_n_matrix_vec[0], 
                        subset_tree->hierarchy_matrix_vec[first_index + i], less_near_n);
            matrix_delete(subset_tree->near_n_matrix_vec[0]);
            subset_tree->near_n_matrix_vec[0] = NULL;

            subset_tree->near_n_matrix_vec[0] = less_near_n;
          }

        }

        auto svd_start_time = chrono::system_clock::now();

        mat *U = matrix_new(subset_tree->near_n_matrix_vec[0]->nrows, d);

        mat *SS = matrix_new(d, d);

        mat *Vt = matrix_new(d, subset_tree->near_n_matrix_vec[0]->ncols);

        truncated_singular_value_decomposition(subset_tree->near_n_matrix_vec[0], U, SS, Vt, d);

        auto end_svd_time = chrono::system_clock::now();
        auto elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(end_svd_time - svd_start_time);

        for(int i = 0; i < d; i++){
          double current_value = matrix_get_element(SS, i, i);
          matrix_set_element(SS, i, i, sqrt(abs(current_value)) );
        }

        U_cur_iter = matrix_new(count_labeled_node, d);

        matrix_matrix_mult(U, SS, U_cur_iter);


        matrix_delete(U);

        matrix_delete(Vt);
        matrix_delete(subset_tree->near_n_matrix_vec[0]);



        U = NULL;
        Vt = NULL;
        subset_tree->near_n_matrix_vec[0] = NULL;

        break;
      }

      int unique_update_times = record_next_level.size();
      cout<<"unique_update_times = "<<unique_update_times<<endl;

      for(unordered_set<int>::iterator it = record_next_level.begin(); it != record_next_level.end(); it++){
        subset_tree->hierarchy_matrix_vec[*it] = matrix_new(subset_tree->row_dim, d);
      }



      for(unordered_set<int>::iterator it = record_next_level.begin(); it != record_next_level.end(); it++){

        int first_index = (*it) * hierarchy_n + 1;

        subset_tree->near_n_matrix_vec[*it] = matrix_new(subset_tree->row_dim, d);

        for(int i = 0; i < hierarchy_n; i++){

          if(i == 0){
            matrix_copy(subset_tree->near_n_matrix_vec[*it], subset_tree->hierarchy_matrix_vec[first_index + i]);
            continue;
          }
          else{
            mat* less_near_n = matrix_new(subset_tree->near_n_matrix_vec[*it]->nrows, 
                          subset_tree->near_n_matrix_vec[*it]->ncols + d);
            append_matrices_horizontally( subset_tree->near_n_matrix_vec[*it], 
                        subset_tree->hierarchy_matrix_vec[first_index + i], less_near_n);
            matrix_delete(subset_tree->near_n_matrix_vec[*it]);
            subset_tree->near_n_matrix_vec[*it] = NULL;

            subset_tree->near_n_matrix_vec[*it] = less_near_n;
          }

        }

        dense_sub_svd_function( d, pass, 
          subset_tree->near_n_matrix_vec[*it], 
          subset_tree->hierarchy_matrix_vec[*it] );

      }

      for(unordered_set<int>::iterator it = record_next_level.begin(); it != record_next_level.end(); it++){
        matrix_delete(subset_tree->near_n_matrix_vec[*it]);
        subset_tree->near_n_matrix_vec[*it] = NULL;
      }

      unordered_set<int> temp_record_next_level;
      for(unordered_set<int>::iterator it = record_next_level.begin(); it != record_next_level.end(); it++){
        int son_index;
        if( (*it) % hierarchy_n == 0){
          son_index = (*it) / hierarchy_n - 1;
          if(son_index == -1){
            end_while = true; 
          }
        }
        else{
          son_index = (*it) / hierarchy_n;
        }
        temp_record_next_level.insert(son_index);
      }
      record_next_level = temp_record_next_level;

    }
    
    cout<<"Writing back embedding for the "<<iter<<" round"<<endl;

    int iter_beginning = (iter-1) * count_labeled_node;


    get_Eigen_matrix_from_mkl(U_cur_iter, U_Eigen);      
    U_all_shots.block(iter_beginning, 0, count_labeled_node, d) = U_Eigen;


    auto iter_end_time = chrono::system_clock::now();
    auto elapsed_iter_time = chrono::duration_cast<std::chrono::seconds>(iter_end_time - iter_start_time);
    cout << "Iter = "<<iter<<", time = "<< elapsed_iter_time.count() << endl;  


  }








  auto end_eb_time = chrono::system_clock::now();

  int residue_order = 0;
  
  if(residuemax * 10000 == 1){
    residue_order = 5;
  }
  else if(residuemax * 100000 == 1){
    residue_order = 6;
  }
  else if(residuemax * 1000000 == 1){
    residue_order = 7;
  }
  else if(residuemax * 10000000 == 1){
    residue_order = 8;
  }
  else if(residuemax * 1000000000 == 1){
    residue_order = 10;
  }
  else{
    cout<<"Other!"<<endl;
  }

  string outUfile = EBpath + queryname + std::to_string(delta) + "_" + std::to_string(residue_order) + "_" + std::to_string(nParts) + "_" + std::to_string(hierarchy_n) + "_" + "De_svd_u_Dppr_parallel_bound_nodegree_U.csv";

  ofstream outU(outUfile.c_str());

  outU << U_all_shots.format(CSVFormat);

  auto end_time = chrono::system_clock::now();
  auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_time - end_eb_time);
  cout << "write out embedding time: "<< elapsed_write_time.count() << endl;
  auto elapsed_time = chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  cout << "total embedding time: "<< elapsed_time.count() << endl;
  outU.close();

}

