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


#include<queue>

#include "ppr_computation_store_dynamic.h"


#include <boost/functional/hash.hpp>


#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<atomic>


using namespace Eigen;

using namespace std;









bool maxNNZCmp(const pair<int, int>& a, const pair<int, int>& b){
    return a.second > b.second;
}




bool maxScoreCmp(const pair<double, pair<int, int>>& a, const pair<double, pair<int, int>>& b){
    return a.first > b.first;
}





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


    auto sub_svd_start_time = chrono::system_clock::now();

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

    auto sub_end_eb_time = chrono::system_clock::now();
    auto sub_elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(sub_end_eb_time - sub_svd_start_time);

}









void mkl_right_matrix_multiplication_without_norm(
d_row_tree_mkl* subset_tree,
mat* mkl_left_matrix, Eigen::MatrixXd &V_matrix,
int vertex_number, vector<int> & line_update_mat_tree_record, int current_out_iter,
vector<long long int>& record_submatrices_nnz
)
{

    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    double total_norm_time = 0;

    for(int iter = 0; iter < subset_tree->nParts; iter++){
      if(line_update_mat_tree_record[iter] != current_out_iter){
        continue;
      }


      unique_update_times++;

      int temp_matrix_rows = subset_tree->mat_mapping[iter].cols();
      int temp_matrix_cols = subset_tree->mat_mapping[iter].rows();


      SparseMatrix<double, RowMajor, int> ppr_matrix_temp(temp_matrix_cols, temp_matrix_rows);


      ppr_matrix_temp = subset_tree->mat_mapping[iter];


      long long int nnz = record_submatrices_nnz[iter];

      mat_coo *ppr_matrix_coo = coo_matrix_new(temp_matrix_cols, temp_matrix_rows, nnz);


      ppr_matrix_coo->nnz = nnz;

      long nnz_iter=0;
      double ppr_norm =0;


      for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
          for (SparseMatrix<double, RowMajor, int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){

              double value1 = it.value();
              if(value1 == 0){
                // continue;
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


      mat_csr* ppr_matrix = csr_matrix_new();


      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


      coo_matrix_delete(ppr_matrix_coo);


      ppr_matrix_coo = NULL;

      mat *mkl_result_mat = matrix_new(temp_matrix_rows, mkl_left_matrix->ncols);


      auto right_matrix_start_time = chrono::system_clock::now();

      csr_matrix_transpose_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);


      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);

      if(iter != subset_tree->nParts-1){
        for(int i = 0; i < temp_matrix_rows; i++){
            for(int j = 0; j < mkl_left_matrix->ncols; j++){
                V_matrix(iter * temp_matrix_rows + i, j) = matrix_get_element(mkl_result_mat, i, j);
                if(isnan(V_matrix(iter * temp_matrix_rows + i, j)) || isinf(V_matrix(iter * temp_matrix_rows + i, j))){
                    cout<<"V_matrix("<<i<<", "<<j<<") = "<<V_matrix(i, j)<<endl;
                }
            }
        }
      }
      else{
        for(int i = 0; i < temp_matrix_rows; i++){
            for(int j = 0; j < mkl_left_matrix->ncols; j++){
                V_matrix(vertex_number - temp_matrix_rows + i, j) = matrix_get_element(mkl_result_mat, i, j);
                if(isnan(V_matrix(vertex_number - temp_matrix_rows + i, j)) || isinf(V_matrix(vertex_number - temp_matrix_rows + i, j))){
                    cout<<"V_matrix("<<i<<", "<<j<<") = "<<V_matrix(i, j)<<endl;
                }
            }
        }
      }
  

      ppr_matrix_temp.resize(0, 0);


      ppr_matrix_temp.data().squeeze();


      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;


      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


    }
    cout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    cout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    cout << "Total norm cost time = "<< total_norm_time << endl;

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


  double residuemax = backward_theta; // PPR error up to residuemax

  double reservemin = backward_theta; // Retain approximate PPR larger than reservemin
  cout << "alpha: " << alpha << ", residuemax: " << residuemax << ", reservemin: " << reservemin <<endl;
  cout << "nParts: "<< nParts << ", hierarchy_n: "<< hierarchy_n << ", vertex_number: "<<vertex_number<<endl; 
  cout << "dynamic_ppr_start_iter: "<<dynamic_ppr_start_iter<<", lazy_update_start_iter: "<<lazy_update_start_iter<<endl;
  cout << "delta: "<<delta<<endl;


  if(vertex_number < 1e7){
    cout<<"less than 1e7 vertex number!"<<endl;

    vector<int> labeled_node_vec;


    string config_path =  "DY_LP_Dataset/" + queryname + "/config.txt";
    ifstream infile3( config_path.c_str() );

    unordered_map<int, int> row_index_mapping;
    
    int d = 128;
    
    int snapshots_number = 0;

    vector<string> shots_address_vec;

    string s2;
    while(getline(infile3, s2)) 
    { 
      shots_address_vec.push_back(s2);
      snapshots_number++;
    }
    
    cout<<"snapshots_number = "<<snapshots_number<<endl;

    int count_labeled_node = 0;
    ifstream infile2(shots_address_vec[0].c_str());
    int node_number;




    while (infile2>>node_number){ 
      labeled_node_vec.push_back(node_number);
      row_index_mapping[node_number] = count_labeled_node;
      count_labeled_node++;
    }

    cout<<"count_labeled_node = "<<count_labeled_node<<endl;

    
    Graph* g = new Graph();

    g->initializeDirectedDynamicGraph(vertex_number);

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


    vector<int> update_mat_tree_record;


    update_mat_tree_record.resize(nParts);

    for(int i = 0; i < update_mat_tree_record.size(); i++){
        update_mat_tree_record[i] = -1;
    }







    int upper_nnz = ceil(1 / residuemax / alpha);

    Queue * queue_list = new Queue[count_labeled_node];

    float **residue = new float* [count_labeled_node];
    float **pi = new float* [count_labeled_node];


    bool **flags = new bool* [count_labeled_node];

    for(int i = 0; i < count_labeled_node; i++){
      residue[i] = new float[vertex_number];

      pi[i] = new float[vertex_number];
  
  
      flags[i] = new bool[vertex_number];
    }
    

    for(int i = 0; i < count_labeled_node; i++){

      queue_list[i] = Queue
      {
        (int*)malloc( sizeof(int) * (upper_nnz + 2) * 2 ),
        (upper_nnz + 2) * 2,
        0,
        0
      };
    }


    Queue * queue_list_transpose = new Queue[count_labeled_node];

    float **residue_transpose = new float* [count_labeled_node];
    float **pi_transpose = new float* [count_labeled_node];


    bool **flags_transpose = new bool* [count_labeled_node];

    for(int i = 0; i < count_labeled_node; i++){
      residue_transpose[i] = new float[vertex_number];

      pi_transpose[i] = new float[vertex_number];


      flags_transpose[i] = new bool[vertex_number];
    }
    



    for(int i = 0; i < count_labeled_node; i++){
      queue_list_transpose[i] = Queue
      {
        (int*)malloc( sizeof(int) * (upper_nnz + 2) * 2 ),
        (upper_nnz + 2) * 2,
        0,
        0
      };
    }



    //Test Part
    string ptestdataset =  "DY_LP_Dataset/" + queryname + "/" + queryname + "-Pos_LP_Test.txt";
    string ntestdataset =  "DY_LP_Dataset/" + queryname + "/" + queryname + "-Neg_LP_Test.txt";

    ifstream ptest(ptestdataset.c_str());
    ifstream ntest(ntestdataset.c_str());
    unordered_set<pair<int, int>, boost::hash< pair<int, int>>> pedge_set;
    unordered_set<pair<int, int>, boost::hash< pair<int, int>>> nedge_set;

    int sample_m = 0;
    while(ptest.good()){
      int from;
      int to;
      ptest >> from >> to;
      pedge_set.insert(make_pair(from, to));
      sample_m++;
    }
    while(ntest.good()){
      int from;
      int to;
      ntest >> from >> to;
      nedge_set.insert(make_pair(from, to));
    }



    vector<MatrixXd> left_matrix_U_cache(snapshots_number - 1);
    for(int i = 0; i < snapshots_number - 1; i++){
      left_matrix_U_cache[i].resize(0, 0);
    }

    vector<int> left_matrix_index_mapping(nParts, -1);

    vector<int> left_matrix_pointer_number(snapshots_number - 1, 0);


    MatrixXd V;
    V.resize(vertex_number, d);

    mat* U_cur_iter;

    U_cur_iter = matrix_new(count_labeled_node, d);


    vector<long long int> record_submatrices_nnz(nParts, 0);



    for(int iter = 1; iter < snapshots_number; iter++){
      auto iter_start_time = chrono::system_clock::now();    
      cout<<"Current Snapshot Number = "<<iter<<endl;

      vector<pair<int, int>> edge_vec;
      cout<<"shots_address_vec[iter] = "<<shots_address_vec[iter]<<endl;
      g->inputDirectedDynamicGraph(shots_address_vec[iter].c_str(), edge_vec);

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


        refresh_threads.push_back(thread(nodegree_DenseDirected_Refresh_PPR_initialization, start, end, g, residuemax, reservemin, 
          alpha, 
          std::ref(labeled_node_vec), 
          residue, 
          pi, 
          flags, 
          queue_list, 
          common_group_size,
          nParts,
          std::ref(inner_group_mapping),
          iter,
          vertex_number,
          subset_tree,
          dynamic_ppr_start_iter )
        );
      }

      for (int t = 0; t < NUMTHREAD ; t++){
        refresh_threads[t].join();
      }
      vector<thread>().swap(refresh_threads);




      auto finish_ppr_refresh_time = chrono::system_clock::now();
      auto elapsed_ppr_refresh_time = chrono::duration_cast<std::chrono::seconds>(finish_ppr_refresh_time - ppr_refresh_time);
      cout << "Iter = "<<iter <<", refresh ppr time: "<< elapsed_ppr_refresh_time.count() << endl;








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



        threads.push_back(thread(nodegree_DenseDirectedDynamicForwardPush, start, end, g, residuemax, reservemin, 
        alpha, std::ref(labeled_node_vec), 
        residue, pi, 
        flags, queue_list, 
        common_group_size,
        nParts,
        std::ref(inner_group_mapping),
        subset_tree
        ));



      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);



      auto start_ppr_matrix_time = chrono::system_clock::now();
      auto elapsed_ppr_time = chrono::duration_cast<std::chrono::seconds>(start_ppr_matrix_time - ppr_start_time);
      cout<< "Iter = "<<iter << ", computing ppr time: "<< elapsed_ppr_time.count() << endl;

























      auto ppr_refresh_transpose_time = std::chrono::system_clock::now();



      for (int t = 1; t <= NUMTHREAD; t++){

        int start = (t-1)*(labeled_node_vec.size()/NUMTHREAD);
        int end = 0;
        if (t == NUMTHREAD){
          end = labeled_node_vec.size();
        } else{
          end = t*(labeled_node_vec.size()/NUMTHREAD);
        }


        refresh_threads.push_back(thread(nodegree_DenseDirected_Refresh_PPR_initialization_Transpose, 
        start, end, g, residuemax, reservemin, 
          alpha, 
          std::ref(labeled_node_vec), 
          residue_transpose, 
          pi_transpose,
          flags_transpose, 
          queue_list_transpose,
          common_group_size,
          nParts,
          std::ref(inner_group_mapping),
          iter,
          vertex_number,
          subset_tree,
          dynamic_ppr_start_iter )
        );

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        refresh_threads[t].join();
      }
      vector<thread>().swap(refresh_threads);



      auto finish_ppr_refresh_transpose_time = chrono::system_clock::now();
      auto elapsed_ppr_refresh_transpose_time = 
      chrono::duration_cast<std::chrono::seconds>(finish_ppr_refresh_transpose_time - ppr_refresh_transpose_time);
      cout << "Iter = "<<iter <<", refresh transpose ppr time: "<< elapsed_ppr_refresh_transpose_time.count() << endl;










      cout << "transpose ppr computation " << endl;
      auto ppr_transpose_start_time = std::chrono::system_clock::now();


      for (int t = 1; t <= NUMTHREAD; t++){

        int start = (t-1)*(labeled_node_vec.size()/NUMTHREAD);
        int end = 0;
        if (t == NUMTHREAD){
          end = labeled_node_vec.size();
        } else{
          end = t*(labeled_node_vec.size()/NUMTHREAD);
        }




        threads.push_back(thread(nodegree_DenseDirectedDynamicForwardPushTranspose, start, end, g, residuemax, reservemin, 
        alpha, std::ref(labeled_node_vec), 
        residue_transpose, pi_transpose, 
        flags_transpose, queue_list_transpose, 
        common_group_size, 
        nParts,
        std::ref(inner_group_mapping),
        subset_tree
        ));

      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);


      auto finish_ppr_transpose_matrix_time = chrono::system_clock::now();
      auto elapsed_ppr_transpose_time = 
      chrono::duration_cast<std::chrono::seconds>(finish_ppr_transpose_matrix_time - ppr_transpose_start_time);
      cout<< "Iter = "<<iter << ", computing transpose ppr time: "<< elapsed_ppr_transpose_time.count() << endl;



      if(iter >= lazy_update_start_iter){

          vector<thread> threads_top_list;

          for(int i = 0; i < vec_mapping.size(); i++){

              threads_top_list.push_back(thread(Log_sparse_matrix_entries_with_norm_computation, 
              i,    
              reservemin, 
              subset_tree,
              std::ref(vec_mapping),
              std::ref(update_mat_tree_record),
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




      }
      else{


          vector<thread> threads_top_list;

          for(int i = 0; i < vec_mapping.size(); i++){    
      
              update_mat_tree_record[i] = iter;

              threads_top_list.push_back(thread(Log_sparse_matrix_entries_LP, 
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
      }
      

      
        int largest_level_start_index = subset_tree->largest_level_start_index;

        int count_current_threads_number = 0;

        unordered_set<int> record_next_level;

        int sparse_unique_update_times = 0;
        
        for(int update_j = 0; update_j < nParts; update_j++){
          if(update_mat_tree_record[update_j] == iter){

            if(subset_tree->mat_mapping[update_j].nonZeros() == 0){
              update_mat_tree_record[update_j] = -1;
              continue;
            }
            sparse_unique_update_times++;

            left_matrix_pointer_number[iter - 1]++;

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


        if(sparse_unique_update_times == 0){
          cout<<"Error bounded! No SVD update for iter = "<<iter<<"!"<<endl;
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

            matrix_copy(U_cur_iter, U);


            matrix_delete(U);

            matrix_delete(SS);
            matrix_delete(Vt);
            matrix_delete(subset_tree->near_n_matrix_vec[0]);


            U = NULL;
            SS = NULL;
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
              // update_i, 
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
        

      auto iter_end_time = chrono::system_clock::now();
      auto elapsed_iter_time = chrono::duration_cast<std::chrono::seconds>(iter_end_time - iter_start_time);
      cout << "Iter = "<<iter<<", time = "<< elapsed_iter_time.count() << endl;  



      MatrixXd Eigen_U;

      get_Eigen_matrix_from_mkl(U_cur_iter, Eigen_U);      

      for(int j_up = 0; j_up < nParts; j_up++){
        if(update_mat_tree_record[j_up] == iter){

          int index = left_matrix_index_mapping[j_up];

          if(index != -1){

            if(left_matrix_pointer_number[index] != 0){

              left_matrix_pointer_number[index]--;

              if(left_matrix_pointer_number[index] == 0){

                if( !(left_matrix_U_cache[index].rows() == 0 &&  left_matrix_U_cache[index].cols() == 0) ){

                  left_matrix_U_cache[index].resize(0, 0);

                  cout<<"matrix_delete(U_cur_iter) for snapshot = "<<index<<endl;
                }
              }
              else{
                //pass
              }
            }
            else{
              cout<<"Error!!!!!!!!!!!!"<<endl;
            }
          }
          else{
            //pass
          }

          left_matrix_index_mapping[j_up] = iter - 1;

        }
      }

      left_matrix_U_cache[iter - 1] = Eigen_U;


      mkl_right_matrix_multiplication_without_norm(subset_tree, U_cur_iter,
      V, vertex_number, update_mat_tree_record, iter, 
      std::ref(record_submatrices_nnz));

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - iter_end_time);
      cout << "Right matrix at iter "<<iter<<", time = "<< elapsed_right_matrix_time.count() << endl;


      int start = subset_tree->start_row;
      int row_number = subset_tree->row_dim;

      vector<pair<double, pair<int, int>>> embedding_score;


      for (auto it = pedge_set.begin(); it != pedge_set.end(); ++it) {
        int i = it->first;
        int row_i = row_index_mapping[i];

        int j = it->second;

        int group_number = vertex_mapping[j];
        int index = left_matrix_index_mapping[group_number];
        double score;
        if(index != -1){
          MatrixXd &U_temp = left_matrix_U_cache[index];
          score = U_temp.row(row_i).dot(V.row(j));
        }
        else{
          MatrixXd &U_temp = Eigen_U;
          score = U_temp.row(row_i).dot(V.row(j));
        }


        embedding_score.push_back(make_pair(score, make_pair(i,j)));
      }
      for (auto it = nedge_set.begin(); it != nedge_set.end(); ++it) {
        int i = it->first;
        int row_i = row_index_mapping[i];

        int j = it->second;

        int group_number = vertex_mapping[j];
        int index = left_matrix_index_mapping[group_number];



        double score;
        if(index != -1){
          MatrixXd &U_temp = left_matrix_U_cache[index];
          score = U_temp.row(row_i).dot(V.row(j));
        }
        else{
          MatrixXd &U_temp = Eigen_U;
          score = U_temp.row(row_i).dot(V.row(j));
        }
        embedding_score.push_back(make_pair(score, make_pair(i,j)));
      }


      // Top sample_m predicted edges is considered
      nth_element(embedding_score.begin(), embedding_score.begin()+sample_m-1, embedding_score.end(), maxScoreCmp);
      sort(embedding_score.begin(), embedding_score.begin()+sample_m-1, maxScoreCmp);
      int predict_positive_number = 0;
      for (auto it = embedding_score.begin(); it != embedding_score.begin()+sample_m; ++it) {
        int i = it->second.first;
        int j = it->second.second;
        if(pedge_set.find(make_pair(i,j)) != pedge_set.end()){
          predict_positive_number ++;
        }
      }


      auto LP_end_time = chrono::system_clock::now();
      auto elapsed_LP_time = chrono::duration_cast<std::chrono::seconds>(LP_end_time - right_matrix_end_time);
      cout << "LP at iter "<<iter<<", time = "<< elapsed_LP_time.count() << endl;
      cout << "link prediction precision: " << predict_positive_number/ (double) (sample_m) << endl;


    }


    auto end_eb_time = chrono::system_clock::now();
    

    auto end_time = chrono::system_clock::now();
    auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_time - end_eb_time);
    cout << "write out embedding time: "<< elapsed_write_time.count() << endl;
    auto elapsed_time = chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    cout << "total embedding time: "<< elapsed_time.count() << endl;
      
  }
  else{
    cout<<"over 1e7 vertex number!"<<endl;

    vector<int> labeled_node_vec;

    string config_path =  "DY_LP_Dataset/" + queryname + "/config.txt";
    ifstream infile3( config_path.c_str() );

    unordered_map<int, int> row_index_mapping;
    
    int d = 128;
    
    int snapshots_number = 0;

    vector<string> shots_address_vec;

    string s2;
    while(getline(infile3, s2)) 
    { 
      shots_address_vec.push_back(s2);
      snapshots_number++;
    }
    
    tcout<<"snapshots_number = "<<snapshots_number<<endl;

    int count_labeled_node = 0;
    ifstream infile2(shots_address_vec[0].c_str());
    int node_number;




    while (infile2>>node_number){ 
      labeled_node_vec.push_back(node_number);
      row_index_mapping[node_number] = count_labeled_node;
      count_labeled_node++;
    }

    tcout<<"count_labeled_node = "<<count_labeled_node<<endl;

    
    Graph* g = new Graph();

    g->initializeDirectedDynamicGraph(vertex_number);


    vector<int> vertex_mapping;

    vector<int> inner_group_mapping;


    vertex_mapping.resize(vertex_number);
    inner_group_mapping.resize(vertex_number);



    int common_group_size;

    common_group_size = vertex_number / nParts;

    int final_group_size = vertex_number - (nParts - 1) * common_group_size;

    tcout<<"common_group_size = "<<common_group_size<<endl;

    for(int t = 0; t < vertex_number; t++){
      int index = t / common_group_size;
      if(index != nParts){
        vertex_mapping[t] = index;
      }
      else{
        vertex_mapping[t] = index - 1;
      }
    }
    

    for(int i = 0; i < vertex_number; i++){
        int group = vertex_mapping[i];
        inner_group_mapping[i] = i - common_group_size * group;
    }



    d_row_tree_mkl_sparse_dynamic* subset_tree;

    subset_tree = new d_row_tree_mkl_sparse_dynamic(count_labeled_node, d, 
    nParts, hierarchy_n,
    //  vec_mapping,
    0, count_labeled_node,
    common_group_size,
    final_group_size);      


    vector<int> update_mat_tree_record;


    update_mat_tree_record.resize(nParts);

    for(int i = 0; i < update_mat_tree_record.size(); i++){
        update_mat_tree_record[i] = -1;
    }



    int upper_nnz = ceil(1 / residuemax / alpha);

    Queue * queue_list = new Queue[count_labeled_node];



    vector<vector<std::map<int, float>>> pi_map(nParts, vector<std::map<int, float>>(count_labeled_node));
    vector<vector<std::map<int, float>>> pi_transpose_map(nParts, vector<std::map<int, float>>(count_labeled_node));
    vector<vector<std::map<int, float>>> residue_map(nParts, vector<std::map<int, float>>(count_labeled_node));
    vector<vector<std::map<int, float>>> residue_transpose_map(nParts, vector<std::map<int, float>>(count_labeled_node));





    //Test Part
    string ptestdataset =  "DY_LP_Dataset/" + queryname + "/" + queryname + "-Pos_LP_Test.txt";
    string ntestdataset =  "DY_LP_Dataset/" + queryname + "/" + queryname + "-Neg_LP_Test.txt";

    ifstream ptest(ptestdataset.c_str());
    ifstream ntest(ntestdataset.c_str());
    unordered_set<pair<int, int>, boost::hash< pair<int, int>>> pedge_set;
    unordered_set<pair<int, int>, boost::hash< pair<int, int>>> nedge_set;

    int sample_m = 0;
    while(ptest.good()){
      int from;
      int to;
      ptest >> from >> to;
      pedge_set.insert(make_pair(from, to));
      sample_m++;
    }
    while(ntest.good()){
      int from;
      int to;
      ntest >> from >> to;
      nedge_set.insert(make_pair(from, to));
    }


    vector<unordered_set<pair<int, int>, boost::hash< pair<int, int>>>> pedge_set_nparts(nParts);
    vector<unordered_set<pair<int, int>, boost::hash< pair<int, int>>>> nedge_set_nparts(nParts);

    vector<vector<pair<double, pair<int, int>>>> nparts_embedding_score(nParts);



    for (auto it = pedge_set.begin(); it != pedge_set.end(); ++it) {
      int i = it->first;
      int j = it->second;

      int group_number = vertex_mapping[j];

      pedge_set_nparts[group_number].insert(make_pair(i, j));

      (nparts_embedding_score[group_number]).push_back(make_pair(0.0, make_pair(i,j)));
    }
    for (auto it = nedge_set.begin(); it != nedge_set.end(); ++it) {
      int i = it->first;
      int j = it->second;

      int group_number = vertex_mapping[j];

      nedge_set_nparts[group_number].insert(make_pair(i, j));

      (nparts_embedding_score[group_number]).push_back(make_pair(0.0, make_pair(i,j)));
    }

    nedge_set.clear();









    vector<vector<vector<pair<int, float>>>> tripletList(nParts, vector<vector<pair<int, float>>>(count_labeled_node));
    vector<vector<vector<pair<int, float>>>> tripletList_transpose(nParts, vector<vector<pair<int, float>>>(count_labeled_node));

    vector<vector<vector<pair<int, float>>>> residue_pair_list(nParts, vector<vector<pair<int, float>>>(count_labeled_node));
    vector<vector<vector<pair<int, float>>>> residue_transpose_pair_list(nParts, vector<vector<pair<int, float>>>(count_labeled_node));


    for(int iter = 1; iter < snapshots_number; iter++){
      auto iter_start_time = chrono::system_clock::now();    
      tcout<<"Current Snapshot Number = "<<iter<<endl;

      vector<pair<int, int>> edge_vec;
      tcout<<"shots_address_vec[iter] = "<<shots_address_vec[iter]<<endl;
      g->inputDirectedDynamicGraph(shots_address_vec[iter].c_str(), edge_vec);


      if(lazy_update_start_iter != 200 && iter < lazy_update_start_iter - 1){
        continue;
      }
      
      // if(lazy_update_start_iter == 200 && iter != snapshots_number - 100 - 1 && iter < snapshots_number - 1){
      //   continue;
      // }
      // if(lazy_update_start_iter == 200 && iter < snapshots_number - 5){
      //   continue;
      // }

      mat* U_cur_iter;

      U_cur_iter = matrix_new(count_labeled_node, d);

      // if(iter < snapshots_number - 1){
      //   continue;
      // }
  
      // if(iter != 5 && iter != 6){
      //   continue;
      // }


      auto all_ppr_start_time = std::chrono::system_clock::now();


      tcout << "ppr computation " << endl;
      
      
      vector<thread> threads;

      for (int t = 1; t <= NUMTHREAD; t++){

        int start = (t-1)*(labeled_node_vec.size()/NUMTHREAD);
        int end = 0;
        if (t == NUMTHREAD){
          end = labeled_node_vec.size();
        } else{
          end = t*(labeled_node_vec.size()/NUMTHREAD);
        }


    
        threads.push_back(thread(All_initialization_and_push_final, 
        start, end, g, residuemax, reservemin, 
        alpha,
        common_group_size,
        nParts,
        iter,
        vertex_number, 
        dynamic_ppr_start_iter,
        t - 1,
        std::ref(labeled_node_vec), 
        std::ref(tripletList),
        std::ref(tripletList_transpose),
        std::ref(residue_pair_list),
        std::ref(residue_transpose_pair_list),
        std::ref(inner_group_mapping),
        std::ref(vertex_mapping),
        std::ref(edge_vec)
        ));


      }

      for (int t = 0; t < NUMTHREAD ; t++){
        threads[t].join();
      }
      vector<thread>().swap(threads);


      auto all_ppr_end_time = chrono::system_clock::now();
      auto elapsed_all_ppr_time = 
      chrono::duration_cast<std::chrono::seconds>(all_ppr_end_time - all_ppr_start_time);
      tcout<< "Iter = "<<iter << ", all ppr time: "<< elapsed_all_ppr_time.count() << endl;





      auto log_start_time = std::chrono::system_clock::now();


      if(iter < dynamic_ppr_start_iter){
        vector<thread> threads_top_list;
        cout<<"iter < dynamic_ppr_start_iter"<<endl;

        for(int i = 0; i < nParts; i++){    
    
            update_mat_tree_record[i] = iter;


            threads_top_list.push_back(thread(
            Log_sparse_matrix_entries_LP_Robinhood_two_with_transpose_nparts_triplet, 
            i,    
            reservemin, 
            subset_tree,
            common_group_size,
            final_group_size,
            std::ref(tripletList),
            std::ref(tripletList_transpose),
            nParts,
            residuemax,
            alpha,
            iter,
            dynamic_ppr_start_iter
            ));
            

        }


        int all_blocks_number = nParts;
        for (int t = 0; t < all_blocks_number; t++){
          threads_top_list[t].join();
        }


        vector<thread>().swap(threads_top_list);
      }
      else{

        cout<<"iter >= dynamic_ppr_start_iter"<<endl;


        for (int submatrix_index = 0; submatrix_index < nParts; submatrix_index++){


          threads.push_back(thread(
          Log_sparse_matrix_entries_LP_Robinhood_two_with_transpose_nparts_triplet_norm_computation, 
          submatrix_index,
          reservemin, 
          subset_tree,
          common_group_size,
          final_group_size,
          std::ref(tripletList),
          std::ref(tripletList_transpose),
          std::ref(update_mat_tree_record),
          nParts,
          residuemax,
          alpha,
          delta,
          iter,
          dynamic_ppr_start_iter
          ));
          
        }

        for (int thread_number = 0; thread_number < NUMTHREAD ; thread_number++){
          threads[thread_number].join();
        }
        vector<thread>().swap(threads);



        int update_times = 0;



        for(int update_j = 0; update_j < nParts; update_j++){
          if(update_mat_tree_record[update_j] == -1){
            continue;
          }
    
          update_times++;

        }

        cout<<"update_times = "<<update_times<<endl;


      }

















      auto log_end_time = chrono::system_clock::now();
      auto elapsed_log_time = chrono::duration_cast<std::chrono::seconds>(log_end_time - log_start_time);
      tcout<< "Iter = "<<iter << ", log time: "<< elapsed_log_time.count() << endl;

      cout<<"finish log operation!!!"<<endl;
      

      int largest_level_start_index = subset_tree->largest_level_start_index;

      int count_current_threads_number = 0;

      unordered_set<int> record_next_level;

      int sparse_unique_update_times = 0;
      
      for(int update_j = 0; update_j < nParts; update_j++){
        if(update_mat_tree_record[update_j] == iter){

          if(subset_tree->svd_mat_mapping[update_j].nonZeros() == 0){
            update_mat_tree_record[update_j] = -1;
            continue;
          }
          sparse_unique_update_times++;

          // left_matrix_pointer_number[iter - 1]++;

          sparse_sub_svd_function_with_norm_computation_Robinhood(d, pass, 
          update_j, 
          subset_tree,
          largest_level_start_index,
          iter,
          lazy_update_start_iter,
          subset_tree->svd_mat_mapping[update_j],
          reservemin);
          



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


      if(sparse_unique_update_times == 0){
        tcout<<"Error bounded! No SVD update for iter = "<<iter<<"!"<<endl;
        auto iter_end_time = chrono::system_clock::now();
        auto all_elapsed_iter_time = chrono::duration_cast<std::chrono::seconds>(iter_end_time - iter_start_time);
        cout << "All time in iter "<<iter<<", time = "<< all_elapsed_iter_time.count() << endl;
        continue;
      }
      else{
        tcout<<"sparse_unique_update_times = "<<sparse_unique_update_times<<endl;
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

          matrix_copy(U_cur_iter, U);


          matrix_delete(U);

          matrix_delete(SS);
          matrix_delete(Vt);
          matrix_delete(subset_tree->near_n_matrix_vec[0]);


          U = NULL;
          SS = NULL;
          Vt = NULL;
          subset_tree->near_n_matrix_vec[0] = NULL;

          break;
        }

        int unique_update_times = record_next_level.size();
        tcout<<"unique_update_times = "<<unique_update_times<<endl;
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

          dense_sub_svd_function_Robinhood( d, pass, 
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


      auto iter_end_time = chrono::system_clock::now();
      auto elapsed_left_SVD_time = chrono::duration_cast<std::chrono::seconds>(iter_end_time - log_end_time);
      tcout << "Iter = "<<iter<<", elapsed left SVD time = "<< elapsed_left_SVD_time.count() << endl;  
      auto elapsed_iter_time = chrono::duration_cast<std::chrono::seconds>(iter_end_time - iter_start_time);
      tcout << "Iter = "<<iter<<", time = "<< elapsed_iter_time.count() << endl;  


      
      mkl_right_matrix_multiplication_without_norm_Robinhood_with_link_prediction(
      subset_tree, U_cur_iter,
      vertex_number, update_mat_tree_record, iter, 
      count_labeled_node, reservemin,
      common_group_size,
      final_group_size,
      dynamic_ppr_start_iter,
      pedge_set_nparts,
      nedge_set_nparts,
      nparts_embedding_score,
      vertex_mapping,
      g,
      inner_group_mapping,
      row_index_mapping
      );

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - iter_end_time);
      tcout << "Right matrix at iter "<<iter<<", time = "<< elapsed_right_matrix_time.count() << endl;


      vector<pair<double, pair<int, int>>> embedding_score;

      for(int part = 0; part < nParts; part++){
        embedding_score.insert(embedding_score.end(), (nparts_embedding_score[part]).begin(), (nparts_embedding_score[part]).end());
      }


      // Top sample_m predicted edges is considered
      nth_element(embedding_score.begin(), embedding_score.begin()+sample_m, embedding_score.end(), maxScoreCmp);
      sort(embedding_score.begin(), embedding_score.begin()+sample_m, maxScoreCmp);
      int predict_positive_number = 0;
      for (auto it = embedding_score.begin(); it != embedding_score.begin()+sample_m; ++it) {
        int i = it->second.first;
        int j = it->second.second;
        if(pedge_set.find(make_pair(i,j)) != pedge_set.end()){
          predict_positive_number ++;
        }
      }

      embedding_score.clear();

      auto LP_end_time = chrono::system_clock::now();
      auto elapsed_LP_time = chrono::duration_cast<std::chrono::seconds>(LP_end_time - right_matrix_end_time);
      tcout << "LP at iter "<<iter<<", time = "<< elapsed_LP_time.count() << endl;

      auto all_elapsed_iter_time = chrono::duration_cast<std::chrono::seconds>(LP_end_time - iter_start_time);
      cout << "All time in iter "<<iter<<", time = "<< all_elapsed_iter_time.count() << endl;



      
      tcout << "link prediction precision: " << predict_positive_number/ (double) (sample_m) << endl;


    }


    auto end_eb_time = chrono::system_clock::now();

    auto end_time = chrono::system_clock::now();
    auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_time - end_eb_time);
    tcout << "write out embedding time: "<< elapsed_write_time.count() << endl;
    auto elapsed_time = chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    tcout << "total embedding time: "<< elapsed_time.count() << endl;
  }

}

