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
#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>

#include<queue>
#include "my_queue.h"

#include<assert.h>
#include <unordered_map>
#include<cmath>

#include<list>

#include<memory.h>

#include <atomic>


#include "../emhash/hash_table7.hpp"
#include <boost/functional/hash.hpp>
using third_float_map = emhash7::HashMap<int, float>;
using third_bool_map = emhash7::HashMap<int, bool>;
using third_int_map = emhash7::HashMap<int, int>;




#include"Graph_dynamic.h"



// #define __DEBUG
#ifdef __DEBUG
#define tcout cout
#else
#define tcout 0 && cout
#endif






using namespace Eigen;

using namespace std;


class column_tuple{
  public:
    int row;
    float pi;
    column_tuple(int row, float pi){
      this->row = row;
      this->pi = pi;
    }
};




class sparse_d_row_tree_mkl{
  public:
  int nParts;

  int level_p;
  int total_nodes;
  vector<mat*> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<mat*> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<mat*> near_n_matrix_vec;
  vector<mat*> less_near_n;

  unordered_map<int, SparseMatrix<double, 0, int64_t>> mat_mapping;

  // unordered_map<int, MatrixXd> dense_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  vector<double> norm_B_Bid_difference_vec;


  sparse_d_row_tree_mkl(int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row){

    this->nParts = nParts;

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;


    norm_B_Bid_difference_vec.resize(vec_mapping.size());

    for(int i = 0; i < vec_mapping.size(); i++){
      vector<int> &v = vec_mapping[i];
      int current_group_size = vec_mapping[i].size();
      mat_mapping[i].resize(row_dim, current_group_size);

      norm_B_Bid_difference_vec[i] = 0;
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    largest_level_start_index = total_nodes - nParts;

    largest_level_end_index = total_nodes - 1;

    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i] = matrix_new(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};













class d_row_tree_mkl{
  public:
  int nParts;

  int level_p;
  int total_nodes;
  vector<mat*> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<mat*> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<mat*> near_n_matrix_vec;
  vector<mat*> less_near_n;

  unordered_map<int, SparseMatrix<double, 0, int>> mat_mapping;

  unordered_map<int, MatrixXd> dense_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  vector<double> norm_B_Bid_difference_vec;


  d_row_tree_mkl(int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row){

    this->nParts = nParts;

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;


    norm_B_Bid_difference_vec.resize(vec_mapping.size());

    for(int i = 0; i < vec_mapping.size(); i++){
      vector<int> &v = vec_mapping[i];
      int current_group_size = vec_mapping[i].size();
      mat_mapping[i].resize(row_dim, current_group_size);
      dense_mat_mapping[i].resize(row_dim, current_group_size);
      norm_B_Bid_difference_vec[i] = 0;
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    cout<<"total_nodes = "<<total_nodes<<endl;

    largest_level_start_index = total_nodes - nParts;

    cout<<"largest_level_start_index = "<<largest_level_start_index<<endl;
    
    largest_level_end_index = total_nodes - 1;
    
    cout<<"largest_level_end_index = "<<largest_level_end_index<<endl;
    
    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i] = matrix_new(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};



class d_row_tree{
  public:

  int level_p;
  int total_nodes;
  vector<MatrixXd> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<MatrixXd> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<Eigen::MatrixXd> near_n_matrix_vec;
  vector<MatrixXd> less_near_n;

  unordered_map<int, SparseMatrix<double, 0, int>> mat_mapping;
  unordered_map<int, MatrixXd> dense_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  d_row_tree(int row_dim, int d, int nParts, int hierarchy_n, unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row){

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;
    
    for(int i = 0; i < vec_mapping.size(); i++){
      vector<int> &v = vec_mapping[i];
      int current_group_size = vec_mapping[i].size();
      mat_mapping[i].resize(row_dim, current_group_size);
      dense_mat_mapping[i].resize(row_dim, current_group_size);
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    largest_level_start_index = total_nodes - nParts;

    largest_level_end_index = total_nodes - 1;

    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i].resize(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};



void DynamicForwardPush(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->degree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;
        continue;
      }

      if(residue[it][v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];

          if(g->degree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->degree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];

        double changed_value = alpha * residue[it][v];
        
        answer->push_back(Triplet<double>(v, src, changed_value));
        answer->push_back(Triplet<double>(src, v, changed_value));


        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        if(update_mat_record[row_v][col_v] != 0){
          update_mat_record[row_v][col_v] = 0;
        }

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }
  return;

}



















































void ForwardPushSymmetric(int start, int end, UGraph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec){

  cout<<"Start ForwardPush!"<<endl;
  cout<<"start = "<<start<<endl;
  cout<<"end = "<<end<<endl;

  int vertices_number = g->n;

  double *residue = new double[vertices_number];

  double *pi = new double[vertices_number];

  bool *flags = new bool[vertices_number];

  int upper_nnz = ceil(1 / residuemax / alpha);



  //Create Queue
	Queue Q =
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

  //The queue to record used elements in this node
	Queue record_Q =
	{
		//.arr = 
		(int*)malloc(sizeof(int) * (upper_nnz + 2) * 2 ),
		//.capacity = 
		(upper_nnz + 2) * 2,
		//.front =
		0,
		//.rear = 
		0
	};




  memset(residue, 0, sizeof(double) * vertices_number);

  memset(pi, 0, sizeof(double) * vertices_number);

  memset(flags, false, sizeof(bool) * vertices_number);

  int record_max;
  
  for(int it = start; it < end; it++){


    Q.front = 0;
    Q.rear = 0;

    int src = labeled_node_vec[it];


    while(!isEmpty(&record_Q)){
      int current = get_front(&record_Q);
      residue[current] = 0;
      flags[current] = false;
      pi[current] = 0;
      dequeue(&record_Q);
    }

    record_Q.front = 0;
    record_Q.rear = 0;


    residue[src] = 1;
    flags[src] = true;
    
    enqueue(&Q, src);



    while(!isEmpty(&Q)){

      int v = get_front(&Q);
      

      if(g->degree[v] == 0){
        flags[v] = false;
        Q.front++;

        continue;
      }

      if(residue[v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[u] += (1-alpha) * residue[v] / g->degree[v];
          
          enqueue(&record_Q, u);

          if(g->degree[u] == 0){
            continue;
          }
          
          if(residue[u] / g->degree[u] > residuemax && !flags[u]){
            
            enqueue(&Q, u);
            flags[u] = true;
          }
        }
        pi[v] += alpha * residue[v];
        residue[v] = 0;
      }
      
      flags[v] = false;

      Q.front++;
    }




    for(int i = 0; i < Q.rear; i++){
      int index = Q.arr[i];
      if(pi[index] != 0){ 
        answer->push_back(Triplet<double>(index, src, pi[index]));
        answer->push_back(Triplet<double>(src, index, pi[index]));
        all_count += 1;
      }
    }


  }



  delete[] residue;
  delete[] pi;
  delete[] flags;

  residue = NULL;
  pi = NULL;
  flags = NULL;

  if (NULL != Q.arr)
    {
        free(Q.arr);
        Q.arr = NULL;
    }

  if (NULL != record_Q.arr)
    {
        free(record_Q.arr);
        record_Q.arr = NULL;
    }

  cout<<"End ForwardPush!"<<endl;
  return;

}


























void DirectedDynamicForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts
)
{
  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];

          if(g->outdegree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];

        double changed_value = alpha * residue[it][v];
        
        answer->push_back(Triplet<double>(src, v, changed_value));


        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        if(update_mat_record[row_v][col_v] != 0){
          update_mat_record[row_v][col_v] = 0;
        }

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }



    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }


  return;

}























void DirectedDynamicForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, 
vector<Triplet<double>>* answer, 
long long int& all_count, double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts
)
{


  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->indegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];
        double changed_value = alpha * residue[it][v];

        answer->push_back(Triplet<double>(src, v, changed_value));


        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        if(update_mat_record[row_v][col_v] != 0){
          update_mat_record[row_v][col_v] = 0;

        }

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}



















void Directed_Refresh_PPR_initialization(int start, int end, Graph* g, double residuemax, double reservemin,
vector<Triplet<double>>* answer, 
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
double** residue_transpose, 
double** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int iter,
int vertex_number){



  if(iter == 1){
    for(int i = start; i < end; i++){
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }

          double changed_value = pi[k][from_node] * 1 / j;
          pi[k][from_node] *= (j + 1) / j;
                
          answer->push_back(Triplet<double>(labeled_node_vec[k], from_node, changed_value));

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }



  if(iter == 1){
    
    for(int i = start; i < end; i++){
      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }

          double changed_value = pi_transpose[k][from_node] * 1 / j;

          answer->push_back(Triplet<double>(labeled_node_vec[k], from_node, changed_value));

          pi_transpose[k][from_node] *= (j + 1) / j;
          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;
          if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose[k][to_node] / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}














void Undirected_Refresh_PPR_initialization(int start, int end, UGraph* g, double residuemax, double reservemin,
vector<Triplet<double>>* answer, 
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number

){

  if(iter == 1){
    for(int i = start; i < end; i++){
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }
          
          double changed_value = pi[k][from_node] * 1 / j;
          answer->push_back(Triplet<double>(labeled_node_vec[k], from_node, changed_value));
          answer->push_back(Triplet<double>(from_node, labeled_node_vec[k], changed_value));

          pi[k][from_node] *= (j + 1) / j;

          
          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }
      }
    }
  }

}
































































































































































































































void DenseDynamicForwardPush(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
float** residue, 
float** pi,

bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,

int col_dim,

int nParts,
d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;


        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;


        continue;
      }


      if(residue[it][v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          
          if(residue[it][u] / g->degree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}

















void nodegree_DenseDynamicForwardPush(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
float** residue, 
float** pi,


bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,


int col_dim,

int nParts,


d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;


        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;


        continue;
      }



      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){
            pi[it][u] += alpha * residue[it][u];
            residue[it][src] += (1-alpha) * residue[it][u];
            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];




        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}



























void nodegree_DenseDynamicForwardPush_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
double** residue, 
double** pi,

bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,

int col_dim,
int nParts,

d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;


        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;

        continue;
      }


      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}






































void DenseDynamicForwardPush_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, 
vector<int>& labeled_node_vec, 
double** residue, 
double** pi,

bool** flags, 
Queue* queue_list,

unordered_map<int, int> &row_index_mapping,

int col_dim,
int nParts,

d_row_tree_mkl* subset_tree,
vector<int> &inner_group_mapping,
vector<int> &indicator

)
{

  int vertices_number = g->n;  

  
  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);


      if(g->degree[v] == 0){
        flags[it][v] = false;

        pi[it][v] = alpha * residue[it][v];

        queue_list[it].front++;


        continue;
      }


      if(residue[it][v] / g->degree[v] > residuemax){
        for(int j = 0; j < g->degree[v]; j++){
          int u = g->AdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->degree[v];


          if(g->degree[u] == 0){

            pi[it][u] += alpha * residue[it][v];
            residue[it][src] += (1-alpha) * residue[it][v];
            continue;
          }
          
          if(residue[it][u] / g->degree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }

        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}
















//undirected version
void DenseDynamicForwardPush_TransposeRefresh(int start, int end, 
vector<vector<column_tuple*>>& pi_transpose_storepush,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts,
vector<d_row_tree_mkl*> &d_row_tree_vec,
vector<int> &inner_group_mapping,
vector<int>& labeled_node_vec
)
{
    for(int i = start; i < end; i++){
        for(int j = 0; j < pi_transpose_storepush[i].size(); j++){
            int row = pi_transpose_storepush[i][j]->row;
            int col = labeled_node_vec[i];
            float pi = pi_transpose_storepush[i][j]->pi;

            int row_v = row / row_dim;
            int col_v = col / col_dim;

            if(row_v == number_of_d_row_tree){
                row_v--;
            }
            if(col_v == nParts){
                col_v--;
            }

            int inner_col_index = inner_group_mapping[col];

            (d_row_tree_vec[row_v])->dense_mat_mapping[col_v](
                row - row_v * row_dim, inner_col_index) += pi;

        }
    }

}
















//undirected version
void DenseDynamicForwardPush_TransposeRefresh_LP(int start, int end, 
vector<vector<column_tuple*>>& pi_transpose_storepush,
vector<vector<int>> &update_mat_record,
int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts,
vector<d_row_tree_mkl*> &d_row_tree_vec,
vector<int> &inner_group_mapping,
vector<int>& labeled_node_vec,
vector<MatrixXd> & transpose_trMat_vec
)
{
    for(int i = start; i < end; i++){
        for(int j = 0; j < pi_transpose_storepush[i].size(); j++){
            int row = pi_transpose_storepush[i][j]->row;

            int col = labeled_node_vec[i];

            float pi = pi_transpose_storepush[i][j]->pi;

            int row_v = row / row_dim;
            int col_v = col / col_dim;
            
            
            if(row_v == number_of_d_row_tree){
                row_v--;
            }
            if(col_v == nParts){
                col_v--;
            }

            int inner_col_index = inner_group_mapping[col];

            (d_row_tree_vec[row_v])->dense_mat_mapping[col_v](
                row - row_v * row_dim, inner_col_index) += pi;
            
            transpose_trMat_vec[row_v](col, row) += pi;

        }
    }

}
































































void DenseUndirected_Refresh_PPR_initialization(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,
int col_dim,
int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree,
unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

            if(j == 0){
              continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
                enqueue(&queue_list[k], from_node);
                flags[k][from_node] = true;
            }

            }

            int col_v = from_node / col_dim;

            if(col_v == nParts){
            col_v--;
            }


            int inner_col_index = inner_group_mapping[from_node];

            subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi[k][from_node] * 1 / j;


            pi[k][from_node] *= (j + 1) / j;

            residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
            residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
            }
            if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
            }
        }
        }
    }
    }

}



















void nodegree_DenseUndirected_Refresh_PPR_initialization(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl* subset_tree,

unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
          int to_node = g->AdjList[from_node][j];
          for(int k = start; k < end; k++){

              if(j == 0){
                continue;
                
              if(residue[k][from_node] > residuemax && !flags[k][from_node]){
                  enqueue(&queue_list[k], from_node);
                  flags[k][from_node] = true;
              }

              }

              int col_v = from_node / col_dim;

              if(col_v == nParts){
              col_v--;
              }


              int inner_col_index = inner_group_mapping[from_node];

              subset_tree->dense_mat_mapping[col_v](
              k, inner_col_index) += pi[k][from_node] * 1 / j;



              pi[k][from_node] *= (j + 1) / j;

              residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
              residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

              if(residue[k][from_node] > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
              }

              if(residue[k][to_node] > residuemax && !flags[k][to_node]){
              enqueue(&queue_list[k], to_node);
              flags[k][to_node] = true;
              }
          }
        }
    }
    }

}





















void nodegree_DenseUndirected_Refresh_PPR_initialization_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,
int col_dim,

int nParts,
vector<int> &inner_group_mapping,


d_row_tree_mkl* subset_tree,

unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

            if(j == 0){
              continue;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
                enqueue(&queue_list[k], from_node);
                flags[k][from_node] = true;
            }

            }

            

            int col_v = from_node / col_dim;
            
            if(col_v == nParts){
            col_v--;
            }


            int inner_col_index = inner_group_mapping[from_node];

            subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi[k][from_node] * 1 / j;



            pi[k][from_node] *= (j + 1) / j;

            residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
            residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
            }

            if(residue[k][to_node] > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
            }
        }
        }
    }
    }

}



















void DenseUndirected_Refresh_PPR_initialization_LP(int start, int end, UGraph* g, double residuemax, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,
int iter,
int vertex_number,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,


d_row_tree_mkl* subset_tree,

unordered_map<int, int> &row_index_mapping,
vector<int> &indicator,
int dynamic_ppr_start_iter
){

    if(iter < dynamic_ppr_start_iter){
      for(int i = start; i < end; i++){
        memset(residue[i], 0, sizeof(float) * vertex_number);
        memset(pi[i], 0, sizeof(float) * vertex_number);
        memset(flags[i], false, sizeof(bool) * vertex_number);
        queue_list[i].front = 0;
        queue_list[i].rear = 0;

        int src = labeled_node_vec[i];
        residue[i][src] = 1;
        flags[i][src] = true;
        enqueue(&queue_list[i], src);
      }
    }
    else{

    for(int i = 0; i < vertex_number; i++){
        int from_node = i;
        for(int j = g->former_degree[from_node]; j < g->degree[from_node]; j++){
        int to_node = g->AdjList[from_node][j];
        for(int k = start; k < end; k++){

            if(j == 0){
              continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
                enqueue(&queue_list[k], from_node);
                flags[k][from_node] = true;
            }

            }

            

            int col_v = from_node / col_dim;
            
            if(col_v == nParts){
            col_v--;
            }


            int inner_col_index = inner_group_mapping[from_node];

            subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi[k][from_node] * 1 / j;



            pi[k][from_node] *= (j + 1) / j;

            residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
            residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
            }
            if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
            }
        }
        }
    }
    }

}

















void Cache_pi_residue_to_parallel_blocks(
MatrixXf &residue,
MatrixXf &pi,
vector<SparseMatrix<float>> &sparse_pi_cache_vec,
vector<SparseMatrix<float>> &sparse_residue_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){
  sparse_pi_cache_vec[block_number] = pi.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();
  sparse_residue_cache_vec[block_number] = residue.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();
}

void Retrieve_pi_residue_from_parallel_blocks(
MatrixXf &residue,
MatrixXf &pi,
vector<SparseMatrix<float>> &sparse_pi_cache_vec,
vector<SparseMatrix<float>> &sparse_residue_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){


  for (int k_iter=0; k_iter<sparse_pi_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_pi_cache_vec[block_number], k_iter); it; ++it){
          pi(it.row(), it.col()) = it.value();
      }
  }

  for (int k_iter=0; k_iter<sparse_residue_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_residue_cache_vec[block_number], k_iter); it; ++it){
          residue(it.row(), it.col()) = it.value();
      }
  }

}


void Cache_pi_residue_transpose_to_parallel_blocks(
MatrixXf &residue_transpose,
MatrixXf &pi_transpose,
vector<SparseMatrix<float>> &sparse_residue_transpose_cache_vec,
vector<SparseMatrix<float>> &sparse_pi_transpose_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){

  sparse_pi_transpose_cache_vec[block_number] 
    = pi_transpose.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();
  sparse_residue_transpose_cache_vec[block_number] 
    = residue_transpose.block(0, start_column_number, count_labeled_node, block_column_size).sparseView();

}

void Retrieve_pi_residue_transpose_from_parallel_blocks(
MatrixXf &residue_transpose,
MatrixXf &pi_transpose,
vector<SparseMatrix<float>> &sparse_residue_transpose_cache_vec,
vector<SparseMatrix<float>> &sparse_pi_transpose_cache_vec,
int count_labeled_node,
int start_column_number,
int block_column_size,
int block_number
){

  for (int k_iter=0; k_iter<sparse_pi_transpose_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_pi_transpose_cache_vec[block_number], k_iter); it; ++it){
          pi_transpose(it.row(), it.col()) = it.value();
      }
  }

  for (int k_iter=0; k_iter<sparse_residue_transpose_cache_vec[block_number].outerSize(); ++k_iter){
      for (SparseMatrix<float, ColMajor, int>::InnerIterator it(sparse_residue_transpose_cache_vec[block_number], k_iter); it; ++it){
          residue_transpose(it.row(), it.col()) = it.value();
      }
  }


}









SparseMatrix<double> sparseBlock(SparseMatrix<double, ColMajor, int64_t> M,
        int ibegin, int jbegin, int icount, int jcount){
        //only for ColMajor Sparse Matrix
        
    typedef Triplet<double> Tri;
    
    
    assert(ibegin+icount <= M.rows());
    assert(jbegin+jcount <= M.cols());
    int Mj,Mi,i,j,currOuterIndex,nextOuterIndex;
    vector<Tri> tripletList;
    tripletList.reserve(M.nonZeros());

    for(j=0; j<jcount; j++){
        Mj=j+jbegin;
        currOuterIndex = M.outerIndexPtr()[Mj];
        nextOuterIndex = M.outerIndexPtr()[Mj+1];

        for(int a = currOuterIndex; a<nextOuterIndex; a++){
            Mi=M.innerIndexPtr()[a];

            if(Mi < ibegin) continue;
            if(Mi >= ibegin + icount) break;

            i=Mi-ibegin;    
            tripletList.push_back(Tri(i,j,M.valuePtr()[a]));
        }
    }
    SparseMatrix<double> matS(icount,jcount);
    matS.setFromTriplets(tripletList.begin(), tripletList.end());
    return matS;
}


void Log_sparse_matrix_entries_sparse_tree(
int k, int i,    
double reservemin, 
vector<sparse_d_row_tree_mkl*> &d_row_tree_vec,
unordered_map<int, vector<int>> &vec_mapping,
SparseMatrix<double, 0, int64_t> &subset_trMat
){

    SparseMatrix<double, 0, int64_t> &current_mat_mapping = d_row_tree_vec[k]->mat_mapping[i];


    current_mat_mapping.resize(0, 0);


    int temp_row_dim = d_row_tree_vec[k]->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(subset_trMat.rows(), current_group_size);



    if(i != vec_mapping.size() - 1){
      int start_col_index = i * vec_mapping[i].size();
      current_mat_mapping = sparseBlock(subset_trMat, 0, start_col_index, subset_trMat.rows(),  vec_mapping[i].size());

    }
    else{
      int start_col_index = i * vec_mapping[i-1].size();
      current_mat_mapping = sparseBlock(subset_trMat, 0, start_col_index, subset_trMat.rows(),  vec_mapping[i].size());

    }



}











void Log_sparse_matrix_entries(
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,
vector<long long int>& record_submatrices_nnz
){

    SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

    current_mat_mapping.resize(0, 0);


    record_submatrices_nnz[i] = 0;

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
    for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(it.value()/reservemin);
                record_submatrices_nnz[i]++;
            }
            else{
                it.valueRef() = 0;
            }


        }
    }

}






void No_Log_sparse_matrix_entries_LP(
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,
vector<long long int>& record_submatrices_nnz
){

    SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

    current_mat_mapping.resize(0, 0);

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
    for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){

            it.valueRef() = it.value();
            record_submatrices_nnz[i]++;

        }
    }

}



void Log_sparse_matrix_entries_LP(
int i,    
double reservemin, 
d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,
vector<long long int>& record_submatrices_nnz
){

    SparseMatrix<double, 0, int> &current_mat_mapping = subset_tree->mat_mapping[i];

    current_mat_mapping.resize(0, 0);




    record_submatrices_nnz[i] = 0;




    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();
    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();
    for (int k_iter=0; k_iter<subset_tree->mat_mapping[i].outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(subset_tree->mat_mapping[i], k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(1 + it.value()/reservemin);
                record_submatrices_nnz[i]++;
            }
            else{

                it.valueRef() = log10(1 + it.value()/reservemin);
                record_submatrices_nnz[i]++;
            }
            

        }
    }

}










void Log_sparse_matrix_entries_with_norm_computation(
// int k, 
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,

vector<int>& update_mat_tree_record,
int iter,
double delta,
int count_labeled_node,
int d,
vector<long long int>& record_submatrices_nnz
){
    
    SparseMatrix<double, 0, int> &old_mat_mapping = subset_tree->mat_mapping[i];

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();


    SparseMatrix<double, 0, int> current_mat_mapping;


    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();


    long long int temp_record_submatrices_nnz = 0;

    for (int k_iter=0; k_iter<current_mat_mapping.outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(current_mat_mapping, k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(it.value()/reservemin);

                temp_record_submatrices_nnz++;
            }
            else if(it.value() == 0){

            }
            else{
                it.valueRef() = 0;
            }

        }
    }



    double A_norm = current_mat_mapping.norm();



    double Ei_norm = (current_mat_mapping - old_mat_mapping).norm();



    delta = delta * sqrt(2);



    if( subset_tree->norm_B_Bid_difference_vec[i] + Ei_norm < delta * A_norm){
      update_mat_tree_record[i] = -1;
      current_mat_mapping.resize(0, 0);
      current_mat_mapping.data().squeeze();


    }
    else{
      update_mat_tree_record[i] = iter;
      old_mat_mapping.resize(0, 0);
      old_mat_mapping.data().squeeze();
      subset_tree->mat_mapping[i] = current_mat_mapping;

      record_submatrices_nnz[i] = temp_record_submatrices_nnz;


    }




}









void Log_sparse_matrix_entries_with_norm_computation_LP(
int i,    
double reservemin, 

d_row_tree_mkl* subset_tree,
unordered_map<int, vector<int>> &vec_mapping,

vector<int>& update_mat_tree_record,
int iter,
double delta,
int count_labeled_node,
int d,
vector<long long int>& record_submatrices_nnz
){


    
    SparseMatrix<double, 0, int> &old_mat_mapping = subset_tree->mat_mapping[i];

    int temp_row_dim = subset_tree->row_dim;
    int current_group_size = vec_mapping[i].size();



    SparseMatrix<double, 0, int> current_mat_mapping;


    current_mat_mapping.resize(temp_row_dim, current_group_size);
    current_mat_mapping = subset_tree->dense_mat_mapping[i].sparseView();


    long long int temp_record_submatrices_nnz = 0;

    for (int k_iter=0; k_iter<current_mat_mapping.outerSize(); ++k_iter){
        for (SparseMatrix<double, ColMajor, int>::InnerIterator it(current_mat_mapping, k_iter); it; ++it){
            if(it.value() > reservemin){
                it.valueRef() = log10(1 + it.value()/reservemin);

                temp_record_submatrices_nnz++;
            }
            else if(it.value() == 0){

            }
            else{
                it.valueRef() = log10(1 + it.value()/reservemin);
                temp_record_submatrices_nnz++;
            }

        }
    }



    double A_norm = current_mat_mapping.norm();



    double Ei_norm = (current_mat_mapping - old_mat_mapping).norm();



    delta = delta * sqrt(2);



    if( subset_tree->norm_B_Bid_difference_vec[i] + Ei_norm < delta * A_norm){
      update_mat_tree_record[i] = -1;
      current_mat_mapping.resize(0, 0);
      current_mat_mapping.data().squeeze();

    }
    else{
      update_mat_tree_record[i] = iter;
      old_mat_mapping.resize(0, 0);
      old_mat_mapping.data().squeeze();
      subset_tree->mat_mapping[i] = current_mat_mapping;

      record_submatrices_nnz[i] = temp_record_submatrices_nnz;

    }




}





















void DenseDirectedDynamicForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }



      if(residue[it][v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){
            continue;
          }
          
          if(residue[it][u] / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];

        int col_v = v / col_dim;


        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}















void nodegree_DenseDirectedDynamicForwardPush(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }




      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;


        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}














void nodegree_DenseDirectedDynamicForwardPush_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }




      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}






















void DenseDirectedDynamicForwardPush_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl * subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }



      if(residue[it][v] / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->outdegree[v];
          
          if(g->outdegree[u] == 0){

            continue;
          }
          
          if(residue[it][u] / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}
















void DenseDirectedDynamicForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          
          if(residue[it][u] / g->indegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}


























void nodegree_DenseDirectedDynamicForwardPushTranspose(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree

)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}





















void nodegree_DenseDirectedDynamicForwardPushTranspose_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree

)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue[it][v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          

          if(residue[it][u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}





































void DenseDirectedDynamicForwardPushTranspose_LP(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree

)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue[it][v] / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue[it][u] += (1-alpha) * residue[it][v] / g->indegree[v];

          if(g->indegree[u] == 0){

            continue;
          }
          
          if(residue[it][u] / g->indegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi[it][v] += alpha * residue[it][v];


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue[it][v];

        residue[it][v] = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}








































void DenseDirected_Refresh_PPR_initialization(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}





























void nodegree_DenseDirected_Refresh_PPR_initialization(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue, 
float** pi,
bool** flags, 
Queue* queue_list,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

          if(residue[k][from_node] > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }

          if(residue[k][to_node] > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}










void nodegree_DenseDirected_Refresh_PPR_initialization_LP(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,



int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;

            if(residue[k][from_node] > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;

          if(residue[k][from_node] > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }

          if(residue[k][to_node] > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}


























void DenseDirected_Refresh_PPR_initialization_Transpose(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue_transpose, 
float** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){




  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;
          if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose[k][to_node] / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }






}















void nodegree_DenseDirected_Refresh_PPR_initialization_Transpose(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

float** residue_transpose, 
float** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){




  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;

          if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }

          if(residue_transpose[k][to_node] > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }






}















void nodegree_DenseDirected_Refresh_PPR_initialization_Transpose_LP(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

double** residue_transpose, 
double** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){




  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;

          if(residue_transpose[k][from_node] > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }

          if(residue_transpose[k][to_node] > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }






}

























void DenseDirected_Refresh_PPR_initialization_LP(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

double** residue, 
double** pi,
bool** flags, 
Queue* queue_list,

double** residue_transpose, 
double** pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue[i], 0, sizeof(float) * vertex_number);
      memset(pi[i], 0, sizeof(float) * vertex_number);
      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];
      residue[i][src] = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){
          if(j == 0){
            continue;
            if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }
          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k , inner_col_index) += pi[k][from_node] * 1 / j;

          pi[k][from_node] *= (j + 1) / j;

          residue[k][from_node] -= pi[k][from_node] / (j+1) / alpha;
          residue[k][to_node] += (1 - alpha) * pi[k][from_node] / (j+1) / alpha;
          if(residue[k][from_node] / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue[k][to_node] / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }


  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){
      memset(residue_transpose[i], 0, sizeof(float) * vertex_number);
      memset(pi_transpose[i], 0, sizeof(float) * vertex_number);
      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];
      residue_transpose[i][src] = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){

          if(j == 0){
            continue;
            if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];

          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose[k][from_node] * 1 / j;

          pi_transpose[k][from_node] *= (j + 1) / j;

          residue_transpose[k][from_node] -= pi_transpose[k][from_node] / (j+1) / alpha;
          residue_transpose[k][to_node] += (1 - alpha) * pi_transpose[k][from_node] / (j+1) / alpha;
          if(residue_transpose[k][from_node] / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose[k][to_node] / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}



















void DenseDirected_Refresh_PPR_initialization_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int col_dim,
int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];

      residue(i,src) = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue(k, from_node) / j > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];
          
          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi(k, from_node) * 1 / j;

          pi(k, from_node) *= (j + 1) / j;

          residue(k, from_node) -= pi(k, from_node) / (j+1) / alpha;
          residue(k, to_node) += (1 - alpha) * pi(k, from_node) / (j+1) / alpha;
          if(residue(k, from_node) / j > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }
          if(residue(k, to_node) / j > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}



















void nodegree_DenseDirected_Refresh_PPR_initialization_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags[i], false, sizeof(bool) * vertex_number);
      queue_list[i].front = 0;
      queue_list[i].rear = 0;

      int src = labeled_node_vec[i];

      residue(i,src) = 1;
      flags[i][src] = true;
      enqueue(&queue_list[i], src);
    }
  }
  else{

    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_outdegree[from_node]; j < g->outdegree[from_node]; j++){
        int to_node = g->outAdjList[from_node][j];
        
        for(int k = start; k < end; k++){

          if(j == 0){
            continue;

            if(residue(k, from_node) > residuemax && !flags[k][from_node]){
              enqueue(&queue_list[k], from_node);
              flags[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];


          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi(k, from_node) * 1 / j;

          pi(k, from_node) *= (j + 1) / j;

          residue(k, from_node) -= pi(k, from_node) / (j+1) / alpha;
          residue(k, to_node) += (1 - alpha) * pi(k, from_node) / (j+1) / alpha;

          if(residue(k, from_node) > residuemax && !flags[k][from_node]){
            enqueue(&queue_list[k], from_node);
            flags[k][from_node] = true;
          }

          if(residue(k, to_node) > residuemax && !flags[k][to_node]){
            enqueue(&queue_list[k], to_node);
            flags[k][to_node] = true;
          }
        }

      }
    }
  }




}



























void DenseDirected_Refresh_PPR_initialization_Transpose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue_transpose, 
MatrixXf &pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];

      residue_transpose(i, src) = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{
    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){
          if(j == 0){
            continue;

            if(residue_transpose(k, from_node) / j > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];


          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose(k, from_node) * 1 / j;

          pi_transpose(k, from_node) *= (j + 1) / j;

          residue_transpose(k, from_node) -= pi_transpose(k, from_node) / (j+1) / alpha;
          residue_transpose(k, to_node) += (1 - alpha) * pi_transpose(k, from_node) / (j+1) / alpha;
          if(residue_transpose(k, from_node) / j > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }
          if(residue_transpose(k, to_node) / j > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}





































void nodegree_DenseDirected_Refresh_PPR_initialization_Transpose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin,
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue_transpose, 
MatrixXf &pi_transpose,
bool** flags_transpose, 
Queue * queue_list_transpose,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

int iter,
int vertex_number,

d_row_tree_mkl* subset_tree,
int dynamic_ppr_start_iter
){

  if(iter < dynamic_ppr_start_iter){
    for(int i = start; i < end; i++){

      memset(flags_transpose[i], false, sizeof(bool) * vertex_number);

      queue_list_transpose[i].front = 0;
      queue_list_transpose[i].rear = 0;

      int src = labeled_node_vec[i];

      residue_transpose(i, src) = 1;
      flags_transpose[i][src] = true;
      enqueue(&queue_list_transpose[i], src);
    }

  }
  else{
    for(int i = 0; i < vertex_number; i++){
      int from_node = i;
      for(int j = g->former_indegree[from_node]; j < g->indegree[from_node]; j++){
        int to_node = g->inAdjList[from_node][j];

        for(int k = start; k < end; k++){
          if(j == 0){
            continue;


            if(residue_transpose(k, from_node) > residuemax && !flags_transpose[k][from_node]){
              enqueue(&queue_list_transpose[k], from_node);
              flags_transpose[k][from_node] = true;
            }

          }


          int col_v = from_node / col_dim;
          
          if(col_v == nParts){
            col_v--;
          }

          int inner_col_index = inner_group_mapping[from_node];


          subset_tree->dense_mat_mapping[col_v](
            k, inner_col_index) += pi_transpose(k, from_node) * 1 / j;

          pi_transpose(k, from_node) *= (j + 1) / j;

          residue_transpose(k, from_node) -= pi_transpose(k, from_node) / (j+1) / alpha;
          residue_transpose(k, to_node) += (1 - alpha) * pi_transpose(k, from_node) / (j+1) / alpha;

          if(residue_transpose(k, from_node) > residuemax && !flags_transpose[k][from_node]){
            enqueue(&queue_list_transpose[k], from_node);
            flags_transpose[k][from_node] = true;
          }

          if(residue_transpose(k, to_node) > residuemax && !flags_transpose[k][to_node]){
            enqueue(&queue_list_transpose[k], to_node);
            flags_transpose[k][to_node] = true;
          }
        }
        
      }
    }
  }


}









































void DenseDirectedDynamicForwardPush_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
MatrixXf & residue, 
MatrixXf & pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue(it, v) / g->outdegree[v] > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];

          residue(it, u) += (1-alpha) * residue(it, v) / g->outdegree[v];

          if(g->outdegree[u] == 0){

            continue;
          }
          
          if(residue(it, u) / g->outdegree[u] > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }

        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }


        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);

        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }


  return;

}























void nodegree_DenseDirectedDynamicForwardPush_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 
MatrixXf & residue, 
MatrixXf & pi,
bool** flags, 
Queue* queue_list,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,

d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->outdegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue(it, v) > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];

          residue(it, u) += (1-alpha) * residue(it, v) / g->outdegree[v];

          if(g->outdegree[u] == 0){

            continue;
          }
          

          if(residue(it, u) > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }

        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }


        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);

        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }


    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }


  return;

}


































void DenseDirectedDynamicForwardPushTranspose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 

double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }

      if(residue(it, v) / g->indegree[v] > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue(it, u) += (1-alpha) * residue(it, v) / g->indegree[v];

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue(it, u) / g->indegree[u] > residuemax && !flags[it][u]){

            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);


        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;

  }


  return;

}






























void nodegree_DenseDirectedDynamicForwardPushTranspose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 

double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,


int col_dim,

int nParts,
vector<int> &inner_group_mapping,
d_row_tree_mkl* subset_tree
)
{

  int vertices_number = g->n;  

  for(int it = start; it < end; it++){

    int src = labeled_node_vec[it];

    while(!isEmpty(&queue_list[it])){

      int v = get_front(&queue_list[it]);

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue(it, v) > residuemax){
        for(int j = 0; j < g->indegree[v]; j++){
          int u = g->inAdjList[v][j];
          residue(it, u) += (1-alpha) * residue(it, v) / g->indegree[v];

          if(g->indegree[u] == 0){
            continue;
          }
          

          if(residue(it, u) > residuemax && !flags[it][u]){
            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi(it, v) += alpha * residue(it, v);


        int col_v = v / col_dim;
        
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];


        subset_tree->dense_mat_mapping[col_v](
          it, inner_col_index) += alpha * residue(it, v);


        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }

    queue_list[it].front = 0;
    queue_list[it].rear = 0;

  }


  return;

}

















































































































































































































































































































































































































































void DenseDirectedDynamicBackwardPushTranspose_MatrixVersion(int start, int end, Graph* g, double residuemax, double reservemin, 
double alpha, vector<int>& labeled_node_vec, 

MatrixXf &residue, 
MatrixXf &pi,
bool** flags, 
Queue* queue_list,

int row_dim,
int col_dim,
int number_of_d_row_tree,
int nParts,
vector<int> &inner_group_mapping,
vector<d_row_tree_mkl*> &d_row_tree_vec
)
{

  int vertices_number = g->n;  


  for(int it = start; it < end; it++){


    int src = labeled_node_vec[it];



    while(!isEmpty(&queue_list[it])){
      int v = get_front(&queue_list[it]);
      

      if(g->indegree[v] == 0){
        flags[it][v] = false;
        queue_list[it].front++;

        continue;
      }


      if(residue(it, v) > residuemax){
        for(int j = 0; j < g->outdegree[v]; j++){
          int u = g->outAdjList[v][j];
          residue(it, u) += (1-alpha) * residue(it, v) / g->indegree[u];

          if(g->indegree[u] == 0){
            continue;
          }
          
          if(residue(it, u) > residuemax && !flags[it][u]){

            enqueue(&queue_list[it], u);
            flags[it][u] = true;
          }
        }
        pi(it, v) += alpha * residue(it, v);
        
        int row_v = it / row_dim;
        int col_v = v / col_dim;
        if(row_v == number_of_d_row_tree){
          row_v--;
        }
        if(col_v == nParts){
          col_v--;
        }

        int inner_col_index = inner_group_mapping[v];

        d_row_tree_vec[row_v]->dense_mat_mapping[col_v](
          it - row_v * row_dim, inner_col_index) += alpha * residue(it, v);

        residue(it, v) = 0;
      }
      
      flags[it][v] = false;

      queue_list[it].front++;
    }



    queue_list[it].front = 0;
    queue_list[it].rear = 0;


  }

  return;

}






























































class d_row_tree_mkl_sparse_dynamic{
  public:
  int nParts;

  int level_p;
  int total_nodes;
  vector<mat*> matrix_vec;
  int largest_level_start_index;
  int largest_level_end_index;
  vector<mat*> hierarchy_matrix_vec;

  Eigen::VectorXd vS_cur_iter;

  Eigen::MatrixXd U_cur_iter;

  vector<mat*> near_n_matrix_vec;
  vector<mat*> less_near_n;

  // unordered_map<int, SparseMatrix<double, 0, int>> mat_mapping;
  // unordered_map<int, SparseMatrix<double, 0, int>> svd_mat_mapping;
  unordered_map<int, SparseMatrix<double, RowMajor, int>> svd_mat_mapping;

  // unordered_map<int, MatrixXd> dense_mat_mapping;
  // unordered_map<int, SparseMatrix<double, 0, int>> ppr_mat_mapping;

  int start_row;
  int end_row;
  int hierarchy_n;
  int row_dim;

  vector<double> norm_B_Bid_difference_vec;


  d_row_tree_mkl_sparse_dynamic(int row_dim, int d, int nParts, int hierarchy_n, 
  // unordered_map<int, vector<int>> &vec_mapping,
  int start_row, int end_row,
  int common_group_size, int final_group_size){

    this->nParts = nParts;

    this->row_dim = row_dim;

    this->start_row = start_row;
    this->end_row = end_row;
    this->hierarchy_n = hierarchy_n;


    // norm_B_Bid_difference_vec.resize(vec_mapping.size());
    norm_B_Bid_difference_vec.resize(nParts);

    for(int i = 0; i < nParts; i++){
      // vector<int> &v = vec_mapping[i];
      // int current_group_size = vec_mapping[i].size();

      // mat_mapping[i].resize(row_dim, current_group_size);
      // svd_mat_mapping[i].resize(row_dim, current_group_size);
      // dense_mat_mapping[i].resize(row_dim, current_group_size);
      // ppr_mat_mapping[i].resize(row_dim, current_group_size);
      
      norm_B_Bid_difference_vec[i] = 0;
    }


    level_p = log(nParts) / log(hierarchy_n) + 1;

    assert(pow(hierarchy_n, level_p - 1) == nParts);

    total_nodes = (pow(hierarchy_n, level_p) - 1) / (hierarchy_n - 1);

    tcout<<"total_nodes = "<<total_nodes<<endl;

    largest_level_start_index = total_nodes - nParts;

    tcout<<"largest_level_start_index = "<<largest_level_start_index<<endl;
    
    largest_level_end_index = total_nodes - 1;
    
    tcout<<"largest_level_end_index = "<<largest_level_end_index<<endl;
    
    hierarchy_matrix_vec.resize(total_nodes);

    for(int i = 0; i < total_nodes; i++){
      hierarchy_matrix_vec[i] = matrix_new(row_dim, d);
    }

    near_n_matrix_vec.resize(total_nodes - nParts);

    less_near_n.resize(total_nodes - nParts);
  }
  
};











































































void All_initialization_and_push_final(
int start, int end, Graph* g, 
double residuemax, double reservemin,
double alpha, 
int col_dim,
int nParts,
int iter,
int vertex_number,
int dynamic_ppr_start_iter,
int thread_number,
vector<int>& labeled_node_vec,
vector<vector<vector<pair<int, float>>>>& tripletList,
vector<vector<vector<pair<int, float>>>>& tripletList_transpose,
vector<vector<vector<pair<int, float>>>>& residue_pair_list,
vector<vector<vector<pair<int, float>>>>& residue_transpose_pair_list,
vector<int> & inner_group_mapping,
vector<int> & vertex_mapping,
vector<pair<int, int>> & edge_vec
){
  int upper_nnz = ceil(1 / residuemax / alpha);

  int reserve_size = 1 / residuemax / alpha;

  vector<int> queue_vec;

  third_float_map pi_map;
  third_float_map pi_transpose_map;
  third_float_map residue;
  third_float_map residue_transpose;

  third_bool_map flags;
  third_bool_map flags_transpose;

  if(iter < dynamic_ppr_start_iter){
    for(int it = start; it < end; it++){
      
    if(iter <= dynamic_ppr_start_iter - 1){
      for(int col_v = 0; col_v < nParts; col_v++){
        vector<pair<int, float>>().swap(residue_pair_list[col_v][it]);
        vector<pair<int, float>>().swap(tripletList[col_v][it]);
        vector<pair<int, float>>().swap(residue_transpose_pair_list[col_v][it]);
        vector<pair<int, float>>().swap(tripletList_transpose[col_v][it]);
      }
    }

      //Push initialization
      pi_map.reserve(reserve_size);

      residue.reserve(reserve_size);

      flags.reserve(reserve_size);

      int src = labeled_node_vec[it];
      residue[src] = 1;
      
      flags[src] = true;

      queue_vec.push_back(src);



      //Push
      src = labeled_node_vec[it];

      int queue_cur_front = 0;

      while(queue_cur_front < queue_vec.size()){

        int v = queue_vec[queue_cur_front];

        if(g->outdegree[v] == 0){
          flags[v] = false;
          queue_cur_front++;
          continue;
        }



        if(residue[v] > residuemax){
          for(int j = 0; j < g->outdegree[v]; j++){
            int u = g->outAdjList[v][j];
            residue[u] += (1-alpha) * residue[v] / g->outdegree[v];
            
            if(g->outdegree[u] == 0){
              continue;
            }
            
            if(residue[u] > residuemax && !flags[u]){
              queue_vec.push_back(u);
              flags[u] = true;
            }
          }

          pi_map[v] += alpha * residue[v];

          residue[v] = 0;
        }
        
        flags[v] = false;

        queue_cur_front++;
      }


      vector<int>().swap(queue_vec);

      third_bool_map().swap(flags);




      if(iter < dynamic_ppr_start_iter - 1){
        third_float_map().swap(residue);
      }
      else{
        third_float_map& cur_residue_map = residue;
        for(auto& key_value: cur_residue_map){
          int index = key_value.first;
          double residue_value = key_value.second;
          if(residue_value <= 0){
            continue;
          }

          int col_v = vertex_mapping[index];
          int inner_col_index = inner_group_mapping[index];
          
          residue_pair_list[col_v][it].push_back(make_pair(inner_col_index, residue_value));

        }

        third_float_map().swap(residue);      
      }

      third_float_map& cur_map = pi_map;
      for(auto& key_value: cur_map){
        int index = key_value.first;
        double ppr_value = key_value.second;
        // if(ppr_value <= reservemin){
        //   continue;
        // }
        if(ppr_value <= 0){
          continue;
        }

        int col_v = vertex_mapping[index];
        int inner_col_index = inner_group_mapping[index];
        tripletList[col_v][it].push_back(make_pair(inner_col_index, ppr_value));
      }

      third_float_map().swap(pi_map);






      //Transpose Push initialization
      residue_transpose.reserve(reserve_size);
      pi_transpose_map.reserve(reserve_size);

      flags_transpose.reserve(reserve_size);

      src = labeled_node_vec[it];


      residue_transpose[src] = 1;
      flags_transpose[src] = true;

      queue_vec.push_back(src);



      //Transpose Push
      queue_cur_front = 0;

      while(queue_cur_front < queue_vec.size()){

        int v = queue_vec[queue_cur_front];

        if(g->indegree[v] == 0){
          flags_transpose[v] = false;
          queue_cur_front++;
          continue;
        }

        if(residue_transpose[v] > residuemax){
          for(int j = 0; j < g->indegree[v]; j++){
            int u = g->inAdjList[v][j];
            residue_transpose[u] += (1-alpha) * residue_transpose[v] / g->indegree[v];

            if(g->indegree[u] == 0){
              continue;
            }
            
            if(residue_transpose[u] > residuemax && !flags_transpose[u]){
              queue_vec.push_back(u);
              flags_transpose[u] = true;
            }
          }

          pi_transpose_map[v] += alpha * residue_transpose[v];

          residue_transpose[v] = 0;
        }
        
        flags_transpose[v] = false;

        queue_cur_front++;
      }






      vector<int>().swap(queue_vec);
      third_bool_map().swap(flags_transpose);



      if(iter < dynamic_ppr_start_iter - 1){
        third_float_map().swap(residue_transpose);
      }
      else{
        third_float_map& cur_transpose_residue_map = residue_transpose;
        for(auto& key_value: cur_transpose_residue_map){
          int index = key_value.first;
          double transpose_residue_value = key_value.second;
          if(transpose_residue_value <= 0){
            continue;
          }

          int col_v = vertex_mapping[index];
          int inner_col_index = inner_group_mapping[index];
          
          residue_transpose_pair_list[col_v][it].push_back(make_pair(inner_col_index, transpose_residue_value));
        }

        third_float_map().swap(residue_transpose);      
      }



      third_float_map& cur_transpose_map = pi_transpose_map;
      for(auto& key_value: cur_transpose_map){
        int index = key_value.first;
        double ppr_value = key_value.second;
        // if(ppr_value <= reservemin){
        //   continue;
        // }
        if(ppr_value <= 0){
          continue;
        }

        int col_v = vertex_mapping[index];
        int inner_col_index = inner_group_mapping[index];      
          
        tripletList_transpose[col_v][it].push_back(make_pair(inner_col_index, ppr_value));
      }

      third_float_map().swap(pi_transpose_map);


    }
  }
  else{
    for(int it = start; it < end; it++){

      //Push initialization
      pi_map.reserve(reserve_size);
      residue.reserve(reserve_size);

      flags.reserve(reserve_size);

      auto restore_start_time = std::chrono::system_clock::now();

      // restore elements from pi, pi_difference, residue
      for(int parts = 0; parts < nParts; parts++){
        int start_col_index = parts * col_dim;
        for(auto pair: tripletList[parts][it]){
          int index = start_col_index + pair.first;
          double value = pair.second;
          pi_map[index] = value;
        }

        vector<pair<int, float>>().swap(tripletList[parts][it]);
      }



      for(int parts = 0; parts < nParts; parts++){
        int start_col_index = parts * col_dim;
        for(auto pair: residue_pair_list[parts][it]){
          int index = start_col_index + pair.first;
          double value = pair.second;
          residue[index] = value;
        }
        vector<pair<int, float>>().swap(residue_pair_list[parts][it]);
      }


      auto restore_end_time = chrono::system_clock::now();
      auto elapsed_restore_time = chrono::duration_cast<std::chrono::seconds>(restore_end_time - restore_start_time);
      tcout<< "it = "<<it<< ", restore time: "<< elapsed_restore_time.count() << endl;








      third_int_map out_degree_map;
      out_degree_map.reserve(20000);


      for(int i = 0; i < edge_vec.size(); i++){
        int from_node = edge_vec[i].first;
        int to_node = edge_vec[i].second;
        
        double j = max(g->former_outdegree[from_node], out_degree_map[from_node]);

        if(j == 0){
          // break;
          // pi_map[from_node] += alpha * residue[from_node];
          residue[from_node] = 0;
          out_degree_map[from_node] = 1;
          continue;
        }

        out_degree_map[from_node] = j + 1;

        pi_map[from_node] *= (double)(j + 1.0) / double(j);
        residue[from_node] -= pi_map[from_node] / (double)(j+1.0) / alpha;
        residue[to_node] += (1 - alpha) * pi_map[from_node] / (double)(j+1.0) / alpha;


        if(residue[from_node] > residuemax && !flags[from_node]){
          queue_vec.push_back(from_node);
          flags[from_node] = true;
        }
        if(residue[to_node] > residuemax && !flags[to_node]){
          queue_vec.push_back(to_node);
          flags[to_node] = true;
        }
      }

      third_int_map().swap(out_degree_map);








      //Push
      int src = labeled_node_vec[it];

      int queue_cur_front = 0;

      while(queue_cur_front < queue_vec.size()){
        
        int v = queue_vec[queue_cur_front];

        if(g->outdegree[v] == 0){
          flags[v] = false;
          queue_cur_front++;
          continue;
        }



        if(residue[v] > residuemax){
          for(int j = 0; j < g->outdegree[v]; j++){
            int u = g->outAdjList[v][j];
            residue[u] += (1-alpha) * residue[v] / g->outdegree[v];
            
            if(g->outdegree[u] == 0){
              continue;
            }
            
            if(residue[u] > residuemax && !flags[u]){
              queue_vec.push_back(u);
              flags[u] = true;
            }
          }


          pi_map[v] += alpha * residue[v];

          residue[v] = 0;
        }
        
        flags[v] = false;

        queue_cur_front++;
      }


      vector<int>().swap(queue_vec);

      third_bool_map().swap(flags);

      auto forward_ppr_end_time = chrono::system_clock::now();
      auto elapsed_forward_ppr_time = chrono::duration_cast<std::chrono::seconds>(forward_ppr_end_time - restore_end_time);
      tcout<< "it = "<<it<< ", forward ppr time: "<< elapsed_forward_ppr_time.count() << endl;



      third_float_map& cur_map_for_pair = pi_map;
      for(auto& key_value: cur_map_for_pair){
        int index = key_value.first;
        double ppr_value = key_value.second;
        // if(ppr_value <= reservemin){
        //   continue;
        // }
        if(ppr_value <= 0){
          continue;
        }

        int col_v  = vertex_mapping[index];
        int inner_col_index = inner_group_mapping[index];
        tripletList[col_v][it].push_back(make_pair(inner_col_index, ppr_value));
      }

      third_float_map& cur_residue_map = residue;
      for(auto& key_value: cur_residue_map){
        int index = key_value.first;
        double residue_value = key_value.second;
        if(residue_value <= 0){
          continue;
        }

        int col_v = vertex_mapping[index];
        int inner_col_index = inner_group_mapping[index];
        
        residue_pair_list[col_v][it].push_back(make_pair(inner_col_index, residue_value));
      }


      auto store_forward_ppr_end_time = chrono::system_clock::now();
      auto elapsed_store_forward_ppr_time = chrono::duration_cast<std::chrono::seconds>(store_forward_ppr_end_time - forward_ppr_end_time);
      tcout<< "it = "<<it<< ", store forward ppr time: "<< elapsed_store_forward_ppr_time.count() << endl;




      //Transpose Push Initialization
      pi_transpose_map.reserve(reserve_size);
      // pi_transpose_difference_map.reserve(reserve_size / 10);
      residue_transpose.reserve(reserve_size);

      // third_bool_map flags_transpose;
      flags_transpose.reserve(reserve_size);

      // restore elements from pi_transpose, pi_transpose_difference, residue_transpose

      for(int parts = 0; parts < nParts; parts++){
        int start_col_index = parts * col_dim;
        for(auto pair: tripletList_transpose[parts][it]){
          int index = start_col_index + pair.first;
          double value = pair.second;
          pi_transpose_map[index] = value;
        }
        vector<pair<int, float>>().swap(tripletList_transpose[parts][it]);
      }


      for(int parts = 0; parts < nParts; parts++){
        int start_col_index = parts * col_dim;
        for(auto pair: residue_transpose_pair_list[parts][it]){
          int index = start_col_index + pair.first;
          double value = pair.second;
          residue_transpose[index] = value;
        }
        vector<pair<int, float>>().swap(residue_transpose_pair_list[parts][it]);
      }



      auto restore_transpose_end_time = chrono::system_clock::now();
      auto elapsed_restore_transpose_time = chrono::duration_cast<std::chrono::seconds>(restore_transpose_end_time - store_forward_ppr_end_time);
      tcout<< "it = "<<it<< ", restore transpose time: "<< elapsed_restore_transpose_time.count() << endl;








      third_int_map in_degree_map;
      in_degree_map.reserve(20000);


      for(int i = 0; i < edge_vec.size(); i++){
        int from_node = edge_vec[i].second;
        int to_node = edge_vec[i].first;
        
        double j = max(g->former_indegree[from_node], in_degree_map[from_node]);

        if(j == 0){
          // break;
          // pi_transpose_map[from_node] += alpha * residue[from_node];
          residue_transpose[from_node] = 0;
          in_degree_map[from_node] = 1;
          continue;
        }

        in_degree_map[from_node] = j + 1;

        pi_transpose_map[from_node] *= (double)(j + 1.0) / double(j);
        residue_transpose[from_node] -= pi_transpose_map[from_node] / (double)(j+1.0) / alpha;
        residue_transpose[to_node] += (1 - alpha) * pi_transpose_map[from_node] / (double)(j+1.0) / alpha;


        if(residue_transpose[from_node] > residuemax && !flags_transpose[from_node]){
          queue_vec.push_back(from_node);
          flags_transpose[from_node] = true;
        }
        if(residue_transpose[to_node] > residuemax && !flags_transpose[to_node]){
          queue_vec.push_back(to_node);
          flags_transpose[to_node] = true;
        }
      }

      third_int_map().swap(in_degree_map);






      //Transpose Push
      queue_cur_front = 0;

      while(queue_cur_front < queue_vec.size()){

        int v = queue_vec[queue_cur_front];

        if(g->indegree[v] == 0){
          flags_transpose[v] = false;
          queue_cur_front++;
          continue;
        }

        if(residue_transpose[v] > residuemax){
          for(int j = 0; j < g->indegree[v]; j++){
            int u = g->inAdjList[v][j];
            residue_transpose[u] += (1-alpha) * residue_transpose[v] / g->indegree[v];

            if(g->indegree[u] == 0){
              continue;
            }
            
            if(residue_transpose[u] > residuemax && !flags_transpose[u]){
              queue_vec.push_back(u);
              flags_transpose[u] = true;
            }
          }

          pi_transpose_map[v] += alpha * residue_transpose[v];

          residue_transpose[v] = 0;
        }
        
        flags_transpose[v] = false;

        queue_cur_front++;
      }





      third_bool_map().swap(flags_transpose);
      vector<int>().swap(queue_vec);




      auto transpose_ppr_end_time = chrono::system_clock::now();
      auto elapsed_transpose_ppr_time = chrono::duration_cast<std::chrono::seconds>(transpose_ppr_end_time - restore_transpose_end_time);
      tcout<< "it = "<<it<< ", transpose ppr time: "<< elapsed_transpose_ppr_time.count() << endl;




      //insert into triplets

      third_float_map& cur_transpose_map_for_pair = pi_transpose_map;
      for(auto& key_value: cur_transpose_map_for_pair){
        int index = key_value.first;
        double ppr_value = key_value.second;
        // if(ppr_value <= reservemin){
        //   continue;
        // }
        if(ppr_value <= 0){
          continue;
        }

        int col_v = vertex_mapping[index];
        int inner_col_index = inner_group_mapping[index];
        tripletList_transpose[col_v][it].push_back(make_pair(inner_col_index, ppr_value));
      }



      third_float_map& cur_transpose_residue_map = residue_transpose;
      for(auto& key_value: cur_transpose_residue_map){
        int index = key_value.first;
        double transpose_residue_value = key_value.second;
        if(transpose_residue_value <= 0){
          continue;
        }

        int col_v = vertex_mapping[index];
        int inner_col_index = inner_group_mapping[index];
        residue_transpose_pair_list[col_v][it].push_back(make_pair(inner_col_index, transpose_residue_value));
      }



      third_float_map().swap(residue);
      third_float_map().swap(pi_map);


      third_float_map().swap(residue_transpose);
      third_float_map().swap(pi_transpose_map);

      auto store_transpose_ppr_end_time = chrono::system_clock::now();
      auto elapsed_store_transpose_ppr_time = chrono::duration_cast<std::chrono::seconds>(store_transpose_ppr_end_time - transpose_ppr_end_time);
      tcout<< "it = "<<it<< ", store transpose ppr time: "<< elapsed_store_transpose_ppr_time.count() << endl;

    }



  }




}

































































void Log_sparse_matrix_entries_LP_Robinhood_two_with_transpose_nparts_triplet(
int submatrix_index,    
double reservemin, 
d_row_tree_mkl_sparse_dynamic* subset_tree,
// unordered_map<int, vector<int>> &vec_mapping,
int common_group_size,
int final_group_size,
vector<vector<vector<pair<int, float>>>>& tripletList,
vector<vector<vector<pair<int, float>>>>& tripletList_transpose,
int nParts,
double residuemax,
double alpha,
int iter,
int dynamic_ppr_start_iter
){

  SparseMatrix<double, RowMajor, int> &svd_mat_mapping = subset_tree->svd_mat_mapping[submatrix_index];

  int count_labeled_node = subset_tree->row_dim;

  int current_group_size;
  if(submatrix_index == nParts - 1){
    current_group_size = final_group_size;
  }
  else{
    current_group_size = common_group_size;
  }

  svd_mat_mapping.resize(count_labeled_node, current_group_size);


  // int INTERVAL_NUMBER = 10;
  int INTERVAL_NUMBER = 30;
  int each_interval_number = count_labeled_node / INTERVAL_NUMBER;
  SparseMatrix<double, RowMajor, int>::IndexVector wi(svd_mat_mapping.outerSize());
  for(int i = 1; i <= INTERVAL_NUMBER; i++){
    wi.setZero();

    int start_index = (i - 1) * each_interval_number;
    int end_index = i * each_interval_number;

    
    for(int j = start_index; j < end_index; j++){
      wi(j) += tripletList[submatrix_index][j].size();
      wi(j) += tripletList_transpose[submatrix_index][j].size();
    }


    svd_mat_mapping.reserve(wi);


    for(int j = start_index; j < end_index; j++){
      for(auto it: tripletList[submatrix_index][j]){
        // int inner_col_index = it.col();
        // double ppr_value = it.value();
        int inner_col_index = it.first;
        double ppr_value = it.second;
        // if(ppr_value <= 0){
        //   continue;
        // }
        svd_mat_mapping.insertBackUncompressed(j, inner_col_index) = ppr_value;
      }
    }



    for(int j = start_index; j < end_index; j++){
      for(auto it: tripletList_transpose[submatrix_index][j]){
        // int inner_col_index = it.col();
        // double ppr_value = it.value();
        int inner_col_index = it.first;
        double ppr_value = it.second;
        // if(ppr_value <= 0){
        //   continue;
        // }
        svd_mat_mapping.insertBackUncompressed(j, inner_col_index) = ppr_value;
      }
    }

    svd_mat_mapping.collapseDuplicates(internal::scalar_sum_op<double, double>());

    if(iter < dynamic_ppr_start_iter - 1){
      for(int j = start_index; j < end_index; j++){
        vector<pair<int, float>>().swap(tripletList[submatrix_index][j]);
        vector<pair<int, float>>().swap(tripletList_transpose[submatrix_index][j]);
      }
    }
  }



  for (int k_iter=0; k_iter<svd_mat_mapping.outerSize(); ++k_iter){
      for (SparseMatrix<double, RowMajor, int>::InnerIterator it(svd_mat_mapping, k_iter); it; ++it){
          // if(it.value() > reservemin){
              // it.valueRef() = log10(it.value()/reservemin);
              it.valueRef() = log10(1 + it.value()/reservemin);
              // it.valueRef() = log10(10 + it.value()/reservemin);
          // }
          // else{
          //     it.valueRef() = 0;
          // }
      }
  }


}












































































void Log_sparse_matrix_entries_LP_Robinhood_two_with_transpose_nparts_triplet_norm_computation(
int submatrix_index,    
double reservemin, 
d_row_tree_mkl_sparse_dynamic* subset_tree,
int common_group_size,
int final_group_size,
vector<vector<vector<pair<int, float>>>>& tripletList,
vector<vector<vector<pair<int, float>>>>& tripletList_transpose,
vector<int>& update_mat_tree_record,
int nParts,
double residuemax,
double alpha,
double delta,
int iter,
int dynamic_ppr_start_iter
){

  int count_labeled_node = subset_tree->row_dim;

  int current_group_size;
  if(submatrix_index == nParts - 1){
    current_group_size = final_group_size;
  }
  else{
    current_group_size = common_group_size;
  }


  SparseMatrix<double, RowMajor, int> current_svd_mat_mapping;
  current_svd_mat_mapping.resize(count_labeled_node, current_group_size);


  int INTERVAL_NUMBER = 30;
  int each_interval_number = count_labeled_node / INTERVAL_NUMBER;
  SparseMatrix<double, RowMajor, int>::IndexVector wi(current_svd_mat_mapping.outerSize());
  for(int i = 1; i <= INTERVAL_NUMBER; i++){
    wi.setZero();

    int start_index = (i - 1) * each_interval_number;
    int end_index = i * each_interval_number;

    
    for(int j = start_index; j < end_index; j++){
      wi(j) += tripletList[submatrix_index][j].size();
      wi(j) += tripletList_transpose[submatrix_index][j].size();
    }


    current_svd_mat_mapping.reserve(wi);


    for(int j = start_index; j < end_index; j++){
      for(auto it: tripletList[submatrix_index][j]){
        // int inner_col_index = it.col();
        // double ppr_value = it.value();
        int inner_col_index = it.first;
        double ppr_value = it.second;
        // if(ppr_value <= 0){
        //   continue;
        // }
        current_svd_mat_mapping.insertBackUncompressed(j, inner_col_index) = ppr_value;
      }
    }



    for(int j = start_index; j < end_index; j++){
      for(auto it: tripletList_transpose[submatrix_index][j]){
        // int inner_col_index = it.col();
        // double ppr_value = it.value();
        int inner_col_index = it.first;
        double ppr_value = it.second;
        // if(ppr_value <= 0){
        //   continue;
        // }
        current_svd_mat_mapping.insertBackUncompressed(j, inner_col_index) = ppr_value;
      }
    }

    current_svd_mat_mapping.collapseDuplicates(internal::scalar_sum_op<double, double>());

    if(iter < dynamic_ppr_start_iter - 1){
      for(int j = start_index; j < end_index; j++){
        vector<pair<int, float>>().swap(tripletList[submatrix_index][j]);
        vector<pair<int, float>>().swap(tripletList_transpose[submatrix_index][j]);
      }
    }
  }



  for (int k_iter=0; k_iter<current_svd_mat_mapping.outerSize(); ++k_iter){
      for (SparseMatrix<double, RowMajor, int>::InnerIterator it(current_svd_mat_mapping, k_iter); it; ++it){
          // if(it.value() > reservemin){
              // it.valueRef() = log10(it.value()/reservemin);
              it.valueRef() = log10(1 + it.value()/reservemin);
              // it.valueRef() = log10(10 + it.value()/reservemin);
          // }
          // else{
          //     it.valueRef() = 0;
          // }
      }
  }



  double A_norm = current_svd_mat_mapping.norm();

  // cout<<"subset_tree->svd_mat_mapping[submatrix_index].rows() = "<<subset_tree->svd_mat_mapping[submatrix_index].rows()<<endl;
  // cout<<"subset_tree->svd_mat_mapping[submatrix_index].cols() = "<<subset_tree->svd_mat_mapping[submatrix_index].cols()<<endl;
  
  double Ei_norm;
  
  if(subset_tree->svd_mat_mapping[submatrix_index].rows() == 0 || subset_tree->svd_mat_mapping[submatrix_index].cols() == 0){
    // subset_tree->svd_mat_mapping[submatrix_index].resize(count_labeled_node, current_group_size);
    Ei_norm = A_norm;
  }
  else{
    Ei_norm = (current_svd_mat_mapping - subset_tree->svd_mat_mapping[submatrix_index]).norm();
  }
  



  delta = delta * sqrt(2);



  if( subset_tree->norm_B_Bid_difference_vec[submatrix_index] + Ei_norm < delta * A_norm){
    update_mat_tree_record[submatrix_index] = -1;
    current_svd_mat_mapping.resize(0, 0);
    current_svd_mat_mapping.data().squeeze();
  }
  else{
    update_mat_tree_record[submatrix_index] = iter;
    subset_tree->svd_mat_mapping[submatrix_index].resize(0, 0);
    subset_tree->svd_mat_mapping[submatrix_index].data().squeeze();

    subset_tree->svd_mat_mapping[submatrix_index] = current_svd_mat_mapping;
    current_svd_mat_mapping.resize(0, 0);
    current_svd_mat_mapping.data().squeeze();
    // cout<<"subset_tree->svd_mat_mapping[submatrix_index].rows() = "<<subset_tree->svd_mat_mapping[submatrix_index].rows()<<endl;
    // cout<<"subset_tree->svd_mat_mapping[submatrix_index].cols() = "<<subset_tree->svd_mat_mapping[submatrix_index].cols()<<endl;
  }





}































































void sparse_sub_svd_function_with_norm_computation_Robinhood(int d, int pass, 
int update_j, 
d_row_tree_mkl_sparse_dynamic* subset_tree,
int largest_level_start_index,
int current_out_iter,
int lazy_update_start_iter,
Eigen::SparseMatrix<double, RowMajor, int> &submatrix,
double reservemin
){



  mat* matrix_vec_t = subset_tree->hierarchy_matrix_vec[largest_level_start_index + update_j];


  SparseMatrix<double, RowMajor, int>& ppr_matrix_temp = submatrix;

  int submatrix_rows = submatrix.rows();
  int submatrix_cols = submatrix.cols();

  long nnz = ppr_matrix_temp.nonZeros();


  cout<<"sparse, nnz = "<<nnz<<endl;

  assert(nnz < INT_MAX);
  auto hash_coo_time = chrono::system_clock::now();


  mat_coo *ppr_matrix_coo = coo_matrix_new(submatrix_rows, submatrix_cols, nnz);
  ppr_matrix_coo->nnz = nnz;

  long nnz_iter=0;
  // double ppr_norm =0;
  
  for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
    for (SparseMatrix<double, RowMajor, int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){
      double value1 = it.value();
      // if(value1 < reservemin){

      // }
      // else{
        ppr_matrix_coo->rows[nnz_iter] = it.row() + 1;
        ppr_matrix_coo->cols[nnz_iter] = it.col() + 1;
        ppr_matrix_coo->values[nnz_iter] = value1;
        // ppr_matrix_coo->values[nnz_iter] = log10(1 + value1 / reservemin);
        // ppr_matrix_coo->values[nnz_iter] = log10(value1 / reservemin);
        nnz_iter ++;
      // }
    }
  }

  // ppr_matrix_temp.resize(0,0);
  // ppr_matrix_temp.data().squeeze();

  auto coo_csr_time = chrono::system_clock::now();
  auto elapsed_sparse_coo_time = chrono::duration_cast<std::chrono::seconds>(coo_csr_time - hash_coo_time);

  mat_csr* ppr_matrix = csr_matrix_new();
  csr_init_from_coo(ppr_matrix, ppr_matrix_coo);


  coo_matrix_delete(ppr_matrix_coo);
  ppr_matrix_coo = NULL;

  mat *U = matrix_new(submatrix_rows, d);
  mat *S = matrix_new(d, 1);
  mat *V = matrix_new(submatrix_cols, d);

  frPCA(ppr_matrix, &U, &S, &V, d, pass);

  csr_matrix_delete(ppr_matrix);

  ppr_matrix = NULL;

  mat * S_full = matrix_new(d, d);
  for(int i = 0; i < d; i++){
    matrix_set_element(S_full, i, i, matrix_get_element(S, i, 0));
  }


  matrix_matrix_mult(U, S_full, matrix_vec_t);

  if(current_out_iter >= lazy_update_start_iter - 1){
    auto norm_start_time = chrono::system_clock::now();

    // cout<<1<<endl;
    mat * V_transpose_matrix = matrix_new(d, submatrix_cols);

    // cout<<2<<endl;
    matrix_build_transpose(V_transpose_matrix, V);

    matrix_delete(V);
    V = NULL;

    mat * final_matrix_shape_for_frobenius = matrix_new(submatrix_rows, submatrix_cols);

    // cout<<4<<endl;
    matrix_matrix_mult(matrix_vec_t, V_transpose_matrix, final_matrix_shape_for_frobenius);


    // cout<<5<<endl;
    matrix_delete(V_transpose_matrix);

    // cout<<6<<endl;

    V_transpose_matrix = NULL;


    // cout<<7<<endl;
    for (int k=0; k<ppr_matrix_temp.outerSize(); ++k){
      for (SparseMatrix<double, RowMajor, int>::InnerIterator it(ppr_matrix_temp, k); it; ++it){
        double value1 = it.value();
        if(value1 == 0){

        }
        else{
          double XY_value = matrix_get_element(final_matrix_shape_for_frobenius, it.row(), it.col());
          matrix_set_element(final_matrix_shape_for_frobenius, it.row(), it.col(), XY_value - value1);
        }
      }
    }

    // cout<<8<<endl;


    subset_tree->norm_B_Bid_difference_vec[update_j] = get_matrix_frobenius_norm(final_matrix_shape_for_frobenius);

    // cout<<9<<endl;

    matrix_delete(final_matrix_shape_for_frobenius);

    // cout<<10<<endl;

    final_matrix_shape_for_frobenius = NULL;

    // cout<<11<<endl;

    auto norm_end_time = chrono::system_clock::now();
    auto elapsed_norm_time = chrono::duration_cast<std::chrono::seconds>(norm_end_time - norm_start_time);
  }






  matrix_delete(U);
  matrix_delete(S);
  matrix_delete(S_full);
  U = NULL;
  S = NULL;
  S_full = NULL;
  

}




























































void dense_sub_svd_function_Robinhood(int d, int pass, 
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
































































void mkl_right_matrix_multiplication_without_norm_Robinhood_with_link_prediction(
d_row_tree_mkl_sparse_dynamic* subset_tree,
mat* mkl_left_matrix, 
int vertex_number, 
vector<int> & line_update_mat_tree_record, 
int iter,
int count_labeled_node,
double reservemin,
int common_group_size,
int final_group_size,
int dynamic_ppr_start_iter,
vector<std::unordered_set<std::pair<int, int>, boost::hash< std::pair<int, int>>>>& pedge_set_nparts,
vector<std::unordered_set<std::pair<int, int>, boost::hash< std::pair<int, int>>>>& nedge_set_nparts,
vector<vector<std::pair<double, std::pair<int, int>>>>& nparts_embedding_score,
vector<int>& vertex_mapping,
Graph* g,
vector<int>& inner_group_mapping,
unordered_map<int, int>& row_index_mapping
)
{
    int unique_update_times = 0;
    
    auto total_right_matrix_start_time = chrono::system_clock::now();
    
    double total_norm_time = 0;

    for(int part = 0; part < subset_tree->nParts; part++){
      if(line_update_mat_tree_record[part] != iter){
        continue;
      }

      int current_group_size;
      if(part == subset_tree->nParts - 1){
        current_group_size = final_group_size;
      }
      else{
        current_group_size = common_group_size;
      }

      unique_update_times++;

      auto right_matrix_start_time = chrono::system_clock::now();

      // SparseMatrix<double, RowMajor, int>& ppr_matrix_temp = subset_tree->svd_mat_mapping[iter];

      long nnz = subset_tree->svd_mat_mapping[part].nonZeros();

      cout<<"right, nnz = "<<nnz<<endl;

      //3000, ppr_matrix_temp.cols() = 1025130
      mat_coo *ppr_matrix_coo = coo_matrix_new(subset_tree->row_dim, current_group_size, nnz);

      ppr_matrix_coo->nnz = nnz;

      long nnz_iter=0;

      for (int k=0; k<subset_tree->svd_mat_mapping[part].outerSize(); ++k){
          for (SparseMatrix<double, RowMajor, int>::InnerIterator it(subset_tree->svd_mat_mapping[part], k); it; ++it){
              double value1 = it.value();
              // if(it.value() < reservemin){

              // }
              // else{
                ppr_matrix_coo->rows[nnz_iter] = it.row() + 1;
                ppr_matrix_coo->cols[nnz_iter] = it.col() + 1;
                ppr_matrix_coo->values[nnz_iter] = value1;
                // ppr_matrix_coo->values[nnz_iter] = log10(1 + value1 / reservemin);
                // ppr_matrix_coo->values[nnz_iter] = log10(value1 / reservemin);
                nnz_iter ++;
              // }
          }
      }

      if(iter < dynamic_ppr_start_iter - 1){
        cout<<"right delete"<<endl;
        subset_tree->svd_mat_mapping[part].resize(0, 0);

        subset_tree->svd_mat_mapping[part].data().squeeze();
      }


      mat_csr* ppr_matrix = csr_matrix_new();

      csr_init_from_coo(ppr_matrix, ppr_matrix_coo);

      coo_matrix_delete(ppr_matrix_coo);

      ppr_matrix_coo = NULL;

      // 128
      mat *mkl_result_mat = matrix_new(current_group_size, mkl_left_matrix->ncols);


      csr_matrix_transpose_matrix_mult(ppr_matrix, mkl_left_matrix, mkl_result_mat);

      auto right_matrix_end_time = chrono::system_clock::now();
      auto elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(right_matrix_end_time - right_matrix_start_time);



      nparts_embedding_score[part].clear();

      unordered_set<std::pair<int, int>, boost::hash< std::pair<int, int>>>& pedge_set = pedge_set_nparts[part];
      for (auto it = pedge_set.begin(); it != pedge_set.end(); ++it) {
        int i = it->first;
        int row_i = row_index_mapping[i];
        int j = it->second;
        int inner_j = inner_group_mapping[j];

        double score;

        vec* U_row = vector_new(mkl_result_mat->ncols);
        matrix_get_row(mkl_left_matrix, row_i, U_row);
        vec* V_row = vector_new(mkl_result_mat->ncols);
        matrix_get_row(mkl_result_mat, inner_j, V_row);
        score = vector_dot_product(U_row, V_row);
        vector_delete(U_row);
        vector_delete(V_row);
        U_row = NULL;
        V_row = NULL;

        score *= (g->outdegree[i] * g->indegree[j]);

        nparts_embedding_score[part].push_back( make_pair(score, make_pair(i,j)) );
      }

      unordered_set<pair<int, int>, boost::hash< pair<int, int>>>& nedge_set = nedge_set_nparts[part];
      for (auto it = nedge_set.begin(); it != nedge_set.end(); ++it) {
        int i = it->first;
        int row_i = row_index_mapping[i];
        int j = it->second;
        int inner_j = inner_group_mapping[j];

        double score;

        vec* U_row = vector_new(mkl_result_mat->ncols);
        matrix_get_row(mkl_left_matrix, row_i, U_row);
        vec* V_row = vector_new(mkl_result_mat->ncols);
        matrix_get_row(mkl_result_mat, inner_j, V_row);
        score = vector_dot_product(U_row, V_row);
        vector_delete(U_row);
        vector_delete(V_row);
        U_row = NULL;
        V_row = NULL;

        score *= (g->outdegree[i] * g->indegree[j]);

        nparts_embedding_score[part].push_back(make_pair(score, make_pair(i,j)));
      }






      matrix_delete(mkl_result_mat);
      mkl_result_mat = NULL;

      csr_matrix_delete(ppr_matrix);
      ppr_matrix = NULL;


    }

    matrix_delete(mkl_left_matrix);
    mkl_left_matrix = NULL;

    tcout<<"right_unique_update_times = "<<unique_update_times<<endl;

    auto total_right_matrix_end_time = chrono::system_clock::now();
    auto total_elapsed_right_matrix_time = chrono::duration_cast<std::chrono::seconds>(total_right_matrix_end_time - total_right_matrix_start_time);
    tcout << "Total right matrix cost time: "<< total_elapsed_right_matrix_time.count() << endl;

    tcout << "Total norm cost time = "<< total_norm_time << endl;

}







