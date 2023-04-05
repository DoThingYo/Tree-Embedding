#ifndef GRAPH_DYNAMIC_H
#define GRAPH_DYNAMIC_H

#define SetBit(A, k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A, k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestBit(A, k)    ( A[(k/32)] & (1 << (k%32)) )

#define DSetBit(A, k, j, n)     ( A[(k*(n/32)+(j/32))] |= (1 << (((k%32)*(n%32))%32+j%32)%32) )
#define DClearBit(A, k, j, n)   ( A[(k*(n/32)+(j/32))] &= ~(1 << (((k%32)*(n%32))%32+j%32)%32) )
#define DTestBit(A, k, j, n)    ( A[(k*(n/32)+(j/32))] & (1 << (((k%32)*(n%32))%32+j%32)%32) )


#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <unordered_set>
#include<string.h>

#include<assert.h>



using namespace std;


class BitMatrix {
public:
  long long int n;
  long long int narray;
  int *bitMatrix;

  BitMatrix() {
  }

  ~BitMatrix() {
  }

  void ConBitMatrix(int nodenumber) {
    n = nodenumber;
    narray = n * (n / 32) + 1;
    bitMatrix = new int[narray];
    for (int i = 0; i < narray; i++) {
      bitMatrix[i] = 0;
    }
  }

  void Update(int i, int j) {
    if (!DTestBit(bitMatrix, i, j, n)) {
      DSetBit(bitMatrix, i, j, n);
    }
  }

  int Find(int i, int j) {
    return DTestBit(bitMatrix, i, j, n);
  }
};


class Graph {
public:
  int n;    //number of nodes
  int m;    //number of edges
  int **inAdjList;
  int **outAdjList;
  int *indegree;
  int *outdegree;


  int *former_indegree;
  int *former_outdegree;
  int** BiggerInAdjList;
  int** BiggerOutAdjList;

  int *pointer_out;
  int *pointer_in;




  Graph() {
  }

  ~Graph() {
  }


  void initializeDirectedDynamicGraph(int vertex_number){
    m = 0;
    n = vertex_number;
    cout << "n=" << n << endl;
    indegree = new int[n];
    outdegree = new int[n];
    for (int i = 0; i < n; i++) {
      indegree[i] = 0;
      outdegree[i] = 0;
    }
    inAdjList = new int *[n];
    BiggerInAdjList = new int *[n];
    
    outAdjList = new int *[n];
    BiggerOutAdjList = new int *[n];    
    
    pointer_in = new int[n];
    pointer_out = new int[n];
    for (int i = 0; i < n; i++) {
      pointer_in[i] = 0;
      pointer_out[i] = 0;
    }


    former_indegree = new int[n];
    former_outdegree = new int[n];
  }


  void inputDirectedDynamicGraph(string filename, vector<pair<int, int>> &edge_vec) {
    cout << filename << endl;
    clock_t t1 = clock();
    ifstream infile(filename.c_str());


    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
    }
    
    for (int i = 0; i < n; i++) {
      former_indegree[i] = indegree[i];
      former_outdegree[i] = outdegree[i];
    }


    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;

      outdegree[from]++;
      indegree[to]++;
    }

    cout << "..." << endl;
    for (int i = 0; i < n; i++) {
      //inAdjList
      if(indegree[i] == 0){

        if(inAdjList[i] == NULL){
          inAdjList[i] = new int[indegree[i]];
        }
      }
      else{
        if(inAdjList[i] == NULL){

          inAdjList[i] = new int[indegree[i]];

          }
        else{
            if(indegree[i] == former_indegree[i]){
            }
            else{
              assert(indegree[i] > former_indegree[i]); 

              BiggerInAdjList[i] = new int[indegree[i]];

              memcpy(BiggerInAdjList[i], inAdjList[i], sizeof(int) * former_indegree[i]);

              delete inAdjList[i];
              inAdjList[i] = NULL;

              inAdjList[i] = BiggerInAdjList[i];

            }
          }
      }

      //outAdjList
      if(outdegree[i] == 0){

        if(outAdjList[i] == NULL){
          outAdjList[i] = new int[outdegree[i]];
        }
      }
      else{
        if(outAdjList[i] == NULL){

          outAdjList[i] = new int[outdegree[i]];

        }
        else{
            if(outdegree[i] == former_outdegree[i]){
              continue;
            }
            else{

              BiggerOutAdjList[i] = new int[outdegree[i]];

              assert(outdegree[i] > former_outdegree[i]);

              memcpy(BiggerOutAdjList[i], outAdjList[i], sizeof(int) * former_outdegree[i]);

              delete outAdjList[i];
              outAdjList[i] = NULL;

              outAdjList[i] = BiggerOutAdjList[i];

            }
        }
      }
    }
    
    cout<<"Finish enlarging In and Out AdjList!"<<endl;



    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;

      to = it->second;

      outAdjList[from][pointer_out[from]] = to;

      pointer_out[from]++;

      inAdjList[to][pointer_in[to]] = from;

      pointer_in[to]++;
      m++;
    }



    clock_t t2 = clock();
    cout << "m=" << m << endl;
  }







  void SubsetRandomSplitGraphWithTrainOutput(vector<pair<int, int>>& edge_vec, string snapshot_outFilename_train, double percent) {
    clock_t t1 = clock();

    // snapshot_edge_number
    int m1 = edge_vec.size();


    ofstream outfile_train(snapshot_outFilename_train.c_str());



    int m2 = percent * m1;
    for (int i = 0; i < m2; i++) {
      int r = rand() % (m1 - i);
      iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
    }

    
  }






  void SubsetNegativeSamples(string outFilename, int total_sample_number, unordered_set<int> &subset_nodes_set) {

    ofstream outfile(outFilename.c_str());

    int npair = 0;
    int nsample = 0;

    int subset_number = subset_nodes_set.size();

    vector<int> subset_nodes_vec;

    for(unordered_set<int>::iterator it = subset_nodes_set.begin(); it != subset_nodes_set.end(); it++){    
      subset_nodes_vec.push_back(*it);
    }

    vector<unordered_set<int>> adjM(subset_number);

    for (int i = 0; i < subset_number; i++) {
      int original_node_id = subset_nodes_vec[i];
      for (int j = 0; j < outdegree[original_node_id]; j++) {
        adjM[i].insert(outAdjList[original_node_id][j]);
      }
    }

    while (nsample < total_sample_number) {
      // int i = rand() % n;
      int i = rand() % subset_number;
      int original_node_id = subset_nodes_vec[i];

      int j = rand() % n;
      npair++;
      if (adjM[i].find(j) == adjM[i].end()) {
        outfile << original_node_id << " " << j << endl;
        nsample++;
      }
    }

    cout << "npair: " << npair << " nsample: " << nsample << endl;
  }


























  void inputGraph(string filename) {
    cout << filename << endl;
    clock_t t1 = clock();
    m = 0;
    ifstream infile(filename.c_str());
    infile >> n;
    cout << "n=" << n << endl;
    indegree = new int[n];
    outdegree = new int[n];
    for (int i = 0; i < n; i++) {
      indegree[i] = 0;
      outdegree[i] = 0;
    }

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
    }
    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      outdegree[from]++;
      indegree[to]++;
    }
    cout << "..." << endl;
    inAdjList = new int *[n];
    outAdjList = new int *[n];
    for (int i = 0; i < n; i++) {
      inAdjList[i] = new int[indegree[i]];
      outAdjList[i] = new int[outdegree[i]];
    }
    int *pointer_in = new int[n];
    int *pointer_out = new int[n];
    for (int i = 0; i < n; i++) {
      pointer_in[i] = 0;
      pointer_out[i] = 0;
    }


    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      outAdjList[from][pointer_out[from]] = to;
      pointer_out[from]++;
      inAdjList[to][pointer_in[to]] = from;
      pointer_in[to]++;
      m++;
    }

    for (int i = 0; i < n; i++) {
      sort(outAdjList[i], outAdjList[i] + outdegree[i]);
    }

    delete[] pointer_in;
    delete[] pointer_out;
    pointer_in = NULL;
    pointer_out = NULL;

    clock_t t2 = clock();
    cout << "m=" << m << endl;
  }

  
  void RandomSplitGraph(string inFilename, string outFilename1, string outFilename2, double percent) {
    clock_t t1 = clock();
    int m1 = 0;
    ifstream infile(inFilename.c_str());
    ofstream outfile1(outFilename1.c_str());
    ofstream outfile2(outFilename2.c_str());
    int n1;
    infile >> n1;
    outfile1 << n1 << endl;
    outfile2 << n1 << endl;

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
      m1++;
    }
    int m2 = percent * m1;
    for (int i = 0; i < m2; i++) {
      int r = rand() % (m1 - i);
      iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
      outfile1 << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }
    for (int i = m2; i < edge_vec.size(); i++) {
      outfile2 << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }
  }

  void NegativeSamples(string outFilename, double percent) {
    int sample_number = 0;
    int total_sample_number = percent * m;
    ofstream outfile(outFilename.c_str());
    vector<unordered_set<int>> adjM(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < outdegree[i]; j++) {
        adjM[i].insert(outAdjList[i][j]);
      }
    }
    outfile << n << endl;
    int npair = 0;
    int nsample = 0;
    while (sample_number < total_sample_number) {
      int i = rand() % n;
      int j = rand() % n;
      npair++;
      if (adjM[i].find(j) == adjM[i].end()) {
        outfile << i << " " << j << endl;
        sample_number++;
        nsample++;
      }
    }
    cout << "npair: " << npair << " nsample: " << nsample << endl;
  }

  int getInSize(int vert) {
    return indegree[vert];
  }

  int getInVert(int vert, int pos) {
    return inAdjList[vert][pos];
  }

  int getOutSize(int vert) {
    return outdegree[vert];
  }

  int getOutVert(int vert, int pos) {
    return outAdjList[vert][pos];
  }

};


















class UGraph {
public:
  int n;    //number of nodes
  int m;    //number of edges
  int **AdjList;
  int *former_degree;
  int *degree;
  int** BiggerAdjList;
  
  int *pointer_out;

  UGraph() {
  }

  ~UGraph() {
  }


  void initializeDynamicUGraph(int vertex_number){
    m = 0;
    n = vertex_number;
    cout << "n=" << n << endl;
    degree = new int[n];
    for (int i = 0; i < n; i++) {
      degree[i] = 0;
    }
    AdjList = new int *[n];
    BiggerAdjList = new int *[n];
    pointer_out = new int[n];
    for (int i = 0; i < n; i++) {
      pointer_out[i] = 0;
    }
    former_degree = new int[n];
  }

  void inputDynamicGraph(string filename, vector<pair<int, int>> &edge_vec) {
    cout << filename << endl;
    clock_t t1 = clock();
    ifstream infile(filename.c_str());

    //read graph and get degree info
    // vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
    }

    for (int i = 0; i < n; i++) {
      former_degree[i] = degree[i];
    }


    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      degree[from]++;
      degree[to]++;
    }

    cout << "..." << endl;
    for (int i = 0; i < n; i++) {
      if(degree[i] == 0){

          AdjList[i] = new int[degree[i]];
      }
      else{
        if(AdjList[i] == NULL){

          AdjList[i] = new int[degree[i]];

        }
        else if(degree[i] == former_degree[i]){
        }
        else{
          BiggerAdjList[i] = new int[degree[i]];

          assert(degree[i] > former_degree[i]);

          memcpy(BiggerAdjList[i], AdjList[i], sizeof(int) * former_degree[i]);

          delete AdjList[i];
          AdjList[i] = NULL;

          AdjList[i] = BiggerAdjList[i];

          }
      }
    }
    
    cout<<"Finish enlarging AdjList!"<<endl;

    cout<<"edge_vec.size() = "<<edge_vec.size()<<endl;


    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      
      AdjList[from][pointer_out[from]] = to;
      pointer_out[from]++;
      AdjList[to][pointer_out[to]] = from;
      pointer_out[to]++;
      m++;
    }


    clock_t t2 = clock();
    cout << "m=" << m << endl;
  }
  








  void RandomSplitGraph(string inFilename, string outFilename1, string outFilename2, double percent) {
    clock_t t1 = clock();
    int m1 = 0;
    ifstream infile(inFilename.c_str());
    ofstream outfile1(outFilename1.c_str());
    ofstream outfile2(outFilename2.c_str());
    int n1;
    infile >> n1;
    outfile1 << n1 << endl;
    outfile2 << n1 << endl;

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
      m1++;
    }
    int m2 = percent * m1;
    for (int i = 0; i < m2; i++) {
      int r = rand() % (m1 - i);
      iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
      outfile1 << edge_vec[i].first << " " << edge_vec[i].second << endl;
      //ptestdataset
    }

    for (int i = m2; i < edge_vec.size(); i++) {
      outfile2 << edge_vec[i].first << " " << edge_vec[i].second << endl;
      //traindataset
    }
  }

















  void SubsetRandomSplitGraphWithTrainOutput(vector<pair<int, int>>& edge_vec, string snapshot_outFilename_train, double percent) {
    clock_t t1 = clock();

    int m1 = edge_vec.size();

    ofstream outfile_train(snapshot_outFilename_train.c_str());



    int m2 = percent * m1;
    for (int i = 0; i < m2; i++) {
      int r = rand() % (m1 - i);
      iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
    }
    for (int i = m2; i < edge_vec.size(); i++) {
      outfile_train << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }




  }


  void SubsetNegativeSamples(string outFilename, int total_sample_number, unordered_set<int> &subset_nodes_set) {

    cout<<"SubsetNegativeSamples: n = "<<n<<endl;

    ofstream outfile(outFilename.c_str());
    vector<unordered_set<int>> adjM(n);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < degree[i]; j++) {
        adjM[i].insert(AdjList[i][j]);
      }
    }
    
    int npair = 0;
    int nsample = 0;

    int subset_number = subset_nodes_set.size();

    vector<int> subset_nodes_vec;

    for(unordered_set<int>::iterator it = subset_nodes_set.begin(); it != subset_nodes_set.end(); it++){    
      subset_nodes_vec.push_back(*it);
    }

    while (nsample < total_sample_number) {

      int i = rand() % subset_number;
      i = subset_nodes_vec[i];

      int j = rand() % n;
      npair++;
      if (adjM[i].find(j) == adjM[i].end()) {
        outfile << i << " " << j << endl;
        nsample++;
      }
    }

    cout << "npair: " << npair << " nsample: " << nsample << endl;
  }







  void SplitDynamicGraph(string inFilename, string outFilename1, int snapshots_number) {
    clock_t t1 = clock();
    int m1 = 0;
    ifstream infile(inFilename.c_str());
    int n1;
    infile >> n1;
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
      m1++;
    }

    int start;
    int end;

    int common_group_size;

    if(m1 / snapshots_number * snapshots_number == m1){
      common_group_size = m1 / snapshots_number;
    }
    else{
      common_group_size  = m1 / snapshots_number + 1;
    }

    int final_group_size = m1 - (snapshots_number - 1) * common_group_size;

    for(int i = 0; i < snapshots_number; i++){
      ofstream outfile( (outFilename1 +"_" + to_string(i) + ".txt").c_str() );


      if(i != snapshots_number - 1){
        start = i * common_group_size;
        end = start + common_group_size;
      
        for (int i = start; i < end; i++) {
          int r = rand() % (m1 - i);
          iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
          outfile << edge_vec[i].first << " " << edge_vec[i].second << endl;
          //ptestdataset
        }
      }
      else{
        start = i * common_group_size;
        end = start + final_group_size;
        for (int i = start; i < end; i++) {
          outfile << edge_vec[i].first << " " << edge_vec[i].second << endl;
        }
      }
    }
  }



  void NegativeSamples(string outFilename, double percent) {
    int sample_number = 0;
    int total_sample_number = percent * m;
    ofstream outfile(outFilename.c_str());
    vector<unordered_set<int>> adjM(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < degree[i]; j++) {
        adjM[i].insert(AdjList[i][j]);
      }
    }
    outfile << n << endl;
    int npair = 0;
    int nsample = 0;
    while (sample_number < total_sample_number) {
      int i = rand() % n;
      int j = rand() % n;
      npair++;
      if (adjM[i].find(j) == adjM[i].end() && adjM[j].find(i) == adjM[j].end()) {
        outfile << i << " " << j << endl;
        sample_number++;
        nsample++;
      }
    }
    cout << "npair: " << npair << " nsample: " << nsample << endl;
  }

  int getSize(int vert) {
    return degree[vert];
  }

  int getVert(int vert, int pos) {
    return AdjList[vert][pos];
  }


  void inputGraph(string filename) {
    cout << filename << endl;
    clock_t t1 = clock();
    m = 0;
    ifstream infile(filename.c_str());
    infile >> n;
    cout << "n=" << n << endl;
    degree = new int[n];
    for (int i = 0; i < n; i++) {
      degree[i] = 0;
    }

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
    }
    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      degree[from]++;
      degree[to]++;
    }



    cout << "..." << endl;
    AdjList = new int *[n];
    for (int i = 0; i < n; i++) {
      AdjList[i] = new int[degree[i]];
    }
    int *pointer_out = new int[n];
    for (int i = 0; i < n; i++) {
      pointer_out[i] = 0;
    }

    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      AdjList[from][pointer_out[from]] = to;
      pointer_out[from]++;
      AdjList[to][pointer_out[to]] = from;
      pointer_out[to]++;
      m++;
    }


    delete[] pointer_out;
    pointer_out = NULL;

    clock_t t2 = clock();
    cout << "m=" << m << endl;
  }



};



#endif

