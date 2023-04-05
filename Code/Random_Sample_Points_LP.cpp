#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <deque>
#include <vector>
#include <unordered_map>
#include<list>
#include "Graph_dynamic.h"

#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>

#include <chrono>
#include <climits>

#include<assert.h>

#include<queue>


#include<assert.h>
#include<cmath>


#include<atomic>

#include <boost/functional/hash.hpp>

#include "my_queue.h"


using namespace std;





bool myfunction (int i,int j) { return (i<j); }




int main(int argc,  char **argv){
  auto start_time = std::chrono::system_clock::now();
  srand((unsigned)time(0));
  char *endptr;

  string queryname = argv[1];

  int count_labeled_nodes = strtod(argv[2], &endptr);
  // int count_labeled_nodes = 3000;

  int vertex_number = strtod(argv[3], &endptr);

  // // For LP task, we make sure no duplicate nodes are selected.
  vector<int> check_if_selected_vec(vertex_number, 0);



  string config_path =  "DY_LP_Dataset/" + queryname + "/config_Alledges.txt";
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






  unordered_set<int> first_snapshot_nodes_set;
  string first_snapshot_string = shots_address_vec[1];
  ifstream first_snapshot_file( first_snapshot_string.c_str() );

  int left_node = 0;
  int right_node = 0;

  while (first_snapshot_file.good()) {
    first_snapshot_file >> left_node >> right_node;

    first_snapshot_nodes_set.insert(left_node);
  }

  vector<int> first_snapshot_nodes_vec;

  for(unordered_set<int>::iterator it = first_snapshot_nodes_set.begin(); it != first_snapshot_nodes_set.end(); it++){    
    first_snapshot_nodes_vec.push_back(*it);
  }


  cout<<"First snapshots have "<<first_snapshot_nodes_vec.size()<<" number of different start nodes"<<endl;

  
  int start_node = 0;

  vector<int> subset_node_vec;
  


  while(subset_node_vec.size() != count_labeled_nodes){

    // Randomly sample a start node, check the sampled node must not be a dangling node or a duplicate node.
    while(1){
      // start_node = rand() % vertex_number;
      int sample_index = rand() % first_snapshot_nodes_vec.size();
      start_node = first_snapshot_nodes_vec[sample_index];

      // cout<<"Sampled node = "<<start_node<<endl;
      if( check_if_selected_vec[start_node] == 0 ){
        break;
      }
    }

    subset_node_vec.push_back(start_node);
    // cout<<"First push: "<<start_node<<endl;

    if(subset_node_vec.size() == count_labeled_nodes){
      break;
    }


    // We remove this label sign because we don't want duplicate nodes to be added again
    check_if_selected_vec[start_node] = 1;
    
  }






  std::sort(subset_node_vec.begin(), subset_node_vec.end(), myfunction);


  string output_subset_target_nodes = "DY_LP_Dataset/Target/" + queryname + ".txt";;

  // write to file
  ofstream output_subset_target_nodes_file(output_subset_target_nodes.c_str());
  for(int i = 0; i < subset_node_vec.size(); i++){
    output_subset_target_nodes_file << subset_node_vec[i] <<endl;
  }




}



