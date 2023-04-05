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

// using namespace Eigen;

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

  // For NC task, We have to make sure the selected subset node at least has one label, but not a null node. We also use this variable to deduplicate.
  vector<int> check_if_label_exist_vec(vertex_number, 0);

  // This should be the path containing all nodes with labels
  string LabelPath = "LABEL/" + queryname + ".txt";

  ifstream infile2(LabelPath.c_str());

  unordered_map<int, string> take_out_multi_labels;

  string s1;
  while(getline(infile2, s1)) 
  {
    std::size_t found_position = s1.find(" ");
    //std::string::npos: the constant is the largest representable value of type size_type. It is assuredly larger than max_size(); hence it serves as either a very large value or as a special code.
    assert( found_position!=std::string::npos );
    string num_str = s1.substr(0, found_position); 
    int node_number;
    node_number = atoi(num_str.c_str());
    check_if_label_exist_vec[node_number] = 1;
    take_out_multi_labels[node_number] = s1;
  }

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









  unordered_set<int> first_snapshot_nodes_set;
  string first_snapshot_string = shots_address_vec[1];
  ifstream first_snapshot_file( first_snapshot_string.c_str() );

  int left_node = 0;
  int right_node = 0;

  while (first_snapshot_file.good()) {
    first_snapshot_file >> left_node >> right_node;
    // cout<<"left node = "<<left_node<<endl;
    first_snapshot_nodes_set.insert(left_node);
  }

  vector<int> first_snapshot_nodes_vec;

  for(unordered_set<int>::iterator it = first_snapshot_nodes_set.begin(); it != first_snapshot_nodes_set.end(); it++){    
    first_snapshot_nodes_vec.push_back(*it);
  }


  cout<<"First snapshots have "<<first_snapshot_nodes_vec.size()<<" number of different start nodes"<<endl;

  
  int start_node = 0;

  vector<int> subset_node_vec;
  
  int count_components = 0;



  while(subset_node_vec.size() != count_labeled_nodes){

    // Randomly sample a start node, check the sampled node must not be a dangling node or a duplicate node.
    while(1){
      // start_node = rand() % vertex_number;
      int sample_index = rand() % first_snapshot_nodes_vec.size();
      start_node = first_snapshot_nodes_vec[sample_index];

      // cout<<"Sampled node = "<<start_node<<endl;
      if( check_if_label_exist_vec[start_node] == 1 ){
        break;
      }
    }

    subset_node_vec.push_back(start_node);
    cout<<"First push: "<<start_node<<endl;

    if(subset_node_vec.size() == count_labeled_nodes){
      break;
    }


    // We remove this label sign because we don't want duplicate nodes to be added again
    check_if_label_exist_vec[start_node] = 0;
    
  }






  std::sort(subset_node_vec.begin(), subset_node_vec.end(), myfunction);


  string output_subset_target_nodes = "DY_Dataset/Target/" + queryname + ".txt";;

  // write to file
  ofstream output_subset_target_nodes_file(output_subset_target_nodes.c_str());
  for(int i = 0; i < subset_node_vec.size(); i++){
    output_subset_target_nodes_file << subset_node_vec[i] <<endl;
  }



  cout<<"Output subset labels"<<endl;
  string output_subset_target_labels = "LABEL/" + queryname + "_subset.txt";;

  // write to file
  ofstream output_subset_target_labels_file(output_subset_target_labels.c_str());
  for(int i = 0; i < subset_node_vec.size(); i++){
    int current_number = subset_node_vec[i];
    string current_string = take_out_multi_labels[current_number];
    output_subset_target_labels_file <<current_string<<endl;
  }


}



