#include <algorithm>
#include <iostream>
#include "Graph_dynamic.h"



int main(int argc,  char **argv){
  srand((unsigned)time(0));
  char *endptr;
  string queryname = argv[1];
  double percent = strtod(argv[2], &endptr);  // ratio of test set
  //The number of vertex
  int vertex_number = strtod(argv[3], &endptr);



  // "traindataset" is just a prefix, we will generate training data for each snapshots
  string traindataset = "DY_LP_Dataset/"+ queryname + "/" + queryname;
  
  // The "ptestdataset" will be the final POS testing edges, are obtained form each snapshot and accumulate all to verify(like the labels for NC task).
  string ptestdataset = "DY_LP_Dataset/" + queryname + "/" + queryname + "-Pos_LP_Test.txt";
 
  // The "ntestdataset" will be the final NEG testing edges, are obtained form each snapshot and accumulate all to verify(like the labels for NC task).
  string ntestdataset = "DY_LP_Dataset/" + queryname + "/" + queryname + "-Neg_LP_Test.txt";
 
 

  string config_path =  "DY_LP_Dataset/" + queryname + "/config_Alledges.txt";
  ifstream config_infile( config_path.c_str() );

  
  int snapshots_number = 0;

  vector<string> shots_address_vec;

  string s2;
  while(getline(config_infile, s2)) 
  { 
    
    shots_address_vec.push_back(s2);
    snapshots_number++;
  }
  
  UGraph* g = new UGraph();

  g->initializeDynamicUGraph(vertex_number);

  string subset_infilename = shots_address_vec[0];

  ifstream subset_infile(subset_infilename.c_str());
  int node;
  unordered_set<int> subset_nodes_set;
  while (subset_infile.good()) {
    subset_infile >> node;
    subset_nodes_set.insert(node);
  }

  vector<pair<int, int>> test_percent_subset_edge_vec;


  for(int iter = 1; iter < snapshots_number; iter++){
    clock_t t0 = clock();
    vector<pair<int, int>> edge_vec;
    g->inputDynamicGraph(shots_address_vec[iter].c_str(), edge_vec);

    clock_t t1 = clock();
    cout << "reading in graph takes " << (t1 - t0)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;
    
    clock_t t2 = clock();
    string snapshot_train_dataset = traindataset + "-Train-" + to_string(iter) + ".txt";
    cout<<snapshot_train_dataset<<endl;

    g->SubsetRandomSplitGraphWithTrainOutput(edge_vec, snapshot_train_dataset, percent);

    clock_t t3 = clock();
    cout << "splitting graph takes " << (t3 - t2)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;

    ofstream outfile_train(snapshot_train_dataset.c_str());

    int m2 = percent * edge_vec.size();

    for(int i = 0; i < m2; i++){
      int from = edge_vec[i].first;
      int to = edge_vec[i].second;
      std::unordered_set<int>::iterator find_iter_from = subset_nodes_set.find(from);
      std::unordered_set<int>::iterator find_iter_to = subset_nodes_set.find(to);
      
      if(find_iter_from != subset_nodes_set.end())
      {
        test_percent_subset_edge_vec.push_back(make_pair(from, to));
      }
      else if(find_iter_to != subset_nodes_set.end()){
        test_percent_subset_edge_vec.push_back(make_pair(to, from));
      }
      // // Should be discarded because we limit the training ratio to (1-percent)
      // else{
      //   outfile_train << from << " " << to << endl;
      // }
    }

    for (int i = m2; i < edge_vec.size(); i++) {
      outfile_train << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }


  }



  int pos_edges_number = test_percent_subset_edge_vec.size();
  int neg_edges_number = pos_edges_number;

  cout<<"!!!!!!!!"<<endl;
  cout<<"number of pos_edges = "<<pos_edges_number<<endl;
  cout<<"!!!!!!!!"<<endl;



  clock_t t4 = clock();

  g->SubsetNegativeSamples(ntestdataset, neg_edges_number, subset_nodes_set);
  clock_t t5 = clock();
  cout << "sampling negative edges takes " << (t5 - t4)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;

  ofstream out_pos_file(ptestdataset.c_str());
  for(int i = 0; i < pos_edges_number; i++){
    out_pos_file << test_percent_subset_edge_vec[i].first << " " << test_percent_subset_edge_vec[i].second<<endl;
  }

  clock_t t6 = clock();
  cout << "storing positive edges takes " << (t6 - t5)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;


  string output_config_path = "DY_LP_Dataset/"+ queryname + "/" + "config.txt";

  ofstream output_config_file(output_config_path.c_str());

  output_config_file<<subset_infilename<<endl;

  for(int iter = 1; iter < snapshots_number; iter++){
  
    string snapshot_train_dataset = traindataset + "-Train-" + to_string(iter) + ".txt";

    output_config_file<< snapshot_train_dataset <<endl;
  
  }

  cout<<"Finished!"<<endl;

}

