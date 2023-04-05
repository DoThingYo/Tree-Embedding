#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <deque>
#include <vector>
#include<fstream>
#include <unordered_map>
using namespace std;










int main(int argc,  char **argv){

  char *endptr;

  // The original queryname
  string queryname = argv[1];

  string batch_queryname = queryname + "_batch";

  //The round to be split
  int round_index = strtod(argv[2], &endptr);

  int snapshots_number = 0;

  vector<string> shots_address_vec;

  // This is the original config, so we use "queryname" here
  string config_path =  "DY_Dataset/" + queryname + "/config.txt";

  ifstream infile3( config_path.c_str() );

  string s2;
  while(getline(infile3, s2)) 
  { 
    shots_address_vec.push_back(s2);
    snapshots_number++;
  }

  cout<<"snapshots_number = "<<snapshots_number<<endl;


  //read graph and get degree info
  vector<pair<int, int>> edge_vec;
  int from;
  int to;
  int count_edges = 0;

  int NUMBER_IN_A_GROUP = 10000;

  int batch_update_snapshots_number = 100;


  ofstream out_config_file(("./DY_Dataset/" + batch_queryname + "/config.txt").c_str());

  //Output the first Round_index - 1 paths, because they are the whole edges and target file paths.
  for(int i = 0; i < round_index; i++){
    out_config_file<<shots_address_vec[i]<<endl;
  }

  int increment_round_index = round_index;

  //make sure we have read enough edges, unless there are not enough consecutive edges
  while(increment_round_index < snapshots_number && count_edges != NUMBER_IN_A_GROUP * batch_update_snapshots_number){

    string filename_with_path = shots_address_vec[increment_round_index];

    cout << filename_with_path << endl;

    ifstream infile(filename_with_path.c_str());

    while (infile.good()) {

        infile >> from >> to;
        edge_vec.push_back(make_pair(from, to));
        count_edges++;
        if(count_edges % NUMBER_IN_A_GROUP == 0){
            int num = count_edges / NUMBER_IN_A_GROUP;
            cout<<"num = "<<num<<endl;
            // cout<<("./DY_Dataset/" + queryname + "/" + queryname + "-Train-" + to_string(Round) 
            //     + "-" + to_string(num) + "-10000" + ".txt")<<endl;
            
            string current_edge_file_name = "./DY_Dataset/" + batch_queryname + "/" + batch_queryname + "-Train-" + to_string(round_index) 
                + "-" + to_string(num) + ".txt";
            
            cout<<current_edge_file_name<<endl;

            out_config_file<<current_edge_file_name<<endl;

            ofstream current_edge_file_stream( current_edge_file_name.c_str() );

            for(int i = count_edges - NUMBER_IN_A_GROUP; i < count_edges; i++){
                current_edge_file_stream << edge_vec[i].first << " " << edge_vec[i].second << endl;            
            }
        }

        if(count_edges == NUMBER_IN_A_GROUP * batch_update_snapshots_number){
            break;
        }

    }

    increment_round_index++;

    //There are two cases the programs are here. First: we have enough edges. Second: we have read all edges in this file but it's not enough
    if(count_edges == NUMBER_IN_A_GROUP * batch_update_snapshots_number){
        break;
    }

  }


  if(count_edges != NUMBER_IN_A_GROUP * batch_update_snapshots_number){
    cout<<"We have read all consecutive edges, but there are not enough edges till the end!"<<endl;
  }


  cout<<"We have used "<<increment_round_index - round_index<<" snapshots to generate batch update edges!"<<endl;



}


