import time

import sys

if __name__ == '__main__':
    queryname = sys.argv[1]
    vertex_number = int(sys.argv[2])

    year_dict = {}

    
    f = open("./DY_Dataset/patent/patent_edges.json", 'r')

    lines = f.readlines()

    import re

    vertex_number_edge_repeat_check_tuplelist = []
    for i in range(vertex_number):
        vertex_number_edge_repeat_check_tuplelist.append([])
    

    print("len(lines) = ", len(lines))

    count_lines = 0

    for line in lines:
        
        line_split_list = re.split(",", line)
        
        # This record_vec is to delete the empty ones.
        record_vec = []
        for i in range(len(line_split_list)):
            if line_split_list[i] == "":
                record_vec.append(i)
        
        # For i'th element, we need to reindex, because we have delete "i" elements for "i" index.
        for i in range(len(record_vec)):
            del line_split_list[ record_vec[i] - i ]
        
        # Extract the start node of an edge, the string often starts with "[start_node,".
        line_split_list[0] = line_split_list[0][1:].strip()
        # Get the end node of an edge and remove the blank space.
        line_split_list[1] = line_split_list[1].strip()
        # The timestamp to be processed.
        line_split_list[2] = line_split_list[2].strip()


        str_timestamp = line_split_list[2]
        
        # The first 8 elements should be year-month-day
        # year_int = str_timestamp[0:8]
        # The first 4 elements should be the year
        year_int = str_timestamp[0:4]

        # # Used for debug
        # count_lines += 1
        # if count_lines == 50:
        #     break
        

        if int(line_split_list[0]) > vertex_number:
            print("The start node exceeds the vertex number!")
            print(int(line_split_list[0]))

        if int(line_split_list[1]) > vertex_number:
            print("The end node exceeds the vertex number!")
            print(int(line_split_list[1]))

        if int(line_split_list[1]) in vertex_number_edge_repeat_check_tuplelist[int(line_split_list[0])]:
            print("Duplicate edges! Not saved again.")
            print(line_split_list[0], line_split_list[1])
            continue
        else:
            vertex_number_edge_repeat_check_tuplelist[int(line_split_list[0])].append( int(line_split_list[1]) )


        if year_int not in year_dict.keys():
            # Initialize this year
            year_dict[year_int] = []
        year_dict[year_int].append([line_split_list[0], line_split_list[1]])

    f.close()    

    list_year_dict_keys = list(year_dict.keys())
    print("len(list_year_dict_keys) = ", len(list_year_dict_keys))


    list_year_dict_values = list(year_dict.values())
    print("len(list_year_dict_values) = ", len(list_year_dict_values))


    print("finish reading!")

    for i in year_dict.keys():
        with open("./DY_Dataset/" + queryname + "/" + i + ".txt", "w") as f:
            for j in range(len(year_dict[i])):
                from_node = year_dict[i][j][0]
                to_node = year_dict[i][j][1]
                f.write(str(from_node) + " " + str(to_node) + "\n")

    print("finish writing")




    with open("./DY_Dataset/patent/config.txt", 'w') as f:
        max_dict_keys_list = sorted(year_dict.keys())
        config_str = "./DY_Dataset/Target/patent.txt\n"

        for i in range(len(max_dict_keys_list)):
            config_str += "./DY_Dataset/patent/" + str(max_dict_keys_list[i]) + ".txt" + "\n"
        
        f.write(config_str)
