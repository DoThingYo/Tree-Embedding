import time

import sys

if __name__ == '__main__':
    queryname = sys.argv[1]
    vertex_number = int(sys.argv[2])

    year_dict = {}

    # # out.youtube-u-growth
    # # out.prosper-loans

    f = open("./DY_Dataset/mag-authors/mag_2019_edges.json", 'r')

    # Remove the first line
    line = f.readline()
    print(line)

    # Read following edges
    lines = f.readlines()
    print(lines[0])

    import re

    vertex_number_edge_repeat_check_tuplelist = []
    for i in range(vertex_number):
        vertex_number_edge_repeat_check_tuplelist.append([])


    print("len(lines) = ", len(lines))

    line_index = 0

    for line in lines:

        line_split_list = re.split(",", line)
        
        #Delete null string
        record_vec = []
        for i in range(len(line_split_list)):
            if line_split_list[i] == "":
                record_vec.append(i)

        for i in range(len(record_vec)):
            del line_split_list[ record_vec[i] - i ]


        line_split_list[0] = line_split_list[0][1:].strip()
        line_split_list[1] = line_split_list[1].strip()
        line_split_list[2] = line_split_list[2].strip()


        str_timestamp = line_split_list[2]

        year_int = int(str_timestamp[0:4])


        if int(line_split_list[1]) > vertex_number:
            print(int(line_split_list[1]))

        #check self-loop
        if int(line_split_list[0]) == int(line_split_list[1]):
            continue

        if int(line_split_list[1]) in vertex_number_edge_repeat_check_tuplelist[int(line_split_list[0])]:
            continue
        else:
            vertex_number_edge_repeat_check_tuplelist[int(line_split_list[0])].append( int(line_split_list[1]) )

        if year_int >= 1800 and year_int <= 1979:
            year_int = 18001979
        elif year_int > 1979 and year_int <= 1983:
            year_int = 19801983
        elif year_int > 1983 and year_int <= 1987:
            year_int = 19841987
        elif year_int > 1987 and year_int <= 1991:
            year_int = 19881991
        elif year_int > 1991 and year_int <= 1995:
            year_int = 19921995
        elif year_int > 1995 and year_int <= 1999:
            year_int = 19961999
        elif year_int > 1999 and year_int <= 2003:
            year_int = 20002003
        elif year_int > 2003 and year_int <= 2007:
            year_int = 20042007
        elif year_int > 2007 and year_int <= 2011:
            year_int = 20082011
        elif year_int > 2011 and year_int <= 2015:
            year_int = 20122015
        elif year_int > 2015 and year_int <= 2019:
            year_int = 20162019

        
        # Add this edge to its corresponding year
        if year_int not in year_dict.keys():
            year_dict[year_int] = []
        year_dict[year_int].append([line_split_list[0], line_split_list[1]])

        

        # # used for debug
        # line_index += 1
        # if line_index == 50:
        #     break

    f.close()    

    list_year_dict_keys = list(year_dict.keys())
    print("len(list_year_dict_keys) = ", len(list_year_dict_keys))


    list_year_dict_values = list(year_dict.values())
    print("len(list_year_dict_values) = ", len(list_year_dict_values))


    print("finish reading!")

    for i in year_dict.keys():
        with open("./DY_Dataset/" + queryname + "/" + str(i) + ".txt", "w") as f:
            #the data structure corresponding to each key is a list
            for j in range(len(year_dict[i])):
                from_node = year_dict[i][j][0]
                to_node = year_dict[i][j][1]
                f.write(str(from_node) + " " + str(to_node) + "\n")

    print("finish writing")






    with open("./DY_Dataset/mag-authors/config.txt", 'w') as f:
        max_dict_keys_list = sorted(year_dict.keys())
        config_str = "./DY_Dataset/Target/mag-authors.txt\n"

        for i in range(len(max_dict_keys_list)):
            config_str += "./DY_Dataset/mag-authors/" + str(max_dict_keys_list[i]) + ".txt" + "\n"
        
        f.write(config_str)


