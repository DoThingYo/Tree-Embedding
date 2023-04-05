import time


import sys

if __name__ == '__main__':
    queryname = sys.argv[1]
    vertex_number = int(sys.argv[2])

    count_lines = 0

    month_dict = {}

    # out.youtube-u-growth

    f = open("./DY_LP_Dataset/" + queryname + "/" + "out." + queryname)

    lines = f.readlines()

    import re

    vertex_number_edge_repeat_check_tuplelist = []
    for i in range(vertex_number):
        vertex_number_edge_repeat_check_tuplelist.append([])

    for line in lines:

        line_split_list = re.split("\s", line)
        record_vec = []
        for i in range(len(line_split_list)):
            if line_split_list[i] == "":
                record_vec.append(i)

        for i in range(len(record_vec)):
            del line_split_list[ record_vec[i] - i ]


        if (len(line_split_list) < 4) or (not line_split_list[0].strip().isdigit()) or (not line_split_list[1].strip().isdigit()) or \
                (not line_split_list[2].strip().isdigit()) or (not line_split_list[3].strip().isdigit()):
            continue
        

        unit_timestamp = line_split_list[3]

        now = int(round(int(unit_timestamp)*1000))
        now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))


        # month_int = now02[0:7]

        month_int = now02[0:4] + now02[5:7]
        
        month_int = int(month_int)

        # print("now02 = ")
        # print(now02)

        # print("month_int = ")
        # print(month_int)

        # count_lines+=1

        # # used for debug
        # if count_lines == 50:
        #     break


        if int(line_split_list[1]) in vertex_number_edge_repeat_check_tuplelist[int(line_split_list[0])]:
            continue
        else:
            vertex_number_edge_repeat_check_tuplelist[int(line_split_list[0])].append( int(line_split_list[1]) )
        

        if month_int not in month_dict.keys():
            month_dict[month_int] = []
        month_dict[month_int].append([line_split_list[0], line_split_list[1]])

    f.close()    

    list_month_dict_keys = list(month_dict.keys())
    print("len(list_month_dict_keys) = ", len(list_month_dict_keys))


    list_month_dict_values = list(month_dict.values())
    print("len(list_month_dict_values) = ", len(list_month_dict_values))


    print("finish reading!")

    for i in month_dict.keys():
        with open("./DY_LP_Dataset/" + queryname + "/" + str(i) + ".txt", "w") as f:
            for j in range(len(month_dict[i])):
                from_node = month_dict[i][j][0]
                to_node = month_dict[i][j][1]
                f.write(str(from_node) + " " + str(to_node) + "\n")

    print("finish writing")



    with open("./DY_LP_Dataset/" + queryname + "/config_Alledges.txt", 'w') as f:
        max_dict_keys_list = sorted(month_dict.keys())

        # The first line of the config_Alledges file
        config_str = "./DY_LP_Dataset/Target/" + queryname + ".txt\n"

        for i in range(len(max_dict_keys_list)):
            config_str += "./DY_LP_Dataset/" + queryname + "/" + str(max_dict_keys_list[i]) + ".txt" + "\n"
        
        f.write(config_str)




