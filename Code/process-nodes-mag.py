import json
from os import write

file = open('./DY_Dataset/mag-authors/mag_2019_nodes.json', 'r', encoding = 'utf-8')

lines = []

count_lines = 0
for line in file.readlines():
    dic = json.loads(line)
    lines.append(dic)
    count_lines+=1

    # # used for debug
    # if count_lines == 50:
    #     break


# with open("./DY_Dataset/mag-authors-u/mag-authors-nodes-labels-debug.txt", 'w') as f:
with open("./LABEL/mag-authors.txt", 'w') as f:
    for i in range(len(lines)):
        current_line = lines[i]

        #The fouth element in this list are list of labels
        number_of_labels = len(current_line[3])
        
        write_str = ""

        write_str = write_str + str(i) + " "

        # This dict is used to count each label's frequency
        max_dict = {}

        for j in range(number_of_labels):            
            #current_line[3][j][1] is the label
            current_label = int(current_line[3][j][1])
            if current_label not in max_dict.keys():
                max_dict[current_label] = 1
            else:
                max_dict[current_label] += 1

        # Sort the max_dict based on the frequency of the labels
        # max_dict_list = sorted(max_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

        # This should be a list, every element in this list is a tuple with key and value, the most frequent labels will be sorted and put in the beginning.
        max_dict_list = sorted(max_dict.items(), key = lambda kv:kv[1], reverse = True)
        
        count_change = 0
        for k in range(len(max_dict_list)):
            if max_dict_list[k][0] > 100:
                # Note that max_dict_list[k][0] is the label, max_dict_list[k][1] is the count(frequency).
                print("Wrong labels. The label should be within 100.")
                continue
            if k != len(max_dict_list) - 1:
                #max_dict_list[k][0] is the labels, max_dict_list[k][1] is the frequency
                write_str = write_str + str(max_dict_list[k][0]) + " "
                count_change += 1
            else:
                write_str = write_str + str(max_dict_list[k][0]) + "\n"
                count_change += 1
        
        # Have at least one label
        if count_change != 0:
            f.write(write_str)

