import json
from os import write

file = open('./DY_Dataset/patent/patent_nodes.json', 'r', encoding = 'utf-8')

lines = []

count_lines = 0
for line in file.readlines():
    dic = json.loads(line)
    lines.append(dic)
    count_lines+=1

    # # used for debug
    # if count_lines == 50:
    #     break


with open("./LABEL/patent.txt", 'w') as f:
    main_dict = {}


    count_index = 0
    

    for i in range(len(lines)):
        current_line = lines[i]

        # The fouth element in this list is a string, representing a label
        write_str = ""

        write_str = write_str + str(i) + " "

        # This dict is used to count each label's frequency
        max_dict = {}

        current_label = current_line[3]
                
        write_str += str(current_label) + "\n"
            
        f.write(write_str)




