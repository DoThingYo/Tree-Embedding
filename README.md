
# Efficient Tree-SVD for Subset Node Embedding over Large Dynamic Graphs

## Tested Environment
- Ubuntu
- C++ 11
- GCC
- Intel C++ Compiler

## Preparation
|Data Set|Millions|N|M|
|:------|:-----:|------:|------:|
|[large-wiki](https://github.com/zjlxgxz/DynamicPPE)|Yes|6,216,199|177,862,656|
|[mag-authors](https://github.com/zjlxgxz/DynamicPPE)|Yes|5,843,822|123,392,120|
|[patent](https://github.com/zjlxgxz/DynamicPPE)|Yes|2,738,013|13,960,811|
|[flickr-growth](http://konect.cc/networks/flickr-growth/)|Yes|2,302,925|33,140,017|
|[youtube-u-growth](http://konect.cc/networks/youtube-u-growth/)|Yes|3,223,589|9,375,374|
|[Twitter](https://law.di.unimi.it/webdata/twitter-2010/)|Yes|41,652,230|1,468,365,182|



Place the prepared data [file].txt in the [DY_Dataset] and [DY_LP_Dataset] for the node classification task and the link prediction task, respectively. Note that the first row of data is the number of nodes in the graph and each row are the starting and ending nodes of an edge [outNode] [inNode].

Note that directed graphs and undirected graphs should be processed separately. 

Datasets used in our paper should be placed in ./DY_Dataset for Node Classification task and ./DY_LP_Dataset for Link Prediction task, respectively.


Please download the hashing library from "https://github.com/ktprime/emhash" and put this folder as ./emhash in this directory


## Compilations
```sh
bash compile.sh
```



Note that We use "NC" as the abbreviation of Node Classification in the following and "LP" as an abbreviation of Link Prediction task in the following descriptions.
At the same time, we use "queryname" to refer to a dataset, for example, it should be "large-wiki" for the wikipedia dataset.

Because directed and undirected graphs are usually processed differently, we often write two type of programs as "u" and "d" to process them, respectively.
Furthermore, because for NC task, we often need to have labels information as part of the program, programs of NC and LP are usually two files, too.

## Data Preprocessing:
We create a folder for each dataset as "./DY_Dataset/queryname/" for NC task and as "./DY_LP_Dataset/queryname/" for LP task,
for example: we create folder "./DY_Dataset/large-wiki/" for dataset "large-wiki" of NC task and create folder "./DY_LP_Dataset/youtube-u-growth/" 
for dataset "youtube-u-growth" of LP task.


### First, we create the directories for each datasets.

```
mkdir ./DY_EB

mkdir ./DY_Dataset
mkdir ./LABEL


mkdir ./DY_LP_Dataset


mkdir ./DY_LP_Dataset/Target
mkdir ./DY_Dataset/Target


mkdir ./DY_Dataset/large-wiki
mkdir ./DY_Dataset/mag-authors
mkdir ./DY_Dataset/patent


mkdir ./DY_LP_Dataset/mag-authors
mkdir ./DY_LP_Dataset/youtube-u-growth
mkdir ./DY_LP_Dataset/flickr-growth
mkdir ./DY_LP_Dataset/Twitter

mkdir ./SIGMOD2023
```



### Second, Download the dataset from corresponding address:
we used the same NC dataset as the baseline "DynPPE" which is given in https://github.com/zjlxgxz/DynamicPPE, they give a dropbox link as https://www.dropbox.com/sh/g3i95yttpjhgm2l/AAD8pF0XtgFv0fzmTrrOO4BWa?dl=0


#### the files for large-wiki dataset should be these two files, put them in ./DY_Dataset/large-wiki/:
enwiki20_edges.json
enwiki20_nodes.json

#### the files for mag-authors dataset should be these two files, put them in ./DY_Dataset/mag-authors/: 
mag_2019_edges.json
mag_2019_nodes.json

#### the files for patent dataset should be these two files, put them in ./DY_Dataset/patent/: 
patent_edges.json
patent_nodes.json


#### the file for flickr-growth(http://konect.cc/networks/flickr-growth/) dataset should be "out.flickr-growth", put it in ./DY_LP_Dataset/flickr-growth/ .


#### the file for youtube-u-growth(http://konect.cc/networks/youtube-u-growth/) dataset should be "out.youtube-u-growth", put it in ./DY_LP_Dataset/youtube-u-growth/ .





### Third, we preprocess all nodes and edges of each dataset into their corresponding folders with timestamp.

#### NC:For large-wiki dataset:


```
python process-edges-wiki.py large-wiki 6216199 > process-wiki-edges.txt
```

```
python process-nodes-wiki.py > process-wiki-nodes.txt
```



#### NC:For mag-authors dataset:

```
python process-edges-mag.py mag-authors 5843822 > process-mag-edges.txt
```

```
python process-nodes-mag.py> process-mag-nodes.txt
```



#### NC:For patent dataset:

```
python process-edges-patent.py patent 2738013 > process-patent-edges.txt
```

```
python process-nodes-patent.py > process-patent-nodes.txt
```



#### LP:For flickr-growth dataset:

```
python process-LP-month-sep.py flickr-growth 2302926 > process-flickr-edges.txt
```


#### LP:For youtube-u-growth dataset:
```
python process-LP-month-sep.py youtube-u-growth 3223590 > process-youtube-edges.txt
```











### Fourth, We use randomly generate the subset nodes for each dataset of Node Classfication task and Link Prediction task, with "$queryname" to represent the dataset name, for example "$queryname = large-wiki"

#### For Node Classification Task: We use RANDOM_SAMPLE_POINTS_NC to put the subset nodes into "./DY_Dataset/Target/$queryname.txt". And we put the labels of subset nodes into "./LABEL/$queryname.txt". Finally we write the edges information of each snapshot into "./DY_Dataset/$queryname/config.txt".


```
./RANDOM_SAMPLE_POINTS_NC large-wiki 3000 6216199
```

```
./RANDOM_SAMPLE_POINTS_NC patent 3000 2738013
```

```
./RANDOM_SAMPLE_POINTS_NC mag-authors 3000 5843822
```


#### For Link Prediction Task: We use RANDOM_SAMPLE_POINTS_LP to put the subset nodes into "./DY_LP_Dataset/Target/$queryname.txt". And we write the edges information of each snapshot into "./DY_LP_Dataset/$queryname/config_Alledges.txt".

```
./RANDOM_SAMPLE_POINTS_LP flickr-growth 3000 2302926
```

```
./RANDOM_SAMPLE_POINTS_LP youtube-u-growth 3000 3223590
```

```
cp ./DY_Dataset/Target/mag-authors.txt ./DY_LP_Dataset/Target/mag-authors.txt
```

```
./RANDOM_SAMPLE_POINTS_LP Twitter 3000 41652230
```


then put these generated subset nodes of concern into "DY_Dataset/Target/" for Node Classification task and "DY_LP_Dataset/Target/" for Link Prediction task. 






### Fifth, generate corresponding training and testing edges for LP task and write the corresponding information into ./DY_LP_Dataset/$queryname/config.txt .

```
./GEN_SUBSET_LP_DATA_D flickr-growth 0.3 2302926
```

```
./GEN_SUBSET_LP_DATA_U youtube-u-growth 0.3 3223590
```

```
./GEN_SUBSET_LP_DATA_U mag-authors 0.3 5843822
```

```
./GEN_SUBSET_LP_DATA_D Twitter 0.3 41652230
```







### Sixth, generate the batch update datasets.

```
mkdir ./DY_Dataset/large-wiki_batch
mkdir ./DY_Dataset/mag-authors_batch
mkdir ./DY_Dataset/patent_batch


mkdir ./DY_LP_Dataset/mag-authors_batch
mkdir ./DY_LP_Dataset/youtube-u-growth_batch
mkdir ./DY_LP_Dataset/flickr-growth_batch
mkdir ./DY_LP_Dataset/Twitter_batch
```


#### For NC task, Split 1 millions consecutive batch edges into 100 files, each containing 10000 edges, write the corresponding information into ./DY_Dataset/($queryname)_batch/config.txt and the corresponding edge files into ./DY_Dataset/($queryname)_batch/ folder.


```
./SPLIT_EDGES_NC large-wiki 9
```

```
./SPLIT_EDGES_NC mag-authors 7
```

```
./SPLIT_EDGES_NC patent 18
```


#### For LP task, Split 1 millions consecutive batch edges into 100 files, each containing 10000 edges, write the corresponding information into ./DY_LP_Dataset/($queryname)_batch/config.txt and the corresponding edge files into ./DY_LP_Dataset/($queryname)_batch/ folder.

```
./SPLIT_EDGES_LP flickr-growth 3
```

```
./SPLIT_EDGES_LP mag-authors 7
```

```
./SPLIT_EDGES_LP youtube-u-growth 8
```

```
./SPLIT_EDGES_LP Twitter 3
```

#### copy label files for batch update
```
cp ./LABEL/large-wiki_subset.txt ./LABEL/large-wiki_batch_subset.txt
```

```
cp ./LABEL/patent_subset.txt ./LABEL/patent_batch_subset.txt
```

```
cp ./LABEL/mag-authors_subset.txt ./LABEL/mag-authors_batch_subset.txt
```


#### copy testing edge files for batch update
```
cp DY_LP_Dataset/youtube-u-growth/youtube-u-growth-Pos_LP_Test.txt DY_LP_Dataset/youtube-u-growth_batch/youtube-u-growth_batch-Pos_LP_Test.txt 
```

```
cp DY_LP_Dataset/youtube-u-growth/youtube-u-growth-Neg_LP_Test.txt DY_LP_Dataset/youtube-u-growth_batch/youtube-u-growth_batch-Neg_LP_Test.txt
```

```
cp DY_LP_Dataset/flickr-growth/flickr-growth-Pos_LP_Test.txt DY_LP_Dataset/flickr-growth_batch/flickr-growth_batch-Pos_LP_Test.txt 
```

```
cp DY_LP_Dataset/flickr-growth/flickr-growth-Neg_LP_Test.txt DY_LP_Dataset/flickr-growth_batch/flickr-growth_batch-Neg_LP_Test.txt
```


```
cp DY_LP_Dataset/mag-authors/mag-authors-Pos_LP_Test.txt DY_LP_Dataset/mag-authors_batch/mag-authors_batch-Pos_LP_Test.txt 
```


```
cp DY_LP_Dataset/mag-authors/mag-authors-Neg_LP_Test.txt DY_LP_Dataset/mag-authors_batch/mag-authors_batch-Neg_LP_Test.txt
```


```
cp DY_LP_Dataset/Twitter/Twitter-Pos_LP_Test.txt DY_LP_Dataset/Twitter_batch/Twitter_batch-Pos_LP_Test.txt 
```


```
cp DY_LP_Dataset/Twitter/Twitter-Neg_LP_Test.txt DY_LP_Dataset/Twitter_batch/Twitter_batch-Neg_LP_Test.txt
```









## Run the bash file for experiments.
```
nohup bash run.sh
```
















































