


nohup ./TREE_U_LP youtube-u-growth DY_EB/ 0.5 12 0.00000001 64 64 8 3223590 100 100 0.3 > ./SIGMOD2023/all_LP_1e-8_youtube-u-growth_tree_result.txt


nohup ./TREE_U_LP mag-authors DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 100 100 0.3 > ./SIGMOD2023/all_LP_1e-8_mag-authors_tree_result.txt


nohup ./TREE_D_LP flickr-growth DY_EB/ 0.5 12 0.0000001 64 64 8 2302926 100 100 0.3 > ./SIGMOD2023/all_LP_1e-7_flickr-growth_tree_result.txt


nohup ./TREE_D_LP Twitter LP_EB/ 0.5 12 0.000001 64 64 8 41652230 200 200 0.3 > ./SIGMOD2023/all_LP_1e-6_Twitter_tree_result.txt





nohup ./TREE_U_NC patent DY_EB/ 0.5 12 0.00000001 64 64 8 2738013 100 100 0.3 > ./SIGMOD2023/all_NC_1e-8_patent_tree.txt
nohup python labelclassification_dynamic.py patent 0.300000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/all_NC_1e-8_patent_tree_result.txt


nohup ./TREE_U_NC mag-authors DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 100 100 0.3 > ./SIGMOD2023/all_NC_1e-8_mag-authors_tree.txt
nohup python labelclassification_dynamic_top1_label.py mag-authors 0.300000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/all_NC_1e-8_mag-authors_tree_result.txt


nohup ./TREE_D_NC large-wiki DY_EB/ 0.5 12 0.00001 64 64 8 6216199 100 100 0.3 > ./SIGMOD2023/all_NC_1e-5_large-wiki_tree.txt
nohup python labelclassification_dynamic.py large-wiki 0.300000_6_64_8_De_svd_d_Dppr_parallel_bound_nodegree > ./SIGMOD2023/all_NC_1e-5_large-wiki_tree_result.txt





# 0.65
nohup ./TREE_D_NC large-wiki_batch DY_EB/ 0.5 12 0.00001 64 64 8 6216199 9 9 0.65 > ./SIGMOD2023/1e-5-large-wiki_batch_SNAP9_0.65_tree.txt
nohup python labelclassification_dynamic.py large-wiki_batch 0.650000_6_64_8_De_svd_d_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-5-large-wiki_batch_SNAP9_0.65_tree_result.txt

nohup ./TREE_U_NC mag-authors_batch DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 7 7 0.65 > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.65_tree.txt
nohup python labelclassification_dynamic_top1_label.py mag-authors_batch 0.650000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.65_tree_result.txt

nohup ./TREE_U_NC patent_batch DY_EB/ 0.5 12 0.00000001 64 64 8 2738013 18 18 0.65 > ./SIGMOD2023/1e-8-patent_batch_SNAP18_0.65_tree.txt
nohup python labelclassification_dynamic.py patent_batch 0.650000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-8-patent_batch_SNAP18_0.65_tree_result.txt

nohup ./TREE_U_LP youtube-u-growth_batch DY_EB/ 0.5 12 0.00000001 64 64 8 3223590 8 8 0.65 > ./SIGMOD2023/1e-8-youtube-u-growth_batch_SNAP8_0.65_tree_result.txt

nohup ./TREE_U_LP mag-authors_batch DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 7 7 0.65 > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.65_tree_result.txt

nohup ./TREE_D_LP flickr-growth_batch DY_EB/ 0.5 12 0.0000001 64 64 8 2302926 3 3 0.65 > ./SIGMOD2023/1e-7-flickr-growth_batch_SNAP3_0.65_tree_result.txt

nohup ./TREE_D_LP Twitter LP_EB/ 0.5 12 0.000001 64 64 8 41652230 3 3 0.65 > ./SIGMOD2023/1e-6-Twitter_batch_SNAP3_0.65_tree_result.txt


# 0.6
nohup ./TREE_D_NC large-wiki_batch DY_EB/ 0.5 12 0.00001 64 64 8 6216199 9 9 0.6 > ./SIGMOD2023/1e-5-large-wiki_batch_SNAP9_0.6_tree.txt
nohup python labelclassification_dynamic.py large-wiki_batch 0.600000_6_64_8_De_svd_d_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-5-large-wiki_batch_SNAP9_0.6_tree_result.txt

nohup ./TREE_U_NC mag-authors_batch DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 7 7 0.6 > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.6_tree.txt
nohup python labelclassification_dynamic_top1_label.py mag-authors_batch 0.600000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.6_tree_result.txt

nohup ./TREE_U_NC patent_batch DY_EB/ 0.5 12 0.00000001 64 64 8 2738013 18 18 0.6 > ./SIGMOD2023/1e-8-patent_batch_SNAP18_0.6_tree.txt
nohup python labelclassification_dynamic.py patent_batch 0.600000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-8-patent_batch_SNAP18_0.6_tree_result.txt

nohup ./TREE_U_LP youtube-u-growth_batch DY_EB/ 0.5 12 0.00000001 64 64 8 3223590 8 8 0.6 > ./SIGMOD2023/1e-8-youtube-u-growth_batch_SNAP8_0.6_tree_result.txt

nohup ./TREE_U_LP mag-authors_batch DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 7 7 0.6 > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.6_tree_result.txt

nohup ./TREE_D_LP flickr-growth_batch DY_EB/ 0.5 12 0.0000001 64 64 8 2302926 3 3 0.6 > ./SIGMOD2023/1e-7-flickr-growth_batch_SNAP3_0.6_tree_result.txt

nohup ./TREE_D_LP Twitter LP_EB/ 0.5 12 0.000001 64 64 8 41652230 3 3 0.6 > ./SIGMOD2023/1e-6-Twitter_batch_SNAP3_0.6_tree_result.txt


# 0.55
nohup ./TREE_D_NC large-wiki_batch DY_EB/ 0.5 12 0.00001 64 64 8 6216199 9 9 0.55 > ./SIGMOD2023/1e-5-large-wiki_batch_SNAP9_0.55_tree.txt
nohup python labelclassification_dynamic.py large-wiki_batch 0.550000_6_64_8_De_svd_d_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-5-large-wiki_batch_SNAP9_0.55_tree_result.txt

nohup ./TREE_U_NC mag-authors_batch DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 7 7 0.55 > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.55_tree.txt
nohup python labelclassification_dynamic_top1_label.py mag-authors_batch 0.550000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.55_tree_result.txt

nohup ./TREE_U_NC patent_batch DY_EB/ 0.5 12 0.00000001 64 64 8 2738013 18 18 0.55 > ./SIGMOD2023/1e-8-patent_batch_SNAP18_0.55_tree.txt
nohup python labelclassification_dynamic.py patent_batch 0.550000_0_64_8_De_svd_u_Dppr_parallel_bound_nodegree > ./SIGMOD2023/1e-8-patent_batch_SNAP18_0.55_tree_result.txt

nohup ./TREE_U_LP youtube-u-growth_batch DY_EB/ 0.5 12 0.00000001 64 64 8 3223590 8 8 0.55 > ./SIGMOD2023/1e-8-youtube-u-growth_batch_SNAP8_0.55_tree_result.txt

nohup ./TREE_U_LP mag-authors_batch DY_EB/ 0.5 12 0.00000001 64 64 8 5843822 7 7 0.55 > ./SIGMOD2023/1e-8-mag-authors_batch_SNAP7_0.55_tree_result.txt

nohup ./TREE_D_LP flickr-growth_batch DY_EB/ 0.5 12 0.0000001 64 64 8 2302926 3 3 0.55 > ./SIGMOD2023/1e-7-flickr-growth_batch_SNAP3_0.55_tree_result.txt

nohup ./TREE_D_LP Twitter LP_EB/ 0.5 12 0.000001 64 64 8 41652230 3 3 0.55 > ./SIGMOD2023/1e-6-Twitter_batch_SNAP3_0.55_tree_result.txt








