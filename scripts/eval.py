import argparse
import json
import os
import numpy as np
import pickle

def main():
    path_TREx_relations = "/private/home/millicentli/BERT-kNN/data/relations.jsonl"
    output_path = "/private/home/millicentli/BERT-kNN/output/results/bert_base/"
    relations_GoogleRE = ["place_of_birth", "date_of_birth", "place_of_death"]
    relations_macro = {"TREx": [], "GoogleRE": relations_GoogleRE, "Squad": ["Squad"], "ConceptNet": ["ConceptNet"]}

    with open(path_TREx_relations, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        relations_macro["TREx"].append(result["relation"])

    for r_macro in relations_macro:
        print(r_macro)
        P_1 = 0
        P_1_BERT = 0
        P_1_NN = 0
        num = 0
        for r_micro in relations_macro[r_macro]:
            if os.path.isdir(output_path + r_micro + "/"):
                with open(output_path + r_micro + '/result.pkl', 'rb') as f:
                    data = pickle.load(f)
                for d in data["list_of_results"]:
                    # print("Here's d:", d)
                    P_1 += d['masked_topk']["P_AT_1"]
                    P_1_BERT += d['masked_topk']["P_AT_1_bert"]
                    P_1_NN += d['masked_topk']["P_AT_1_nn"]
                    num += 1
        print("BERT-kNN: ", P_1/num)
        print("BERT: ", P_1_BERT/num)
        print("kNN: ", P_1_NN/num)
        print("")

if __name__ == "__main__":
    main()

# import pickle
# import glob
# import os
# import json
# import numpy as np

# experiment = "results"
# # output_path = "../output/" + experiment + "/bert_base/"
# output_path = "/private/home/millicentli/BERT-kNN/output/results/bert_base/"
# P_path = glob.glob(output_path + "P*")
# relations_collected = ["TREx", "GoogleRE", "Squad", "ConceptNet"]
# relations = ["place_of_birth", "date_of_birth", "place_of_death"]
# # path_TREx_relations = "/mounts/work/kassner/LAMA/data/relations.jsonl"
# path_TREx_relations = "/private/home/millicentli/BERT-kNN/data/relations.jsonl"

# with open(path_TREx_relations, 'r') as json_file:
#     json_list = list(json_file)

# GoogleRE_relations = relations

# TREx_relations = []
# for json_str in json_list:
#     result = json.loads(json_str)
#     relations.append(result["relation"])
#     TREx_relations.append(result["relation"])

# relations.append("Squad")
# relations.append("ConceptNet")

# P = {}
# results = {}
# for r in relations:
#     print(r)
#     P_1 = 0
#     num = 0
#     #try:
#     if os.path.isdir(output_path + r + "/"):
#         with open(output_path + r + '/result.pkl', 'rb') as f:
#             data = pickle.load(f)
#         for d in data["list_of_results"]:
#             P_1 += d['masked_topk']["P_AT_1"]
#             num += 1
#         results[r] = P_1/num
#     #except:
#     #    pass

# overall = 0
# num_overall = 0
# correlations_BERT = {}
# correlations_NN = {}
# correlations_BERT_KNN = {}

# for r in relations_collected:
#     correlations_BERT[r] = []
#     correlations_NN[r] = []
#     correlations_BERT_KNN[r] = []

# for r in relations_collected:
#     if r=="TREx":
#         num_P = 0
#         P_collected = 0
#         for P in TREx_relations:
#             if P in results:
#                 P_collected += results[P]
#                 num_P += 1

#         print(r)
#         print(P_collected/num_P)
#         num_overall += 1
#         overall += P_collected/num_P
#     elif r=="GoogleRE":
#         num_P = 0
#         P_collected = 0
#         for P in GoogleRE_relations:
#             if P in results:
#                 P_collected += results[P]
#                 num_P += 1

#         print(r)
#         print(P_collected/num_P)
#         num_overall += 1
#         overall += P_collected/num_P

#     else:
#         print(r)
#         print(results[r])
#         num_overall += 1
#         overall += results[r]

# print(np.mean(list(results.values())))