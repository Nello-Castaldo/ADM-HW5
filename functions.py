import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import scipy
import math
import operator
from collections import defaultdict



def generate_graph():
    dataframe=pd.read_csv('wikigraph_reduced.csv', sep="\t", index_col=0)
    dataframe.columns= ['Out','In']
    graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(dataframe,'Out','In', create_using=graphtype)

    return G



def inverted_dictionary(graph):
    dictionary= defaultdict(list)
    final_inv_dic=defaultdict(list)
    a = list(graph.nodes)

    file = open('wiki-topcats-categories.txt','r')
    var = file.readlines()

    for item in var:
        value=item.strip().split(';')[1].split()
        key=item.strip().split(';')[0]
        key=key.replace('Category:','')
        dictionary[key]=value

    inv_dic= defaultdict(list)

    for key in dictionary.keys():
        for value in dictionary[key]:
            inv_dic[value].append(key)

    for item in inv_dic.keys():
        try:
            category= get_category(inv_dic[item])
        except :
            print(item)
        
        final_inv_dic[item].append(category)
    
    key_list = final_inv_dic.keys()
    cleaned_dic = {}
    i = 0

    for x in a:
        cleaned_dic[str(x)] = final_inv_dic[str(x)]
        i += 1

    return cleaned_dic



def get_category(cat_list):
    length= len(cat_list)
    
    if length > 1:
        index= np.random.randint(length, size=1)[0]
    else:
        index=0

    new_value= cat_list[index]
    return new_value



def category_dict(cleaned_dic):
    category_dict = defaultdict(list)
    for key in cleaned_dic.keys():
        
        for item in cleaned_dic[key]:
            category_dict[item].append(key)

    return category_dict



def check_if_is_direct(graph):
    flag = False

    for node in graph:
        for item in graph[node]:
            if node not in graph[item]:
                flag = True
                break
    
    return flag