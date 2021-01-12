import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

    categories = open('wiki-topcats-categories.txt','r')
    items = categories.readlines()

    for item in items:
        value=item.strip().split(';')[1].split()
        key=item.strip().split(';')[0]
        key=key.replace('Category:','')
        dictionary[key]=value

    inv_dic = defaultdict(list)

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

def get_degree_dict(graph):
    degree_dic = defaultdict(int)

    for item in graph:
        degree_dic[item]= len(graph[item])

    return degree_dic

def check_if_is_direct(graph):
    flag = False

    for node in graph:
        for item in graph[node]:
            if node not in graph[item]:
                flag = True
                break

    return flag

def get_articles(graph):
    return nx.number_of_nodes(graph)

def get_hyperlinks(graph):
    return nx.number_of_edges(graph)

def plot_our_graph(graph):
    distribution = [len(graph[item]) for item in graph]
    mean = round(sum(distribution)/len(distribution))
    sum_dist =sum(distribution)
    normalized =[(float(i)/sum_dist) for i in distribution]

    plt.figure(figsize=(16,9))
    plt.plot(range(len(distribution)),normalized)
    plt.title('Graph degree distribution')
    plt.show()

def path_within_clicks(graph, node, d):
    result = set()  # in here we insert all the pages we can reach within "d" clicks
    arr=[x for x in graph[node]] # here we store the list of all neighbours of the first node
    while(d>0): # as long as we have available clicks we keep iterating
        new_arr=[] 
        for item in arr: # for each neighbour
            result.add(item) # we add it to the result set
            new_arr.append(item) # and we also update the list of neighbours we still have to inspect 

        d=d-1 # we update the number of clicks available
        arr=[x for y in new_arr for x in graph[y]] # we update the list of neighbours to be visited

    return result

def bfs(graph, category_dict, degree_dict):
    visited = [] # List to keep track of visited nodes.
    queue = []   # Initialize a queue
    Category = input('Insert the category: ')
    p = input('Insert pages separated by a blank space: ')
    p = p.split()
    p_arr = list(map(int,p)) # we convert the given nodes as integers and inser them into an array

    V= {x: degree_dict[x] for x in p_arr} # we get the degree value for each given node
    v= max(V.items(), key=operator.itemgetter(1))[0] # we select the node with the highest degree

    node=v
    visited.append(node) # List of visited nodes
    queue.append(node)   # nodes of whom I have to visit neighbours
    counter=0
    while p_arr and queue: # keep iterating as long as there are nodes to visit or to find 
        s = queue.pop(0) # we take the first element of the queue 

        for neighbour in graph[s]:  # For each element of the neighbour we are iterating
            if neighbour not in visited: # if it has not been visited yet
                visited.append(neighbour) # then add it to the list of visited nodes
                queue.append(neighbour) # add it to the list of nodes I still have to inspect
                if neighbour in p_arr:
                    p_arr.remove(neighbour) # if node is found, we remove it from the list of nodes to be found

        counter+=1 # we update the number of levels we have visited

    if len(p_arr)>0:
        print('Not possible')
    else:
        print('Found')
        
    return counter # we return the number of levels visited by the algorithm

def create_subGraph(graph, category_dict):
    Category_1 = input('Insert First Category: ').strip()
    Category_2 = input('Insert Second Category: ').strip()
    nodes_1 = list(map(int,category_dict[Category_1]))
    nodes_2 = list(map(int,category_dict[Category_2]))
    nodes = nodes_1 + nodes_2
    H = graph.subgraph(nodes).copy()

    return nx.Graph(H) # lo trattiamo come diretto perchè siamo interessati solo alla disconnessione di due nodi, non alla direzione degli archi che rimuoviamo

def backtrace(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def bfs_min_cut(graph, u, v):
    visited = [] # List to keep track of visited nodes.
    queue = []   # Initialize a queue
    path=[]
    parents={}
    node=u
    flag=False 
    visited.append(node) #nodi visitati 
    queue.append(node)  # nodi di cui devo visitare i vicini
    counter=0
    while queue:# finchè ci sono nodi da visitare 
        s = queue.pop(0) # prendi il primo elemento della coda  

        for neighbour in graph[s]: # per ogni vicino dell'elemento in questione 
            if neighbour not in visited: # se non è stato ancora visitato
                parents[neighbour]=s
                visited.append(neighbour) #aggiungilo alla lista dei visitati
                queue.append(neighbour) # aggiungilo alla lista dei nodi da attraversare
                if neighbour==v:
                    path= backtrace(parents,u,v)
                    flag=True 
                    print('Found')
                    break
       
        counter+=1

    return counter, path, flag 

def min_cut(graph, u, v):
    H1=graph.copy()
    min_edges=0
    counter, path, flag = bfs_min_cut(H1,u,v)
    print(path)

    while flag:
        min_edges+=1
        i=0
        while i < len(path)-1:
            try:
                H1.remove_edge(path[i],path[i+1])
            except:
                H1.remove_edge(path[i+1],path[i])

            counter, path, flag = bfs_min_cut(H1, u, v)

    return min_edges

def get_shortest_paths(graph, category_dict, cleaned_dict):
    queue = []
    a = defaultdict(list)
    category = input('Insert a Category: ')
    nodes = list(map(int,category_dict[category]))
    all_categories = list(category_dict.keys())
    all_categories.remove(category)

    for node in nodes:
        visited = []
        visited.append(node) #nodi visitati 
        queue.append(node)  # nodi di cui devo visitare i vicini
        counter = 0
        while queue:# finchè ci sono nodi da visitare 
            s = queue.pop(0) # prendi il primo elemento della coda  

            for neighbour in graph[s]:  # per ogni vicino dell'elemento in questione 
                if neighbour not in visited: # se non è stato ancora visitato
                    visited.append(neighbour) #aggiungilo alla lista dei visitati
                    queue.append(neighbour) # aggiungilo alla lista dei nodi da attraversare
                    current_cat = cleaned_dict[str(neighbour)][0]
                    a[current_cat].append(counter)

            counter+=1

    final = {x: np.median(a[x]) if (len(a[x]) > 0) else 0.0 for x in a.keys() }
    final_sorted=sorted(final.items(), key=operator.itemgetter(1), reverse=True)
    result = [x[0] for x in final_sorted]

    return result

def model_network(graph, category_dict):
    g1 = graph.copy()
    for category in category_dict.keys():
        nodes = list(map(int,category_dict[category]))
        m = nodes[0]
        for node in nodes[1:]:
            nx.algorithms.minors.contracted_nodes(g1, m, node, self_loops=False, copy=False)

    return g1

def get_mapping_dict(category_graph):
    category_map_integer = defaultdict(int)
    inv_category_map = defaultdict(int)
    i = 0
    for item in category_graph:
        category_map_integer[item] = i
        i += 1

    for item in category_map_integer.keys():
        key = category_map_integer[item]
        inv_category_map[key] = item

    return category_map_integer, inv_category_map

def pagerank(graph, category_map_int, inv_category_map, cleaned_dict):
    pagerank_dict = {}
    final_result = {}
    n = len(graph)
    L = np.zeros((n,n))
    r = np.array([1/n for i in range(n)])
    for node in graph:
        out_deg = len(graph[node])
        i = category_map_int[node]
        for item in graph[node]:
            j = category_map_int[item]
            L[i][j] += 1/out_deg

    L = L.T

    for k in range(10):
        r = 0.85*(np.dot(L,r))+(1-0.85)/n

    for i in range(len(r)):
        node = inv_category_map[i]
        pagerank_dict[node] = r[i]

    for key in pagerank_dict:
        cat = cleaned_dict[str(key)][0]
        final_result[cat] = pagerank_dict[key]

    pagerank_list = sorted(final_result.items(), key=operator.itemgetter(1), reverse=True)
    pd_pagerank = pd.DataFrame(pagerank_list, columns=['Category','PageRank'], index=[i+1 for i in range(len(pagerank_list))])

    return pd_pagerank

def get_pagerank_head(pagerank, amount):
    return pagerank.head(amount)

def plot_pagerank(pagerank):
    plt.figure(figsize=(16,9))
    plt.barh(pagerank['Category'],pagerank['PageRank'])
    plt.show()