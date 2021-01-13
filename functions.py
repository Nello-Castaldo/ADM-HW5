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
    graphtype = nx.DiGraph()    # since we concluded that the graph has to be directed, we use the method DiGraph of networkx
    G = nx.from_pandas_edgelist(dataframe,'Out','In', create_using=graphtype)

    return G

def inverted_dictionary(graph):
    '''
    Here we remove all the non necessary categories, as requested by the assignment,
    and return a "cleaned" version of the given data
    '''
    dictionary= defaultdict(list)
    final_inv_dic=defaultdict(list)
    a = list(graph.nodes)

    categories = open('wiki-topcats-categories.txt','r')
    items = categories.readlines()

    for item in items:
        '''
        From "wiki-topcats-categories.txt" we collect the necessary information,
        prepare the data and store such result in an intermediate dictionary
        '''
        value=item.strip().split(';')[1].split()
        key=item.strip().split(';')[0]
        key=key.replace('Category:','')
        dictionary[key]=value

    inv_dic = defaultdict(list)

    for key in dictionary.keys():
        '''
        In here we "invert" the previous dictionary, so that we map each node to
        its categories (each node might belong to more than one category)
        '''
        for value in dictionary[key]:
            inv_dic[value].append(key)

    for item in inv_dic.keys():
        '''
        Here we associate each node to a single category and return a dictionary
        with the result
        '''
        try:
            category= get_category(inv_dic[item])
        except :
            print(item)

        final_inv_dic[item].append(category)

    key_list = final_inv_dic.keys()
    cleaned_dic = {}
    i = 0

    for x in a:
        '''
        We are ready to output a dictionary "cleaned" from the categories,
        and hence the nodes, that does not belong to the created graph
        '''
        cleaned_dic[str(x)] = final_inv_dic[str(x)]
        i += 1

    return cleaned_dic

def get_category(cat_list):
    '''
    Remember, each node might be shared among multiple categories. Therefore
    we need to choose a single category for each node. To do so, we make use 
    of the numpy random.randint method
    '''
    length= len(cat_list)

    if length > 1:
        index= np.random.randint(length, size=1)[0]
    else:
        index=0

    new_value= cat_list[index]
    return new_value

def category_dict(cleaned_dic):
    '''
    With respect to the "cleaned" dictionary, it is also useful to have
    a dictionary associating to each category a list of all its related
    nodes
    '''
    category_dict = defaultdict(list)
    for key in cleaned_dic.keys():

        for item in cleaned_dic[key]:
            category_dict[item].append(key)

    return category_dict

def get_degree_dict(graph):
    '''
    To plot the degree distribution of our graph, we need to calculate the
    out-degree for each node. We store all these computations in a dictionary
    '''
    degree_dic = defaultdict(int)

    for item in graph:
        degree_dic[item]= len(graph[item])

    return degree_dic

def check_if_is_direct(graph):
    '''
    This function checks if there are two nodes, of which only one is connected
    to the other. If so, the Graph is said to be directed, and the output is True
    '''
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
    '''
    We create a subgraph out of two given categories. We choose to output an 
    Undirected Graph rather than Directed because we are interested in 
    disconnecting the two given nodes, not at the direction of the edges to
    be removed. Such decision is only going to improve the effectiveness of
    the algorithm we are going to implement
    '''
    Category_1 = input('Insert First Category: ').strip()
    Category_2 = input('Insert Second Category: ').strip()
    nodes_1 = list(map(int,category_dict[Category_1]))
    nodes_2 = list(map(int,category_dict[Category_2]))
    nodes = nodes_1 + nodes_2
    H = graph.subgraph(nodes).copy()

    return nx.Graph(H)

def backtrace(parent, start, end):
    '''
    This function is propedeutic to "bfs_min_cut" and is used to 
    trace back the path between the node in origin and the one of
    destination
    '''
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def bfs_min_cut(graph, u, v):
    '''
    This is a modified version of the Breadth First Search algorithm that we modified so 
    that is stops iterating when a connection between two given node has been found
    '''
    visited = [] # List to keep track of visited nodes.
    queue = []   # Initialize a queue
    path=[]
    parents={}
    node=u
    flag=False 
    visited.append(node) # nodes already visited
    queue.append(node)  # nodes of whom we still have to inspect their neighbours
    counter=0
    while queue: # keep iterating as long as there are nodes to explore
        s = queue.pop(0) # get the first node from the queue

        for neighbour in graph[s]: # for each neighbour of the node we are currently visiting
            if neighbour not in visited: # if the current neighbour has not yet been visited
                parents[neighbour]=s
                visited.append(neighbour) # add it to the list of visited neighbours
                queue.append(neighbour) # add it to the list of neighbours yet to be visited
                if neighbour==v:    # if the node given node has been found
                    path= backtrace(parents,u,v)    # trace back the path between "u" and "v"
                    flag=True   # output True to let the "min_cut" function understand that there is a connection between "u" and "v"
                    print('Found') # a connection has been found
                    break # no need to keep iterating once the connection has been found
       
        counter+=1  # once we are done with inspecting all the neighbours of the current node, we can proceed further of one level

    return counter, path, flag

def min_cut(graph, u, v):
    '''
    Keep running iteratively the function "bfs_min_cut" as long as paths 
    between "u" and "v" are being found
    '''
    H1=graph.copy()
    min_edges=0
    counter, path, flag = bfs_min_cut(H1,u,v)
    print(path)

    while flag:
        min_edges+=1
        print(min_edges)
        i=0
        while i < len(path)-1:  # we remove all the edges between the nodes that belong to the path we just found
            try:
                H1.remove_edge(path[i],path[i+1])
            except:
                H1.remove_edge(path[i+1],path[i])
            i += 1

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
        visited.append(node) # nodes already visited
        queue.append(node)  # nodes of whom we still have to inspect their neighbours
        counter = 0
        while queue:    # keep iterating as long as there are nodes to explore
            s = queue.pop(0) # get the first node from the queue

            for neighbour in graph[s]:  # for each neighbour of the node we are currently visiting
                if neighbour not in visited: # if the current neighbour has not yet been visited
                    visited.append(neighbour) # add it to the list of visited neighbours
                    queue.append(neighbour) # add it to the list of neighbours yet to be visited
                    current_cat = cleaned_dict[str(neighbour)][0]   # we get the name of the category the current node belongs to
                    a[current_cat].append(counter) # add the current level to the related category. The level represent the distance between the neighbour we are visiting and the origin node 

            counter+=1 # go deeper of one level

    final = {x: np.median(a[x]) if (len(a[x]) > 0) else 0.0 for x in a.keys() }     # we computer the median of shortest paths for each category and save the result in a dictionary
    final_sorted=sorted(final.items(), key=operator.itemgetter(1), reverse=True)    # we sort the previous dictionary according to the value of the medians, from highest to lowest
    result = [x[0] for x in final_sorted]   # extract the list of the sorted categories from the previous variable

    return result   # return the list

def model_network(graph, category_dict):
    '''
    We use the minors.contracted_nodes method of networkx to join together all the nodes of 
    each category, while preserving all their edges. The result is going to be a Graph model
    representing all the categories (we consider them as supernodes)
    '''
    g1 = graph.copy()
    for category in category_dict.keys():
        nodes = list(map(int,category_dict[category]))
        m = nodes[0]
        for node in nodes[1:]:
            nx.algorithms.minors.contracted_nodes(g1, m, node, self_loops=False, copy=False)

    return g1

def get_mapping_dict(category_graph):
    '''
    The purpose of this function is specified on the Jupyter notebook
    '''
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
    '''
    This function aims at computing the google matrix related to the network
    representing all our categories. 
    '''
    pagerank_dict = {}
    final_result = {}
    n = len(graph)
    L = np.zeros((n,n))     # we initialize a matrix that has size equal to the number of nodes in our graph
    r = np.array([1/n for i in range(n)])   # we initialize a vector with a temporary assigned probability for each node in the graph
    for node in graph:  
        out_deg = len(graph[node])  # we compute the length of the degree for each node in the graph
        i = category_map_int[node]  # we map the current node to an index of the google matrix [i]
        for item in graph[node]:
            j = category_map_int[item]  # we map each neighbour of the current node to an index of the google matrix [j]
            L[i][j] += 1/out_deg    # we update the google matrix at the index i,j with the probability of going from node_i to node_j (1/out_deg because they are uniform by definition)

    L = L.T     # according to formulas, we need to work over columns rather than rows. Therefore we comput the transpose of our original google matrix

    for k in range(10):
        r = 0.85*(np.dot(L,r))+(1-0.85)/n   # here we implement the official formula, with a value for alpha of 0.85. We improve iteratively the original vector of probablity, that will result in not being anymore uniform

    for i in range(len(r)):
        node = inv_category_map[i]  # from each index, we calculate back the original value of each node
        pagerank_dict[node] = r[i]  # we associate to each category the respective pagerank

    for key in pagerank_dict:
        cat = cleaned_dict[str(key)][0]
        final_result[cat] = pagerank_dict[key]  # here we associate to each category name the respective pagerank

    pagerank_list = sorted(final_result.items(), key=operator.itemgetter(1), reverse=True)  # we sort the final dictionary according each pagerank value
    pd_pagerank = pd.DataFrame(pagerank_list, columns=['Category','PageRank'], index=[i+1 for i in range(len(pagerank_list))])  # we turn the dictionary into a pandas dataframe

    return pd_pagerank

def get_pagerank_head(pagerank, amount):
    return pagerank.head(amount)

def plot_pagerank(pagerank):
    plt.figure(figsize=(16,9))
    plt.barh(pagerank['Category'],pagerank['PageRank'])
    plt.show()