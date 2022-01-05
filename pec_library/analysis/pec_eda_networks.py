# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: pec_library
#     language: python
#     name: pec_library
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import re
import itertools
import boto3
import random

from datetime import datetime
from collections import Counter
import networkx as nx
from itertools import combinations
from pyvis.network import Network

from pec_library.getters.data_getters import get_library_data

# text cleaning packages
import string
from toolz import pipe
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pattern.text.en import singularize
import community as community_louvain
from time import sleep

# %%
# choose initial key word per mission for API search

asf_keyword = "heat pumps"
ahl_keyword = "obesity"
afs_keyword = "early childhood"

# %% [markdown]
# ## 0. Get data

# %%
# eda with asf_keyword

asf = get_library_data(asf_keyword)


# %% [markdown]
# ## 1.0 Clean subject data
#
# clean subject texts to:
#
# * lowercase
# * remove punctuation
# * lemmatise
# * remove whitespace
# * split subject into multiple subjects if subject contains /, ; or ,

# %%
# https://stackoverflow.com/questions/406121/flattening-a-shallow-list-in-python
def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


# %%
def clean_subject(subjects):
    """
    Args:
        subjects (list): list of lists of string subjects to be cleaned.
    
    Returns:
        list: list of lists of cleaned string subjects.
    """

    lemmatizer = WordNetLemmatizer()
    all_clean_subs = []

    for subject in subjects:
        # remove punct
        subject = [sub.translate(str.maketrans("", "", ".:()-")) for sub in subject]
        # remove stopwords
        subject = [
            sub.lower() for sub in subject if sub not in stopwords.words("english")
        ]
        # remove digits
        subject = [sub for sub in subject if not sub.isdigit()]
        # trim trailing whitespace
        subject = [re.sub("\s+", " ", sub).strip() for sub in subject]
        # lemmatize subject
        subject = [lemmatizer.lemmatize(sub) for sub in subject]
        # trim short subjects
        subject = [sub for sub in subject if len(sub) > 5]
        # singularise subject
        subject = [singularize(sub) for sub in subject]
        # split multiple subjects
        for clean_sub_ind, clean_sub in enumerate(subject):
            split_chars = "/,;"
            split_subs = re.split("[" + split_chars + "]", clean_sub)
            if len(split_subs) > 1:
                subject[clean_sub_ind] = [sub.strip() for sub in split_subs]
        all_clean_subs.append(subject)

    # make sure all elements in subject list are str
    return [flatten(clean) for clean in all_clean_subs]


# %% [markdown]
# ## 1. Network Analysis

# %% [markdown]
# ### 1.1 Prepare data

# %%
# build subject co-occurence matrix


def build_subject_matrix(mission_data, keyword):
    """
    Args:
        mission_data: returned results from calling library API based on keyword.
        keyword: keyword(s) used to return results from calling library api.
    
    Returns:
        Weighted network of subjects. 
    
    """
    # clean subjects
    clean_subjects = clean_subject(
        [
            book["bibliographic_data"]["subject"]
            for book in mission_data
            if "subject" in book["bibliographic_data"].keys()
        ]
    )

    # remove searched keyword(s)
    if isinstance(keyword, str):
        clean_subjects = [
            [sub for sub in subject if sub != keyword]
            for subject in clean_subjects
            if subject
        ]

    elif isinstance(keyword, list):
        clean_subjects = [
            [sub for sub in subject if sub not in keyword]
            for subject in clean_subjects
            if subject
        ]

    # build graph
    clean_subjects = [sub for sub in clean_subjects if sub != []]
    weighted_co_matrix = Counter(
        itertools.chain.from_iterable(
            itertools.combinations(line, 2) for line in clean_subjects
        )
    )

    G = nx.Graph()

    for graph_info in list(weighted_co_matrix.items()):
        nodes = graph_info[0]
        edge_weight = graph_info[1]
        G.add_edge(nodes[0], nodes[1], weight=edge_weight)

    return G


# %%
# build subject matrix using climate change data
G_climate = build_subject_matrix(asf, "heat pump")

# %% [markdown]
# ### 1.1.2 Prune network based on minimum spanning tree
# where optimal network simplification is where the loss of connectivity should be minimized.
# https://core.ac.uk/download/pdf/81624193.pdf

# %%
# minimum spanning tree
G_climate_min_pruned = nx.minimum_spanning_tree(G_climate)


# %% [markdown]
# ## 1.2 Network EDA

# %%
def network_eda(network):
    print(f" the number of nodes in the network is {network.number_of_nodes()}.")
    print(f" the number of edges in the network is {network.number_of_edges()}.")
    print(
        f" there are {len(list(nx.selfloop_edges(network, data=True)))} self loops in the network."
    )

    centrality = nx.eigenvector_centrality(network)
    print(
        f"the graph's average eigenvector centrality value is {statistics.mean(list(centrality.values()))}."
    )
    print(
        f"based on eigenvector centrality, the most central nodes are: {[node for node, centrality_value in centrality.items() if centrality_value in sorted(list(centrality.values()), reverse=True)[:5]]}."
    )
    print("---------------")


# %%
print("ASF")
network_eda(G_climate)

# %%
# plot histogram of clustering coeffients
clustering_coefs = nx.clustering(G_climate)
pd.DataFrame(list(clustering_coefs.values())).plot.hist()

# %% [markdown]
# ## 1.3 visualise data

# %%
# visualise climate subject co-occurence matrix

# remove self loops in graph
G_climate.remove_edges_from(list(nx.selfloop_edges(G_climate)))

g = Network(height=500, width=800, notebook=True)
g.from_nx(G_climate)
g.show("climate.html")

# %% [markdown]
# ### 1.4 run community detection algorithm

# %%
print("FOR A SUSTAINABLE FUTURE")
clusters = community_louvain.best_partition(G_climate)
nx.set_node_attributes(G_climate, clusters, "cluster_group")

dfcommunities = pd.DataFrame(
    [(x[0], x[1]["cluster_group"]) for x in list(G_climate.nodes.data())]
)
dfcommunities.columns = ["subject", "cluster_group"]

print(f"there are {len(list(set(dfcommunities['cluster_group'].tolist())))} clusters.")
print(
    f"the modularity of the clusters is {community_louvain.modularity(clusters, G_climate)}"
)

# look into communities
communities = list(set(dfcommunities["cluster_group"].tolist()))

print(dfcommunities[(dfcommunities["cluster_group"] == random.choice(communities))])
print(dfcommunities[(dfcommunities["cluster_group"] == random.choice(communities))])
print(dfcommunities[(dfcommunities["cluster_group"] == random.choice(communities))])


# %% [markdown]
# ### From the meeting 25-11
#
# * trimming ideas - Cath to talk to Juan
# * Get rid of central node i.e. keyword
# * Query expansion - play around with getting more data
#     * via the most modular node
#     * how much is a duplicate? Make a graph here
# * can you randomly sample the API?
# * call with George about other ideas forward

# %% [markdown]
# ## 2. API sampling
# can randomly-ish sample up to 400 pages of english language publications - otherwise, the API returns a 500 server error.

# %% [markdown]
# ## 2.1 Query expansion

# %%
def expand_node_queries(G, std1=1.5, std2=2.5):
    """
    expands initial seed keyword based on eigenvector centrality.
    """
    # expand node queries based on eigenvector centrality
    centrality = nx.eigenvector_centrality(G)
    ordered_centrality_scores = sorted(centrality.values(), reverse=True)

    # plot distribution of centrality scores
    plt.hist(ordered_centrality_scores, bins=30)

    # don't take too central - take x standard deviations above the mean
    two_stdevs = statistics.mean(ordered_centrality_scores) + std1 * statistics.stdev(
        ordered_centrality_scores
    )
    three_stdevs = statistics.mean(ordered_centrality_scores) + std2 * statistics.stdev(
        ordered_centrality_scores
    )

    plt.axvline(two_stdevs, color="red", linestyle="dashed", linewidth=1)
    plt.axvline(three_stdevs, color="blue", linestyle="dashed", linewidth=1)

    # get list of scores between standard deviations of the mean
    most_central_scores = [
        most_central_score
        for most_central_score in ordered_centrality_scores
        if two_stdevs <= most_central_score <= three_stdevs
    ]

    # get list of expanded node queries
    node_queries = list({k for k, v in centrality.items() if v in most_central_scores})
    print(node_queries)

    expanded_books = []
    for node in node_queries:
        node_data = get_library_data(node)
        expanded_books.extend(node_data)

    return expanded_books, node_queries


# %%
# look at the number of overlapping subject nodes
expanded_books, node_queries = expand_node_queries(G_climate)
expanded_books_graph = build_subject_matrix(expanded_books, node_queries)

# %%
# nodes in both graphs
# print number of nodes in original graph
print(len(list(G_climate.nodes())))
# print number of nodes in expanded node graph
print(len(list(expanded_books_graph.nodes())))
# print the number of nodes in common
print(
    len(
        list(
            set(list(G_climate.nodes())).intersection(
                list(expanded_books_graph.nodes())
            )
        )
    )
)
# print nodes in the expanded graph that isn't present in the original graph
print(set(list(expanded_books_graph.nodes())) ^ set(list(G_climate.nodes())))


# %%
# combine graphs
combined = nx.compose(G_climate, expanded_books_graph)
