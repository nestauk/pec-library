"""
Util functions to create subgraphs by timeslice, cluster timeslices and propagate
cluster labels across timeslices to make communities comparable across time.
"""
####
import networkx as nx
from typing import Dict
import leidenalg as la
import igraph as ig
import distance
import numpy as np
import random
from collections import Counter
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer

####


def get_tfidf_top_features(documents: list, n_top: int):
    """
    get top n features using tfidf. 
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = tfidf_vectorizer.fit_transform(documents)
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())
    return tfidf_feature_names[importance[:n_top]]


def get_subgraph_clusters(G_timeslice):
    subgraph_cluster = [
        node[1] for node in G_timeslice.nodes(data="timeslice cluster number")
    ]
    return [node[0] for node in Counter(subgraph_cluster).most_common()]


def get_subgraph_cluster_nodes(G_timeslice, cluster):
    return [
        node
        for node, node_info in G_timeslice.nodes(data=True)
        if node_info["timeslice cluster number"] == cluster
    ]


def add_cluster_names(subgraph_communities: dict, n_top:int):
    """
    Generate tf-idf of each cluster at latest time point 
    and assign cluster name across all time slices for consistency.

    Args:
        subgraph_communities (Dict): A dictionary where the keys refer to timeslices
        and the values are undirected networkx subgraphs with timeslice cluster
        group node attribute. 
    Returns:
        subgraph_communities (Dict): A dictionary where the keys refer to timeslices
        and the values are undirected networkx subgraphs with timeslice cluster
        group and cluster name attributes.   
    """
    community_names = dict()
    for subgraph in subgraph_communities.values():
        communities = get_subgraph_clusters(subgraph)
        for community in communities:
            node_names = [
                y["_nx_name"]
                for x, y in subgraph.nodes(data=True)
                if y["timeslice cluster number"] == community
            ]
            community_names[community] = "-".join(get_tfidf_top_features(node_names, 3))

    for subgraph in subgraph_communities.values():
        node_cluster_names = dict(
            zip(
                list(subgraph.nodes()),
                [
                    community_names[subgraph.nodes[n]["timeslice cluster number"]]
                    for n in subgraph.nodes
                ],
            )
        )
        nx.set_node_attributes(subgraph, node_cluster_names, "timeslice cluster name")


def add_cluster_colors(subgraph_communities: dict):
    """Generates 6 digit HEX color codes per cluster per subgraph and appends HEX colors 
    as 'cluster color' node attribute to subgraphs.
    Args:
        subgraph_communities (Dict): A dictionary where the keys refer to timeslices
        and the values are undirected networkx subgraphs with timeslice cluster
        group node attributes. 
    Returns:
           subgraph_communities (Dict): A dictionary where the keys refer to timeslices
        and the values are undirected networkx subgraphs with timeslice cluster
        group and cluster color node attributes.   
    """
    cluster_numbers = [
        list(
            set(
                [
                    node_info["timeslice cluster number"]
                    for node, node_info in subgraph.nodes(data=True)
                ]
            )
        )
        for subgraph in subgraph_communities.values()
    ]
    all_clusters = list(set(itertools.chain(*cluster_numbers)))
    hex_colors = [
        "#%06x" % random.randint(0, 0xFFFFFF) for _ in range(len(all_clusters))
    ]
    cluster_colors = dict(zip(all_clusters, hex_colors))

    for subgraph in subgraph_communities.values():
        node_cluster_colors = dict(
            zip(
                list(subgraph.nodes()),
                [
                    cluster_colors[subgraph.nodes[n]["timeslice cluster number"]]
                    for n in subgraph.nodes
                ],
            )
        )

        nx.set_node_attributes(subgraph, node_cluster_colors, "timeslice cluster color")

    return subgraph_communities


def timeslice_subject_pair_coo_graph(G, timeslice_interval: int) -> Dict:
    """
    Creates timesliced subject-pair co-occurance subgraphs every X year interval.

    Inputs:
        G (graph): subject pair cooccurance graph with time based edge attributes. 
        timeslice_interval (int): timeslice interval in years.    

    Output:
        G_timeslices (dicts): A dictionary where keys refer to timeslices and 
        the values refer to subject pair cooccurance subgraphs. 
    """
    pairs_first_published = [e["first_published"] for u, v, e in G.edges(data=True)]
    min_timeslice, max_timeslice = (
        min(pairs_first_published) + timeslice_interval,
        max(pairs_first_published) + timeslice_interval,
    )

    G_timeslices = dict()
    for i, timeslice in enumerate(
        range(min_timeslice, max_timeslice, timeslice_interval)
    ):
        subgraph_edges = [
            (u, v)
            for u, v, e in G.edges(data=True)
            if e["first_published"] <= timeslice
        ]
        # subgraph induced by specified edges
        G_timeslices["G_timeslice_" + str(i)] = G.edge_subgraph(subgraph_edges)

    return G_timeslices


def cluster_timeslice_subject_pair_coo_graph(G_timeslices: dict):
    """
    Clusters timesliced subject-pair co-occurance subgraphs every X year interval 
    using the leiden algorithm.

    Input:
        G_timeslices (dict): A dictionary where keys refer to timeslices and 
        the values refer to subject pair cooccurance subgraphs. 

    Output:
        modularity (dict): A dictionary where keys refer to timeslices and
        values refer to subgraph partition modularity. 
        subgraph_communities (dict): A dictionary where keys refer to timeslices and 
        the values refer to subject pair cooccurance subgraphs w/ 
        a node cluster attributes. 
    """
    subgraph_communities = dict()
    modularity = dict()

    for timeslice, subgraph in G_timeslices.items():
        subgraph_igraph = ig.Graph.from_networkx(subgraph)
        partitions = la.find_partition(subgraph_igraph, la.ModularityVertexPartition)

        modularity[timeslice] = partitions.quality()

        for node in range(len(subgraph_igraph.vs)):
            subgraph_igraph.vs["cluster number"] = partitions.membership
        subgraph_communities[timeslice] = subgraph_igraph.to_networkx()

    # add timeslice cluster node attribute per subgraph
    for timestamp, subgraph_community in subgraph_communities.items():
        for node in subgraph_community.nodes(data=True):
            node[1]["timeslice cluster number"] = (
                timestamp + "_" + str(node[1]["cluster number"])
            )

    return subgraph_communities, modularity


def sanitise_clusters(timeslice_x, timeslice_y):
    """
    Enforces cluster label consistency across timeslices based on jaccard similarity. 

    Input:
        timeslice_x (Graph): subject pair cooccurance subgraph at timeslice x 
        timeslice_y (Graph): subject pair cooccurance subgraphs at timeslice y (x + 1)
    """
    subgraph_clusters = [
        get_subgraph_clusters(subgraph) for subgraph in (timeslice_x, timeslice_y)
    ]
    cluster_perms = list(itertools.product(subgraph_clusters[0], subgraph_clusters[1]))

    perm_dists = []
    for cluster_perm in cluster_perms:
        timeslice_x_nodes, timeslice_y_nodes = (
            get_subgraph_cluster_nodes(timeslice_x, cluster_perm[0]),
            get_subgraph_cluster_nodes(timeslice_y, cluster_perm[1]),
        )
        dists = 1 - distance.jaccard(timeslice_x_nodes, timeslice_y_nodes)
        if dists != 0:
            perm_dists.append((cluster_perm, dists))

    sorted_perm_dists = sorted(perm_dists, key=lambda x: x[1], reverse=True)
    while len(sorted_perm_dists) > 0:
        most_similar_clusts = sorted_perm_dists[0]
        # update labels in timeslice y
        for node in timeslice_y.nodes(data=True):
            if node[1]["timeslice cluster number"] == most_similar_clusts[0][1]:
                node[1]["timeslice cluster number"] = most_similar_clusts[0][0]

        # remove perms
        clusters_to_remove = list(most_similar_clusts[0])
        for i, perm_dist in enumerate(sorted_perm_dists):
            if (
                perm_dist[0][0] in clusters_to_remove
                or perm_dist[0][1] in clusters_to_remove
            ):
                sorted_perm_dists.pop(i)
