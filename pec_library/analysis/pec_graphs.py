# %%
"""Graphs generated for the PEC blogpost."""

import itertools
import pigeonXT as pixt
from collections import Counter
import cmasher as cmr
import statistics
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pec_library import config, bucket_name, PROJECT_DIR
from pec_library.pipeline.timeslice_cluster_network_utils import (
    timeslice_subject_pair_coo_graph,
)
from pec_library.getters.data_getters import s3, load_s3_data

# %% [markdown]
# #### 0. Load Data

# %%
asf_data = load_s3_data(s3, bucket_name, config["raw_data_path"])
G_timeslices = load_s3_data(s3, bucket_name, config["G_timeslices_path"])
G_library = load_s3_data(s3, bucket_name, config["network_path"])
G_timeslices_not_clustered = timeslice_subject_pair_coo_graph(
    G_library, config["timeslice_interval"], 1945
)

# %% [markdown]
# #### 1. Summary Graphs

# %% [markdown]
# ###### 1.1 Line graph of the number of publications per year

# %%
keywords_df = pd.DataFrame([book for book in asf_data])
# aggregate general terms
keyword_area_dict = {}
for keyword in list(set(keywords_df["keyword"])):
    if "solar" in keyword:
        keyword_area_dict[keyword] = "solar energy"
    elif keyword == "heat pump" or keyword == "home retrofit":
        keyword_area_dict[keyword] = "heat pump"
    else:
        keyword_area_dict[keyword] = "general"

keywords_df["keyword_area"] = keywords_df["keyword"].map(keyword_area_dict)
# plot number of publications per year per query term
keywords_df_counts = keywords_df[["keyword_area", "publication_year"]].value_counts()
query_terms = list(set(list(keywords_df["keyword_area"])))

for query_term in query_terms:
    ax = keywords_df_counts[query_term].sort_index().plot()
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(query_terms)
    plt.xlim([1960, 2020])
    ax.set_xlabel("Publication Year", weight="bold")
    ax.set_ylabel("Publication Count")


# %% [markdown]
# #### 2. Network Graphs

# %% [markdown]
# ##### 2.0 Network utils

# %%
def update_node_color(
    network,
    color_map="viridis",
    timeslice_cluster_color="timeslice cluster color colormap",
):
    """Map original node cluster color to color map.
    network: Graph to update node color.
    color_map (str): color map to choose from
    """
    current_colors = set(
        [x[1] for x in list(network.nodes(data="timeslice cluster color"))]
    )
    colors = cmr.take_cmap_colors(
        color_map, len(current_colors), cmap_range=(0, 1), return_fmt="hex"
    )
    new_colors_dict = dict(zip(current_colors, colors))

    # add new colormap feature
    node_new_colors = dict(
        zip(
            list(network.nodes()),
            [
                new_colors_dict[network.nodes[n]["timeslice cluster color"]]
                for n in network.nodes
            ],
        )
    )
    nx.set_node_attributes(network, node_new_colors, timeslice_cluster_color)


def visualise_network(network, graph_name: str, timeslice_cluster_color: str):
    """Visualise whole network.
    Args:
        network: Graph to visualise.
        graph_name (str): The name used to save graph.
        timeslice_cluster_color (str): color node attribute.
    """

    g = Network(height="750px", width="100%", font_color="black")
    g.add_nodes(
        [node[0] for node in network.nodes(data=True)],
        label=[node[1]["_nx_name"] for node in network.nodes(data=True)],
        color=[node[1][timeslice_cluster_color] for node in network.nodes(data=True)],
        size=[20 for _ in network.nodes(data=True)],
    )
    g.add_edges([(x[0], x[1]) for x in network.edges(data=True)])

    g.save_graph(graph_name)


# %%
def visualise_cluster(
    G_timeslices,
    timeslice_x: str,
    timeslice_y: str,
    cluster_name: str,
    graph_name: str,
    multiplier=12,
    big_node_degree=20,
):
    """Visualise cluster at two different timestamps where
    new nodes to the cluster are identified in red.
    Node size is determined by logged node degree.
    Args:
        G_timeslices: Dictionary of networks where key refers to timestamp
                      and value is the network at each timestamp.
        timeslice_x (str): Timestamp at earlier time.
        timeslice_y (str): Timestamp at later time.
        cluster_name (str): Name of cluster to generate subgraph for.
        graph_name (str): The name used to save graph.
        multiplier (int): Number to multiply logged node degree by.
        big_node_degree (int): The logged node degree threshold for showing
                               node names.
    """

    subgraph1 = G_timeslices[timeslice_x].subgraph(
        [
            node[0]
            for node in G_timeslices[timeslice_x].nodes(data=True)
            if node[1]["timeslice cluster name"] == cluster_name
        ]
    )
    subgraph2 = G_timeslices[timeslice_y].subgraph(
        [
            node[0]
            for node in G_timeslices[timeslice_y].nodes(data=True)
            if node[1]["timeslice cluster name"] == cluster_name
        ]
    )

    # identify new nodes at later timestamp
    new_nodes = set([node[1]["_nx_name"] for node in subgraph2.nodes(data=True)]) - set(
        [node[1]["_nx_name"] for node in subgraph1.nodes(data=True)]
    )

    # change new nodes color to red
    for node in subgraph2.nodes(data=True):
        if node[1]["_nx_name"] in new_nodes:
            node[1]["timeslice cluster color"] = "#FF0000"  # change colors
        else:
            color = node[1]["timeslice cluster color"]

    # change node size
    degrees = [
        statistics.log(subgraph2.degree[node] + 1) * multiplier
        for node in subgraph2.nodes()
    ]
    big_nodes_indx = [i for i, d in enumerate(degrees) if d > big_node_degree]
    nx.set_node_attributes(subgraph2, degrees, "size")
    
    # only label high degree nodes
    nodes_to_label = [list(subgraph2.nodes(data=True))[i] for i in big_nodes_indx]
    nodes_to_label = [x[1]["_nx_name"] for x in nodes_to_label]

    # instantiate network
    g = Network(height="750px", width="100%", font_color="black")
    g.add_nodes(
        [node[0] for node in subgraph2.nodes(data=True)],
        label=[node[1]["_nx_name"] for node in subgraph2.nodes(data=True)],
        color=[
            node[1]["timeslice cluster color"] for node in subgraph2.nodes(data=True)
        ],
        size=degrees,
    )
    g.add_edges([(x[0], x[1]) for x in subgraph2.edges(data=True)])

    # visualise climate subject co-occurence matrix
    g.save_graph(graph_name)


# %% [markdown]
# ##### 2.1 Network at earliest timestamp

# %%
update_node_color(G_timeslices["G_timeslice_0"])
visualise_network(
    G_timeslices["G_timeslice_0"],
    config["earliest_timeslice_graph_name"],
    "timeslice cluster color colormap",
)

# %% [markdown]
# ##### 2.2 Subgraph of network at latest timestamp

# %%
# subset latest timestamp based on cluster size
cluster_names = [
    x[0]
    for x in Counter(
        [
            node[1]["timeslice cluster name"]
            for node in G_timeslices["G_timeslice_6"].nodes(data=True)
        ]
    ).most_common()[5:15]
]
latest_subgraph = G_timeslices["G_timeslice_6"].subgraph(
    [
        node[0]
        for node in G_timeslices["G_timeslice_6"].nodes(data=True)
        if node[1]["timeslice cluster name"] in cluster_names
    ]
)

update_node_color(latest_subgraph)
visualise_network(
    latest_subgraph,
    config["latest_timeslice_graph_name"],
    "timeslice cluster color colormap",
)

# %% [markdown]
# ##### 2.3 congress-energy-power network in focus

# %%
visualise_cluster(
    G_timeslices,
    config["congress_timeslice_x"],
    config["congress_timeslice_y"],
    config["congress_cluster_name"],
    config["congress_graph_name"],
)

# %% [markdown]
# ##### 2.4 great-britain-energy network in focus

# %%
visualise_cluster(
    G_timeslices,
    config["gb_timeslice_x"],
    config["gb_timeslice_y"],
    config["gb_cluster_name"],
    config["gb_graph_name"],
)


# %% [markdown]
# #### 3. Heat Pump Focus Graphs

# %% [markdown]
# ##### 3.1 ego graph utils

# %%
def generate_egograph(timeslice_network, node_name: str = 'heat pump'):
    """Generate heat pump ego graph at different network timeslices."""
    node = [x for x,y in timeslice_network.nodes(data=True) if y['_nx_name'] == node_name][0]
    ego = nx.ego_graph(timeslice_network, node)
    print(f"the number of edges in the ego-network is {ego.number_of_edges()}.")
    print(f"the number of nodes in the ego-network is {ego.number_of_nodes()}.")
    
    nodes = [x for x,y in ego.nodes(data=True)]
    colors = [y['timeslice cluster color'] for x,y in ego.nodes(data=True)]
    print(Counter([y['timeslice cluster name'] for x,y in ego.nodes(data=True)]))
    print(len(Counter([y['timeslice cluster number'] for x,y in ego.nodes(data=True)])))

    pos = nx.spring_layout(ego, seed=42)
    
    ec = nx.draw_networkx_edges(ego, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(ego, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)

    # Draw ego as large and red
    options = {"node_size": 300, "node_color": "r"}
    nx.draw_networkx_nodes(ego, pos, nodelist=[node], **options)
    
    return ego


# %% [markdown]
# #### 3.2 hp ego graphs

# %%
generate_egograph(timeslice_network=G_timeslices['G_timeslice_0'])
generate_egograph(timeslice_network=G_timeslices['G_timeslice_6'] )
