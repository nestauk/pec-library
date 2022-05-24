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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""Graphs generated for the PEC blogpost."""
########################################################
import itertools
import pigeonXT as pixt
from collections import Counter
import cmasher as cmr
import statistics
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pec_library import config, BUCKET_NAME
from pec_library.pipeline.timeslice_cluster_network_utils import (
    timeslice_subject_pair_coo_graph,
)
from pec_library.getters.data_getters import s3, load_s3_data
import sys

sys.path.append("/Users/india.kerlenesta/Projects/pec_library")


########################################################

# %%
labels = ["science", "technology", "policy", "finance", "other"]

# %% [markdown]
# #### 0. Load Data

# %%
asf_data = load_s3_data(s3, BUCKET_NAME, config["raw_data_path"])
G_timeslices = load_s3_data(s3, BUCKET_NAME, config["G_timeslices_path"])
G_library = load_s3_data(s3, BUCKET_NAME, config["network_path"])
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
            for node in G_timeslices["G_timeslice_8"].nodes(data=True)
        ]
    ).most_common()[5:15]
]
latest_subgraph = G_timeslices["G_timeslice_8"].subgraph(
    [
        node[0]
        for node in G_timeslices["G_timeslice_8"].nodes(data=True)
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
# ##### 2.4 irrigation-solar-energy network in focus

# %%
visualise_cluster(
    G_timeslices,
    config["irrigation_timeslice_x"],
    config["irrigation_timeslice_y"],
    config["irrigation_cluster_name"],
    config["irrigation_graph_name"],
)


# %% [markdown]
# #### 3. Heat Pump Focus Graphs

# %% [markdown]
# ##### 3.1 heat pump focus utils

# %%
def hp_nodes_to_label():
    """Generate DataFrame of target nodes to label."""
    bc_df = []
    for timeslice, subgraph in G_timeslices_not_clustered.items():
        hp_df = pd.DataFrame(
            [
                (timeslice, edge[0], edge[1], edge[2]["weight"])
                for edge in subgraph.edges(data=True)
                if edge[0] == "heat pump"
            ],
            columns=["timeslice", "source node", "target node", "weight"],
        )
        hp_df["prob"] = hp_df["weight"] / sum(hp_df["weight"]) * 100
        bc_df.append(
            hp_df.sort_values("prob", ascending=False)[
                ["timeslice", "source node", "target node", "weight", "prob"]
            ]
        )

    bc_df = pd.concat(bc_df)
    # get rid of self loops
    bc_df = bc_df[bc_df["source node"] != bc_df["target node"]]

    return bc_df


# %%
def label_hp_nodes(labels: list) -> pd.DataFrame:
    """Label target nodes using list of potential labels.

    Args:
        labels (list): List of potential labels to label target nodes with.

    Returns:
        annotations (pd.DataFrame): Labelled target nodes

    """
    bc_df = hp_nodes_to_label()

    annotations = pixt.annotate(
        list(set(bc_df["target node"])),
        options=labels,
        task_type="classification",
        buttons_in_a_row=3,
        reset_buttons_after_click=True,
        include_next=True,
        include_back=True,
    )

    return annotations


# %%
def generate_hp_focus_bc(annotations: pd.DataFrame):
    """Generate HP in focus bar chart.

    Args:
        annotations (pd.DataFrame): labelled target nodes.

    """
    bc_df["target node label"] = bc_df["target node"].map(
        annotations.set_index("example")["label"].T.to_dict()
    )
    bc_df_weightcount = (
        bc_df.groupby(["timeslice", "target node label"])
        .agg({"weight": sum})
        .reset_index()
    )

    bc_df_weightcount = bc_df_weightcount[
        bc_df_weightcount["target node label"] != "other"
    ]

    bc_df_stacked = bc_df_weightcount.pivot("timeslice", "target node label", "weight")

    years = list(
        itertools.chain(
            *[
                [
                    max(
                        (
                            set(
                                [
                                    e[2]["first_published"]
                                    for e in subgraph.edges(data=True)
                                ]
                            )
                        )
                    )
                ]
                for graph_timestamp, subgraph in G_timeslices.items()
            ]
        )
    )
    bc_df_stacked["years"] = [str(x) for x in years]
    bc_df_stacked.at["G_timeslice_8", "years"] = 2025
    bc_df_stacked.index = bc_df_stacked.years
    bc_df_stacked.plot(kind="bar", stacked=True)


# %% [markdown]
# ##### 3.2 heat pump focus bar chart

# %%
annotations = label_hp_nodes(labels)

# %%
generate_hp_focus_bc(annotations)
