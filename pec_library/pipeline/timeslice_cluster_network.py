# %%
"""
Script to create subgraphs by timeslice, cluster timeslices and propagate
cluster labels across timeslices to make communities comparable across time.

In activated conda environment, run python timeslice_cluster_network.py
"""
####
from pec_library.getters.data_getters import (
    s3,
    save_to_s3,
    load_s3_data,
    get_library_data,
)
from pec_library import BUCKET_NAME, CONFIG_PATH, get_yaml_config
from pec_library.pipeline.timeslice_cluster_network_utils import (
    timeslice_subject_pair_coo_graph,
    cluster_timeslice_subject_pair_coo_graph,
    sanitise_clusters,
    add_cluster_colors,
    add_cluster_names,
)

# %% [markdown]
# ###

# %%
if __name__ == "__main__":
    # get config file with relevant paramenters
    config_info = get_yaml_config(CONFIG_PATH)
    G_library_path = config_info["network_path"]
    G_timeslices_path = config_info["network_timeslices_path"]
    timeslice_interval = config_info["timeslice_interval"]
    n_top = config_info["n_top"]

    # get network data
    G_library = load_s3_data(s3, BUCKET_NAME, G_library_path)
    # timeslice network into subgraphs
    G_timeslices = timeslice_subject_pair_coo_graph(G_library, timeslice_interval)
    # cluster subgraphs
    subgraph_communities, modularity = cluster_timeslice_subject_pair_coo_graph(
        G_timeslices
    )

    # sanitise cluster labels for cluster evolution
    for i in range(len(subgraph_communities) - 1):
        timeslice_x = 'G_timeslice_' + str(i)
        timeslice_y = 'G_timeslice_' + str(i + 1)
        sanitise_clusters(subgraph_communities[timeslice_x], subgraph_communities[timeslice_y])
        
    # add cluster name and cluster colors to sanitised timeslices
    subgraph_communities = add_cluster_colors(subgraph_communities)
    subgraph_communities = add_cluster_names(subgraph_communities, n_top)
    # save data
    save_to_s3(s3, BUCKET_NAME, subgraph_communities, G_timeslices_path)
