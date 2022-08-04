# %%
"""
To run flow:

python timeslice_cluster_network_flow.py run --lib_network_name "G_library.pickle" --lib_time "G_timeslices.pickle" --time_interval 10 --min_time 1945 --n_top 3
"""

# %%
####
from metaflow import FlowSpec, step, project, Parameter, S3, pip

# %%
import json
import pickle

# %% [markdown]
# ###


# %%
@project(name="pec_library")
class TimesliceClusterNetwork(FlowSpec):
    """Generates library network subgraphs based on time intervals.
    "Sanitises" clusters by propogating cluster labels across time
    intervals based on jaccard similarity at time interval t and
    time interval t+1. Clusters, names and colors subgraph clusters.

    Attributes:
    library_network_name: library network file name.
    timeslice_interval: the time interval to timeslice the library network
                        into subgraphs.
    n_top: top n tf-idf words to name clusters.
    """

    library_network_name: str
    library_timeslices: str
    timeslice_interval: int
    min_timeslice: int
    n_top: int

    library_network_name = Parameter(
        "lib_network_name",
        help="file name to store subject pair cooccurance network in s3.",
        type=str,
        default="G_library.pickle",
    )

    library_timeslices = Parameter(
        "lib_time",
        help="file name to store a dictionary of timesliced, clustered subgraphs in s3.",
        type=str,
        default="G_timeslices.pickle",
    )

    timeslice_interval = Parameter(
        "time_interval",
        help="the time interval to timeslice the library network into subgraphs.",
        type=int,
        default=10,
    )

    min_timeslice = Parameter(
        "min_time",
        help="time minimum year to timeslice library network into subgraphs.",
        type=int,
        default=1945,  # parameter defined by distribution of years across network
    )

    n_top = Parameter(
        "n_top", help="top n tf-idf words to name clusters.", type=int, default=3
    )

    @step
    def start(self):
        """Load library data from s3."""
        from pec_library import bucket_name

        with S3(s3root="s3://" + bucket_name + "/outputs/") as s3:
            library_network_obj = s3.get(self.library_network_name)
            self.library_network = pickle.loads(library_network_obj.blob)

        print(
            f"successfully loaded library data from {'s3://' + bucket_name + '/outputs/' + self.library_network_name}"
        )

        self.next(self.cluster_timeslice_network)

    @step
    def cluster_timeslice_network(self):
        """slice network into subgraphs based on t time interval.
        Cluster every subgraph using leiden algorithm."""
        from pec_library.pipeline.timeslice_cluster_network_utils import (
            timeslice_subject_pair_coo_graph,
            cluster_timeslice_subject_pair_coo_graph,
        )

        self.timeslices = timeslice_subject_pair_coo_graph(
            self.library_network, self.timeslice_interval, self.min_timeslice
        )
        self.subgraph_communities = cluster_timeslice_subject_pair_coo_graph(
            self.timeslices
        )

        print("timesliced network and clustered subgraphs!")

        self.next(self.sanitise_clusters)

    @step
    def sanitise_clusters(self):
        """Propogate cluster labels greedily across time intervals
        based on jaccard similarity at timeslice t and timeslice t + 1."""
        from pec_library.pipeline.timeslice_cluster_network_utils import (
            sanitise_clusters,
        )

        for i in range(len(self.subgraph_communities) - 1):
            timeslice_x = "G_timeslice_" + str(i)
            timeslice_y = "G_timeslice_" + str(i + 1)
            sanitise_clusters(
                self.subgraph_communities[timeslice_x],
                self.subgraph_communities[timeslice_y],
            )

        print("sanitised clusters!")

        self.next(self.color_name_clusters)

    @step
    def color_name_clusters(self):
        """generate cluster name and color using tf-idf at latest time interval
        per cluster. Propogate cluster name and color across time intervals."""
        from pec_library.pipeline.timeslice_cluster_network_utils import (
            add_cluster_colors,
            add_cluster_names,
        )

        self.subgraph_communities = add_cluster_colors(self.subgraph_communities)
        self.subgraph_communities = add_cluster_names(
            self.subgraph_communities, self.n_top
        )

        print("added cluster name and cluster color as node attributes!")

        self.next(self.end)

    @step
    def end(self):
        """Saves dictionary of clustered, named subgraphs based on timeslices
        to s3."""
        from pec_library import bucket_name

        with S3(s3root="s3://" + bucket_name + "/outputs/") as s3:
            timeslice_byte_obj = pickle.dumps(self.subgraph_communities)
            s3.put(self.library_timeslices, timeslice_byte_obj)

        print(
            f"successfully saved library data to {'s3://' + bucket_name + '/outputs/' + self.library_timeslices}"
        )


# %%
if __name__ == "__main__":
    TimesliceClusterNetwork()
