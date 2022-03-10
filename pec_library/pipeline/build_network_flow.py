"""
To run flow:

python build_network_flow.py run --min_edge_weight 1 --lib_name asf.pickle --lib_network_name G_library.pickle
"""

####
from metaflow import (FlowSpec, 
    step, 
    project, 
    Parameter, 
    S3
    )

import pickle
####

@project(name="pec_library")
class BuildNetwork(FlowSpec):
    """Pulls data from Library Hub API, preprocesses subject lists and extracts
    publication year from publication detail. Builds subject pair cooccurance network. 

    Attributes:
        mind_edge_weight: the minimum edge weight to include subject pair in graph
        library_data_name: file name to save library data to. 
        libray_network_name: file name to save library network to. 
    """
    min_edge_weight: int
    library_data_name: str
    library_network_name: str

    min_edge_weight = Parameter(
        "min_edge_weight",
        help="the minimum edge weight to include subject pair in graph.",
        type=int,
        default=1
        )

    library_data_name = Parameter(
        "lib_name",
        help="file name to store cleaned library data in s3.",
        type=str,
        default="asf.pickle"
        )

    library_network_name = Parameter(
        "lib_network_name",
        help="file name to store subject pair cooccurance network in s3.",
        type=str,
        default="G_library.pickle"
        )

    @step
    def start(self):
        """Query Library Hub API with predefined keywords."""
        from pec_library.getters.data_getters import get_library_data
        from pec_library.pipeline.build_network_utils import (KEYWORDS,
            get_all_library_data)

        self.library_data = get_all_library_data(KEYWORDS)

        self.next(self.clean_data)

    @step
    def clean_data(self):
        """Clean subject list and extract publication year from
        publication details. 
        """
        from pec_library.pipeline.build_network_utils import (
            extract_publication_year,
            clean_subject)

        self.library_data = extract_publication_year(self.library_data)

        for book in self.library_data:
            if "subject" in book.keys():
                book["subject"] = clean_subject(book["subject"])

        print("cleaned subject list and extracted publication year!")

        self.next(self.build_network)

    @step
    def build_network(self):
        """Build subject pair cooccurance network from cleaned 
        library data."""
        from pec_library.pipeline.build_network_utils import (
            build_subject_pair_coo_graph,
            )

        self.library = build_subject_pair_coo_graph(self.library_data, self.min_edge_weight)

        print("generated subject pair coo graph!")

        self.next(self.end)

    @step
    def end(self):
        from pec_library import BUCKET_NAME
        """Save both cleaned library data and subject pair coocurance network
        to s3."""
        with S3(s3root='s3://' + BUCKET_NAME + '/outputs/') as s3:
            library_byte_obj = pickle.dumps(self.library_data)
            network_byte_obj = pickle.dumps(self.library)

            s3.put(self.library_data_name, library_byte_obj)
            print(f"successfully saved library data to {'s3://' + BUCKET_NAME + '/outputs/' + self.library_data_name}")
            s3.put(self.library_network_name, network_byte_obj)
            print(f"successfully saved library data to {'s3://' + BUCKET_NAME + '/outputs/' + self.library_network_name}")
            
if __name__ == '__main__':
    BuildNetwork()