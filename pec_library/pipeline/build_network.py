"""
Script to build library network and add relevant time-based edge attributes.

In activated conda environment, run python build_network.py

"""
####
from pec_library.getters.data_getters import s3, save_to_s3, get_library_data
from pec_library import BUCKET_NAME, CONFIG_PATH, get_yaml_config
from pec_library.pipeline.build_network_utils import (
    KEYWORDS,
    get_all_library_data,
    extract_publication_year,
    clean_subject,
    build_subject_pair_coo_graph,
)

####

if __name__ == "__main__":

    # get config file with relevant paramenters
    config_info = get_yaml_config(CONFIG_PATH)
    min_edge_weight = config_info["min_edge_weight"]
    G_library_path = config_info["network_path"]
    raw_data_path = config_info["raw_data_path"]

    # get all library data across all keywords
    all_library_data = get_all_library_data(KEYWORDS)

    # extract publication year from publication details per record
    all_library_data = extract_publication_year(all_library_data)

    # clean subjects
    for book in all_library_data:
        if "subject" in book["bibliographic_data"].keys():
            book["bibliographic_data"]["subject"] = clean_subject(
                book["bibliographic_data"]["subject"]
            )

    # build subject pair cooccurance network w/ clean subjects
    # and extracted publication years
    G_library = build_subject_pair_coo_graph(all_library_data, min_edge_weight)

    # save network to s3
    save_to_s3(s3, BUCKET_NAME, G_library, G_library_path)
    #save raw data to s3
    save_to_s3(s3, BUCKET_NAME, all_library_data, raw_data_path)
    
