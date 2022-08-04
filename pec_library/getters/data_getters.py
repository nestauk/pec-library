"""
Functions to call library API, loading and saving to s3.
"""
####
import requests
import math
import json
import pickle
import gzip
import os
import logging
import boto3
import random
from typing import List
import pandas as pd
from fnmatch import fnmatch
####

s3 = boto3.resource("s3")

logger = logging.getLogger(__name__)

BASE_REQUEST = "https://discover.libraryhub.jisc.ac.uk/search?subject="
JSON_FORMAT = "&format=json&page="


def get_library_data(keyword: str) -> List:
    """
    Query Libraryhub's API based on keyword.

    Input:
        keyword (str): query term used to query Libraryhub's API.

    Output:
        lib_data (list of dicts): query results where each element
        of the list is a dictionary with
        data on bibliographic data, holdings and uri.

    """
    if len(keyword.split(" ")) > 0:
        request_url = BASE_REQUEST + keyword.replace(" ", "+") + JSON_FORMAT
    else:
        request_url = BASE_REQUEST + keyword + JSON_FORMAT

    response = requests.get(request_url).json()

    if response["hits"] != 0:
        no_pages = math.ceil(response["hits"] / len(response["records"]))
        print(
            f"there are a total of {response['hits']} results and {no_pages} pages of results"
        )
        lib_data = []
        for page in range(1, no_pages + 1):
            all_response_urls = request_url + str(page)
            responses = requests.get(all_response_urls)
            print(f"getting data from page {page}...")
            if responses.status_code == 200:
                lib_response = responses.json()
                if lib_response["records"]:
                    lib_data.extend(lib_response["records"])
                else:
                    logger.warning(f"records not in key!")
            else:
                logger.warning(f"{responses.status_code} response code.")

        for book in lib_data:
            book["bibliographic_data"]["keyword"] = keyword.replace("*", "")

        return lib_data
    else:
        logger.info(f"no results found for {keyword} keyword.")


def get_s3_data_paths(s3, bucket_name, root, file_types=["*.jsonl"]):
    """
    Get all paths to particular file types in a S3 root location
    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    root: The root folder to look for files in
    file_types: List of file types to look for, or one
    """
    if isinstance(file_types, str):
        file_types = [file_types]

    bucket = s3.Bucket(bucket_name)

    s3_keys = []
    for obj in bucket.objects.all():
        key = obj.key
        if root in key:
            if any([fnmatch(key, pattern) for pattern in file_types]):
                s3_keys.append(key)

    return s3_keys


def save_to_s3(s3, bucket_name, output_var, output_file_dir):

    obj = s3.Object(bucket_name, output_file_dir)

    if fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
        byte_obj = pickle.dumps(output_var)
        obj.put(Body=byte_obj)

    logger.info(f"Saved to s3://{bucket_name} + {output_file_dir} ...")


def load_s3_data(s3, bucket_name, file_name):
    """
    Load data from S3 location.
    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read()
        return pickle.loads(file)
    else:
        print(
            'Function not supported for file type other than "*.jsonl.gz", ".pickle", "*.jsonl", or "*.json"'
        )
