"""
Util functions to help build library network and add relevant node and edge attributes.
"""
####
import re
from string import digits
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pattern.text.en import singularize
import networkx as nx
import itertools
from collections import Counter
from datetime import datetime

from typing import List
from pec_library.getters.data_getters import get_library_data

####

KEYWORDS = [
    "heat pump*",
    "solar panel*",
    "home retrofit*",
    "decarbonisation",
    "solar pv",
]


def get_all_library_data(keywords: List) -> List:
    """
    Wrapper function to query Libraryhub's API based on a list of keywords.

    Input:
        keyword (list): list of query terms used to query Libraryhub's API.

    Output:
        all_library_data (list of dicts): query results where each element
        of the list is a dictionary with
        data on bibliographic data, holdings and uri.
    """
    all_library_data = get_library_data(keywords[0])

    for keyword in keywords[1 : len(keywords)]:
        try:
            all_library_data.extend(get_library_data(keyword))
        except TypeError:
            pass

    return all_library_data


def extract_publication_year(all_library_data: List) -> List:
    """
    Regex extract publication year from publication details field.

    Input:
        all_library_data (list): query results where each element
        of the list is a dictionary with
        data on bibliographic data, holdings and uri.

    Output:
        all_library_data (list of dicts): query results where each element
        of the list is a dictionary with
        data on bibliographic data incl. publication year, holdings
        and uri.
    """

    for book in all_library_data:
        if "publication_details" in book["bibliographic_data"].keys():
            book_pub_detail = book["bibliographic_data"]["publication_details"][0]
            potential_years = re.findall(r"\d{4}", book_pub_detail)
            if potential_years != []:
                potential_years = potential_years[0]
                if (
                    potential_years.startswith("18")
                    or potential_years.startswith("19")
                    or potential_years.startswith("2")
                ):
                    potential_years_datetime_format = datetime.strptime(
                        potential_years, "%Y"
                    )
                    if "subject" in book["bibliographic_data"]:
                        book["bibliographic_data"][
                            "publication_year"
                        ] = potential_years_datetime_format.year

    return all_library_data


def clean_subject(subject: List) -> List:
    """
    Args:
        subject (list): list of string subjects to be cleaned.

    Returns:
        subject (list): list of cleaned string subjects.
    """

    lemmatizer = WordNetLemmatizer()
    all_clean_subs = []

    subject = [sub.translate(str.maketrans("", "", ".:()-*")) for sub in subject]
    # remove stopwords
    subject = [sub.lower() for sub in subject if sub not in stopwords.words("english")]
    # remove digits
    no_digits = str.maketrans("", "", digits)
    subject = [sub.translate(no_digits) for sub in subject]
    # trim trailing whitespace
    subject = [re.sub("\s+", " ", sub).strip() for sub in subject]
    # lemmatize subject
    subject = [lemmatizer.lemmatize(sub) for sub in subject]
    # trim short subjects
    subject = [sub for sub in subject if len(sub) > 3]
    # singularise subject
    subject = [singularize(sub) for sub in subject]

    return subject


def build_subject_pair_coo_graph(all_library_data: List, min_edge_weight):
    """
    Builds subject pair cooccurance graph from records with both
    publicaion year and subject list associated to them.

    Input:
        all_library_data (list): query results where each element
        of the list is a dictionary with
        data on bibliographic data incl. publication year, holdings and uri.

    Output:
        G (graph): An undirected, unweighted networkx graph where each
        node is a subject, each edge contains year first published attribute.
    """

    # filter records for records w/ subject AND publication year
    all_library_data = [
        book
        for book in all_library_data
        if "subject" and "publication_year" in book["bibliographic_data"].keys()
    ]

    subjects = [book["bibliographic_data"]["subject"] for book in all_library_data]

    # Get a list of all of subject combinations
    expanded_subjects = itertools.chain(
        *[tuple(itertools.combinations(d, 2)) for d in subjects]
    )

    # Sort and count the combinations so that A,B and B,A are treated the same
    weighted_expanded_subjects = Counter([tuple(sorted(d)) for d in expanded_subjects])

    # remove pairs with edgeweight 1
    weighted_expanded_subjects = Counter(
        subject_pair
        for subject_pair in weighted_expanded_subjects.elements()
        if weighted_expanded_subjects[subject_pair] > min_edge_weight
    )

    # add time attribute to subject pairs
    subject_pair_years = {}
    for subject_pair, weight in weighted_expanded_subjects.items():
        years = []
        for record in all_library_data:
            if (
                subject_pair[0]
                and subject_pair[1] in record["bibliographic_data"]["subject"]
            ):
                years.append(record["bibliographic_data"]["publication_year"])
                subject_pair_years[subject_pair] = {"years published": years,
                                                   "weight": weight}

    # instantiate and populate network
    G = nx.Graph()
    for subject_pair, subject_pair_info in subject_pair_years.items():
        G.add_edge(subject_pair[0],
                  subject_pair[1],
                  first_published=sorted(subject_pair_info["years published"])[0],
                  weight=subject_pair_info['weight'])
    return G
