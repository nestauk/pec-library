"""
Util functions to help build library network and add relevant node and edge attributes.
"""
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
from toolz import pipe

from pec_library.getters.data_getters import get_library_data, s3, load_s3_data
from pec_library import bucket_name

KEYWORDS = [
    "heat pump*",
    "solar panel*",
    "solar energy",
    "home retrofit*",
    "decarbonisation",
    "solar pv",
]

UPDATED_RENEWABLE_ENERGY = load_s3_data(s3, bucket_name, "outputs/all_renewable_energy_deduped.pickle")
    
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
    print(f"the total number of results is {len(all_library_data)}.")
    # load updated renewable energy data
    # extend it with renewable energy
    all_library_data.extend(UPDATED_RENEWABLE_ENERGY)
    print(
        f"after adding renewable energy, the total number of results is {len(all_library_data)}."
    )

    # deduplicate based on book title name
    all_library_data = [book["bibliographic_data"] for book in all_library_data]
    all_library_data_deduped = [
        list(grp)[0]
        for _, grp in itertools.groupby(all_library_data, lambda d: d["title"])
    ]
    print(
        f"the total number of deduped results based on book title is {len(all_library_data_deduped)}."
    )
    # only return results with publication year as an attribute
    return [book for book in all_library_data if "publication_year" in book.keys()]

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

def clean_library_data(library_data):
    """Cleaning library data pipeline. The function:
            - only keeping records with key fields;
            - removing duplicate items;
            - extracting publication year;
            - cleaning subject lists;
            - keeping records with keywords in subjects
    """
    def keep_records_with_key_fields(library_data): 
        """Keep records with key fields used for analysis."""
        key_fields = ['title', 'subject', 'publication_details']
        library_data_with_key_fields = []
        for item in library_data:
            if len(set(item.keys()).intersection(key_fields)) == len(key_fields):
                library_data_with_key_fields.append(item)
        return library_data_with_key_fields
    def remove_duplicate_items(library_data):
        """De-duplicate items in library data."""
        deduplicated_library_data = []
        for item in library_data:
            if item not in deduplicated_library_data:
                deduplicated_library_data.append(item)
        return deduplicated_library_data
    def clean_subject_list(library_data: list):
        """Cleans subject lists in library data."""
        for book in library_data:
            book["subject"] = clean_subject(book["subject"])
        return library_data
    def extract_publication_year(library_data):
        """
        Regex extract publication year from publication details field.
        Input:
            library_data (list): query results where each element
            of the list is a dictionary with
            data on bibliographic data, holdings and uri.
        Output:
            all_library_data (list of dicts): query results where each element
            of the list is a dictionary with
            data on bibliographic data incl. publication year, holdings
            and uri.
        """
        for book in library_data:
            book_pub_detail = book["publication_details"][0]
            potential_years = re.findall(r"\d{4}", book_pub_detail)
            if potential_years != []:
                years = [
                    year
                    for year in potential_years
                    if year.startswith("18")
                    or year.startswith("19")
                    or year.startswith("20")
                ]
                if len(years) > 1:
                    # take earliest year for publication_year
                    potential_years_datetime_format = datetime.strptime(
                        min(years), "%Y"
                    )
                else:
                    potential_years_datetime_format = datetime.strptime(years[0], "%Y")
                book["publication_year"] = potential_years_datetime_format.year
        
        return [book for book in library_data if "publication_year" in book.keys()]
        
    def keep_records_with_keyword_in_subject(library_data):
        """Keep records that contain at least one keyword in subject list."""
        library_data_with_keywords = []
        keywords_clean = [keyword.replace('*', '') for keyword in KEYWORDS + ["renewable energy"]]
        keyword_pattern = '|'.join(f"\\b{k}\\b" for k in keywords_clean)
        for book in library_data:
            if re.findall(keyword_pattern, ' '.join(book['subject'])):
                library_data_with_keywords.append(book)
        return library_data_with_keywords

    return pipe(library_data, keep_records_with_key_fields, remove_duplicate_items, extract_publication_year, clean_subject_list, keep_records_with_keyword_in_subject)

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

    # filter records for records w/ subject
    print(f"the number of records w/ publication year is: {len(all_library_data)}")
    all_library_data = [book for book in all_library_data if "subject" in book.keys()]
    print(
        f"the number of records w/ publication year AND subjects is: {len(all_library_data)}"
    )
    subjects = [book["subject"] for book in all_library_data]

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
            if subject_pair[0] and subject_pair[1] in record["subject"]:
                years.append(record["publication_year"])
                subject_pair_years[subject_pair] = {
                    "years published": years,
                    "weight": weight,
                }

    # instantiate and populate network
    G = nx.Graph()
    for subject_pair, subject_pair_info in subject_pair_years.items():
        G.add_edge(
            subject_pair[0],
            subject_pair[1],
            first_published=sorted(subject_pair_info["years published"])[0],
            weight=subject_pair_info["weight"],
        )

    # whole network
    return G
