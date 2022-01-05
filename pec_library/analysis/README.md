# PEC EDA

There are two scripts that explore the state of the data using [Library Hub Discover](https://discover.libraryhub.jisc.ac.uk/advanced-search/):

1. `pec_eda.py`
2. `pec_eda_networks.py`

Both scripts rely on the following keywords as the initial 'seed' keywords to query the API:

`asf = "heat pumps"`
`ahl = "obesity"`
`afs = "early childhood"`

The first script reports on general data quality across all three missions, while the second uses `asf_keyword` for network eda.

When you query the API by keyword, the search will query the whole entry, not a specificed field. When you query the API by keyword, the data looks like:

`{'bibliographic_data': {'author': ['Neal, L. G.'], 'physical_description': ['ix, 227 p. :'], 'publication_details': ['Washington, D.C. : National Aeronautics and Space Administration ; Springfield, Va. : For sale by the National Technical Information Service [distributor], 1971.'], 'subject': ['Nuclear electric power generation.', 'Heat pipes.', 'Feasibility.', 'Rankine cycle.', 'Heat radiators.', 'Heat pumps.', 'Electric generators.', 'Capillary flow.', 'Electric power production.', 'Heat pipes.', 'Rankine cycle.', 'Heat Transmission.', 'Heat pumps.', 'Electric generators.'], 'title': ['Study of a heat rejection system using capillary pumping / L.G. Neal, D.J. Wanous, and O.W. Clausen.'], 'url': ['http://hdl.handle.net/2027/uiug.30112106857045']}, 'holdings': [{'document_type': ['book'], 'held_at': [{'institution': {'institution_id': 'hat', 'name': 'Hathi Trust Digital Library'}}], 'item_id': 230169895, 'local_id': '011429427', 'physical_format': ['online']}], 'uri': 'https://discover.libraryhub.jisc.ac.uk/search?id=230169895&rn=1'}`

## pec_eda.py

The first script aims to get a high level overview of the state of the data and answers the following:

1. What is the state of missing data?
2. How many subjects are there on average per book?
3. What are the most common subjects? How many different subjects?
4. How many books are published over time?
5. Holdings EDA - document types count, how many institutions on average?

The results are as follows:

### 'heat pumps'

![asf_1](eda_results/asf_1.png?raw=true)
![asf_2](eda_results/asf_2.png?raw=true)

### 'obesity'

![ahl_1](eda_results/ahl_1.png?raw=true)
![ahl_2](eda_results/ahl_2.png?raw=true)

### 'early childhood'

![afs_1](eda_results/afs_1.png?raw=true)
![afs_2](eda_results/afs_2.png?raw=true)

(NOTE: I quickly pulled years from publication details so the graphs might have some funny min. and max years.)

Some key findings from the initial overview of the data include:

1. **Data is relatively incomplete.** However...
2. **Title, Subject and Publication Details relatively complete.**
3. **Similar holding institutions across Missions.** Top holding institutions tend to be large libraries with diverse collections.

## pec_eda_networks.py

As a result of the initial general data quality check, we decided to focus on using data that was complete. Namely, each material's subject list and publication details.

In this script, we build and explore undirected, weighted networks of subjects where each node is a subject and each edge represents subject co-occurence in subject lists. You can take a look at the `asf_keyword` subject network by downloading and opening `climate.html` on your browser.

We also played around with:

1. preprocessing subject lists
2. running an initial community detection algorithm to explore subject clusters
3. expanding the initial network by keyword query expansion based on eigenvector centrality
4. taking a random sample of the API -> **API issues.** can randomly-ish sample up to 400 pages of the API - otherwise, the API returns a 500 server error.

## Questions for George

**Goal:** Ultimately, we _think_ want to cluster networks of subjects across the missions over time and analyse them i.e. when were subject clusters introduced? Do subject clusters become more or less 'dense' over time? What does that mean about how publications have been talking about these mission areas?

My main questions/themes for you are:

0. A general sense check of the above goal. Do you think this seems like an interesting goal? Do you think there are more interesting questions to ask of the networks/data?

1. Ways of expanding the network beyond data derived from the initial 'seed' keyword - we played around with taking the most central subject nodes from the initial graph and using them as keywords to query the API. Any other ideas? Do you think its even necessary to expand the networks?

2. How best to approach the time component of the question? Each material has a list of subjects associated to it and publication details incl. year. We want to assign a year to each subject node by finding the 'average age' of the subject. How dubious does this sound? Other ways of doing so?

3. Do you think its worth adding and analysing node attributes? What additional 'metadata' can we include about subject nodes beyond? i.e. subject langauage, entity type. What can we do with that information that might be interesting?
