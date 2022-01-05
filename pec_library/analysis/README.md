# PEC EDA

There are two scripts that explore the state of the data using [Library Hub Discover](https://discover.libraryhub.jisc.ac.uk/advanced-search/):

1. `pec_eda.py`
2. `pec_eda_networks.py`

Both scripts rely on the following keywords as the initial 'seed' keywords to query the API:

`asf = "heat pumps"`
`ahl = "obesity"`
`afs = "early childhood"`

The first script reports on general data quality across all three missions, while the second uses `asf_keyword` for network eda. 

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
![asf_2](/eda_results/asf_2.png?raw=true)

### 'obesity'

![ahl_1](/eda_results/ahl_1.png?raw=true)
![ahl_2](/eda_results/ahl_2.png?raw=true)

### 'early childhood'

![afs_1](/eda_results/afs_1.png?raw=true)
![afs_2](/eda_results/afs_2.png?raw=true)

Some key findings from the initial overview of the data include:

1. *Data is relatively incomplete.* However... 
2. *Title, Subject and Publication Details relatively complete.* 
3. *Similar holding institutions across Missions.* Top holding institutions tend to be large libraries with diverse collections.

## pec_eda_networks.py

*Some experiments*
1. taking a random sample of the API -> *API issues.* can randomly-ish sample up to 400 pages of english language publications - otherwise, the API returns a 500 server error.

2. keyword query expansion based on eigenvector centrality  
3. node attribute ideas - language of subject node, entity resolution of node 

## Questions for George

0. a general sense check
	- does this seem interesting? 
1. Which possible next steps sound most promising? 
	- analysing subject clusters over time
		- when where subject clusters first introduced? Do subject cluster become more or less 'dense' over time?  
	- adding and analysing node attributes
		- what additional 'metadata' can we include about subject nodes beyond? i.e. subject langauage, entity type. What can we do with that information that might be interesting? 
2. More specifically, how best to capture time?
	- each materal has a list of subjects associated to it and publication year. We want to assign a year to each subject node by finding the 'average age' of the subject. How dubious does this sound? Other ways of doing so? 
