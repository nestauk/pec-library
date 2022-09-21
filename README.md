# Exploring trends in library catalogue data

This repo contains the code behind the PEC library project on exploring trends in renewable energy in library catalogue data.  

We take a follow six steps to the network based approach to exploring trends in renewable energy, which is summarised in the below chart:

<p align="center">
  <img width="638" alt="Screenshot 2022-09-21 at 10 04 38" src="https://user-images.githubusercontent.com/46863334/191463351-cf4bf54d-9fc2-4dc5-ba54-95a99834d4f6.png">
</p>

By clustering subgraphs over time, we are able to explore subject group dynamics over time, answering interesting questions such as: what can the changes in clusters over time tell us about the renewable energy space? What topic areas have remained popular in renewable energy literature? What new topics have emerged? 

To call [the Library Hub Discover API](https://discover.libraryhub.jisc.ac.uk/support/api/) with a list of keywords, clean up the returned results, generate a network and save to Nesta S3:

`python build_network_flow.py run --min_edge_weight 1 --lib_name asf.pickle --lib_network_name G_library.pickle run`

To timeslice the overall network; cluster each timeslice network and save to Nesta S3: 

`python timeslice_cluster_network_flow.py run --lib_network_name "G_library.pickle" --lib_time "G_timeslices.pickle" --time_interval 10 --min_time 1965 --n_top 3`

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt` and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

If you update the requirements then run `make conda-update`.

## Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
