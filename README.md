# *CoVaxLies v1*: Automatic Detection of COVID-19 Vaccine Misinformation with Graph Link Prediction

This repository contains the code and annotations utilized in the following research paper:

[Automatic Detection of COVID-19 Vaccine Misinformation with Graph Link Prediction](https://doi.org/10.1016/j.jbi.2021.103955)

Please cite as the following:

```
@article{weinzierl-covid-glp,
	title        = {Automatic detection of COVID-19 vaccine misinformation with graph link prediction},
	author       = {Maxwell A. Weinzierl and Sanda M. Harabagiu},
	year         = 2021,
	journal      = {Journal of Biomedical Informatics},
	volume       = 124,
	pages        = 103955,
	doi          = {https://doi.org/10.1016/j.jbi.2021.103955},
	issn         = {1532-0464},
	url          = {https://www.sciencedirect.com/science/article/pii/S1532046421002847},
	keywords     = {Natural Language Processing, Machine learning, COVID-19, vaccine misinformation, Social Media, knowledge graph embedding}
}
```

## *CoVaxLies* v2 Annotations

We have expanded the dataset as *CoVaxLies* v2 with stance annotations, more misinformation targets, and a taxonomy of misinformation.

[Link](https://github.com/Supermaxman/vaccine-lies/tree/master/covid19)


## *CoVaxLies* v1 Annotations

Annotated tweet ids and misinformation targets for *CoVaxLies* v1 can be found in the annotations folder.
You will need to use the Twitter API to download the text of these tweets, as we cannot directly provide this info.
Stance annotations from *CoVaxLies* v2 have additionally been provided, all tweets annotated with Agree, Disagree, and No Stance
are Relevant.
