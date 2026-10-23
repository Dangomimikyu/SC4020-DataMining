# SC4020 assignment 2
## overview
this assignment is in 3 parts:
1. Dataset analysis for the DAVIS AND KIBA datasets
2. Recommend a drug based on a ligand 
3. Cold target generalization

### Part 1: Data description and analysis (30%)
Explore score distributions and observation of the drug-target pairs. Explain the main features and key differences between DAVIS and KIBA.

datasets can be found in *Data/* folder

**column info for both datasets**
* canonical_smiles Isomeric SMILES string representing the compound structure
* target_sequence Amino acid sequence of the protein target
* affinity Binding affinity value (e.g., pKd for Davis, KIBA score for KIBA)


**DAVIS Dataset Structure**
• Ligands and targets: 68 ligands and 442 protein targets.
• Observations: 30,056 ligand–target pairs; the supplied matrix is complete.
• Scores: Dissociation constants Kd in nM. Use pKd for analysis and modelling.

**KIBA Dataset Structure**
• Ligands and targets: 2,111 ligands and 229 protein targets.
• Observations: 118,254 measured pairs in a sparse matrix. Missing entries are not zero scores.
• Scores: Composite KIBA activity scores in the supplied transformed benchmark scale.

**background info from brief**
Drug–target binding: A ligand is a small molecule that may bind to a target protein. One ligand may interact with several targets.
* Binding affinity: Lower Kd indicates stronger binding. Higher pKd indicates stronger binding.
    * E.g., convert in DAVIS using pKd = −log10(Kd / 10⁹) = 9 − log10(Kd), with Kd in nM.
* Composite activity score: KIBA combines information from Ki, Kd and IC50 measurements.
In the supplied version, higher scores indicate stronger affinity.

#### Tasks involved
* Data Cleaning
* Visualisation
    * heatmaps
    * distributions

### Part 2: Recommending drug to given target (40%)
For a ligand represented in training, predict scores and rank targets whose scores are held out. Assess the value of collaborative filtering beyond looking up known measurements.

#### Tasks involved
* Data preprocessing
* Report MSE and CI and Recommendation Quality
* Save models
* Case study using unseen test cases

### Part 3: Cold-target Generalization (30%)
provide solution to CF's inability to generalise to unseen target recommendations

## Deliverables
Code                    - in *Source/*
Report                  - in *Docs/*
Contribution breakdown  - in *Docs/*
Presentation video link - in *Docs/*

Due: 15 nov 2026, 2359H GMT+8