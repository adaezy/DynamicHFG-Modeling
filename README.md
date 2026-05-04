# DynamicHFG-Modeling
Modeling of utility infrastructures using Dynamic Hetero-Functional Graphs to inform and optimize operational decisions based on community values.

## How to Run the Code

This repository includes a Python application designed to simulate operations based on Dynamic Hetero-Functional Graph Modeling for utility infrastructures. Below are the instructions to run the simulations for both power and water infrastructures.

### Prerequisites

Ensure you have Python installed on your machine. Python 3.7 was used for this work.

### Installation

1. Clone the repository to your local machine and navigate to the project directory:
   ```
   git clone <repository_URL>
   cd <repository_directory>
   ```

2. Install the necessary Python packages using pip:
   ```
   pip install -r requirements.txt
   ```

### Running Power Infrastructure Simulations

Run the simulation for power infrastructure using the command below.
```
python main.py --graph "data/community_hfg_model.graphml" --infra "power" --alpha "sc" --time-steps 15 --mean 0.3 --num-crews 2 --failed-nodes Powerline2 Powerline7 Powerline12 --dns-dicts ElementarySchool1:100 ResidentialBuilding1:200 CommercialBuilding1:500 Hospital1:1000 ResidentialBuilding2:200 ElementarySchool2:100 CommercialBuilding2:500 ResidentialBuilding3:200 ElementarySchool3:100 Hospital2:1000 --sims 1000 --ratio 0.4 0.6
```

### Running Water Infrastructure Simulations

For water infrastructure simulations, use the following command.
```
python main.py --graph "data/community_hfg_model.graphml" --infra "water" --alpha "sc" --time-steps 15 --mean 0.3 --num-crews 1 --failed-nodes WaterPipeline12 WaterPipeline13 --sims 1000 --ratio 0.4 0.6
```



### Computing Kendall's Tau Correlation (Table II)

To compute the correlation between repair orderings:
```
python compute_kendall_tau.py
```

### Additional Information

Graph File: The community_hfg_model.graphml file contains the network structure needed for the simulations. Place it inside the data/ folder.

Parameter Details:

--infra: Specifies the type of infrastructure (power or water).

--alpha: Ordering metric used. s = social vulnerability, c = criticality, sc = weighted combination of both.

--time-steps: Defines the number of time steps in the simulation.

--mean: Rate parameter value used in the transition probability matrix calculation.

--num-crews: Number of crews available for repairs.

--failed-nodes: Nodes that have failed at the start of the simulation.

--dns-dicts: Demand not served for each consumption node in the power simulation. Format is ResourceName:Value (e.g. Hospital1:1000).

--sims: Number of simulation iterations.

--ratio: Weights used for the weighted metric sc. First value is SVS weight, second is criticality weight.
