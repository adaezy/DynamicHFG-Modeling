# DynamicHFG-Modeling
Modeling of utility infrastructures using Dynamic Hetero-Functional Graphs to inform and optimize operational decisions based on community values.


## How to Run the Code

This repository includes a Python application designed to simulate operations based on Dynamic Hetero-Functional Graph Modeling for utility infrastructures. Below are the instructions to run the simulations for both power and water infrastructures.

### Prerequisites

Ensure you have Python installed on your machine. Python 3.7 was used for this work.

### Installation

#### 1. Clone the Repository:**
   Clone the repository to your local machine and navigate to the project directory:
   ```
   git clone <repository_URL>
   cd <repository_directory>
   ```

#### 2. Install Required Packages:
   Install the necessary Python packages using pip:
   ```
    pip install -r requirements.txt 
   ```

### Simulations
#### 3. Running Power Infrastruture Simulations
  Run the simulation for power infrastructure using the command below. This includes parameters for the graph file, type of infrastructure, and operational settings.
  ```
  python main.py --graph "data/community_hfg_model.graphml" --infra "power" --alpha "sc" --time-steps 15 --mean 0.3 --num-crews 2 --failed-nodes 'Powerline2' 'Powerline7' 'Powerline12' --dns-dicts 100 200 500 1000 200 100 500 200 100 1000 --sims 1 -- 
  ratio 0.4 0.6
  ```

#### 4. Water Infratructure Simulations
For water infrastructure simulations, use the following command. It specifies settings tailored for water system scenarios.
```
  python main.py --graph "data/community_hfg_model.graphml" --infra "water" --alpha "sc" --time-steps 15 --mean 0.3 --num-crews 1 --failed-nodes 'WaterPipeline12' 'WaterPipeline13' --sims 100 --ratio 0.4 0.6
```

### Additional Information
Graph File: The community_hfg_model.graphml file contains the network structure needed for the simulations.
Parameter Details:
--infra: Specifies the type of infrastructure (e.g., power, water).
--alpha: Ordering metrics used (s -social vulnerability, c -criticality, sc- weighted social vulnerability and criticality ).
--time-steps: Defines the number of time steps in the simulation.
--mean: rate parameter value used in transition probability matrix calculation.
--num-crews: Number of crews available for repairs.
--failed-nodes: Nodes that have failed at the start of the simulation.
--dns-dicts: Demand not served for each consumption functionality in power simulation.
--sims: Number of simulation iterations.
--ratio: Ratios used for weighted metric sc.
