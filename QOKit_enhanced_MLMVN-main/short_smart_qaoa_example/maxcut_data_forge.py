# Smart QAOA short example by ⚛️ Sigma PublishinQ Team ⚛️  
# https://www.linkedin.com/company/sigma-publishinq/about/

import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/short_smart_qaoa_example/')
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/')
from erdos_renyi_fabric import ErdosRenyiFabric
from qaoa_maxcut_engine import QAOAMaxCutEngine

def get_iteration_count(opt_result, optimizer_name):
    """
    Extracts the number of iterations from the optimization result.
    Different optimizers return this information in different fields.
    """
    # Try different fields depending on the optimizer
    if optimizer_name == 'BFGS':
        if hasattr(opt_result, 'nit') and opt_result.nit is not None:
            return opt_result.nit
        elif hasattr(opt_result, 'nfev') and opt_result.nfev is not None:
            return opt_result.nfev
    elif optimizer_name == 'COBYLA':
        if hasattr(opt_result, 'nfev') and opt_result.nfev is not None:
            return opt_result.nfev
    elif optimizer_name == 'L-BFGS-B':
        if hasattr(opt_result, 'nit') and opt_result.nit is not None:
            return opt_result.nit
        elif hasattr(opt_result, 'nfev') and opt_result.nfev is not None:
            return opt_result.nfev
    elif optimizer_name == 'Nelder-Mead':
        if hasattr(opt_result, 'nit') and opt_result.nit is not None:
            return opt_result.nit
        elif hasattr(opt_result, 'nfev') and opt_result.nfev is not None:
            return opt_result.nfev
    elif optimizer_name == 'SLSQP':
        if hasattr(opt_result, 'nit') and opt_result.nit is not None:
            return opt_result.nit
        elif hasattr(opt_result, 'nfev') and opt_result.nfev is not None:
            return opt_result.nfev
    
    # If nothing found, try common fields
    for attr in ['nit', 'nfev', 'njev', 'nhev']:
        if hasattr(opt_result, attr):
            value = getattr(opt_result, attr)
            if value is not None:
                return value
    
    # If still not found, print warning and return 1
    print(f"⚠️ Failed to extract iteration count for {optimizer_name}")
    print(f"Available attributes: {[attr for attr in dir(opt_result) if not attr.startswith('_')]}")
    return 1

parser = argparse.ArgumentParser(
    description="Generates random Erdős–Rényi graphs and optimizes QAOA for MaxCut problem")

parser.add_argument("-n", "--number_of_nodes", help="number of nodes", type=int)
parser.add_argument("-p", "--number_of_layers", help="number of layers in quantum circuit", type=int)
parser.add_argument("-data", "--dataset_size", help="number of graphs for data collection", type=int, default=5000)
parser.add_argument("-prob", "--edge_probabilities", 
                    help="edge creation probabilities U(low,high), e.g.: 0.3 0.9", 
                    nargs='+', type=float, default=[0.5, 0.5])
parser.add_argument("-weighted", "--weighted_graph", 
                    help="use weighted edges", default=False, action='store_true')
parser.add_argument("-optimizer", "--optimizer", 
                    help="optimizer: 'COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'", 
                    type=str, default='BFGS')
parser.add_argument("-bounds", "--with_bounds", 
                    help="add parameter constraints", default=False, action='store_true')

args = parser.parse_known_args()[0]

# Extract parameters
nodes_num = args.number_of_nodes
p = args.number_of_layers
dataset_size = args.dataset_size
prob = tuple(args.edge_probabilities)
weighted_bool = args.weighted_graph
optimizer = args.optimizer
bounds_bool = args.with_bounds

print(f"n = {nodes_num}")
print(f"p = {p}")
print(f"dataset size = {dataset_size}")
print(f"edge probability: {prob}")
print(f"weighted graph: {weighted_bool}")
print(f"optimizer type: {optimizer}")
print(f"add constraints: {bounds_bool}")

# Check parameter correctness
assert (0.0 <= prob[0] <= 1.0) and (0.0 <= prob[1] <= 1.0) and (prob[0] <= prob[1]), \
    'Probabilities must be between 0 and 1'
assert optimizer in ['COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'], \
    'Optimizer must be from suggested list'

# Calculate number of possible edges
num_possible_edges = int(nodes_num * (nodes_num - 1) / 2)
bounds = [(0, np.pi)] * p + [(0, 2 * np.pi)] * p

assert dataset_size <= 2**num_possible_edges, \
    "Dataset size cannot be larger than possible configurations"

# Create columns for DataFrame
adj_vec_cols = [f'x({i},{j})' for i in range(nodes_num) for j in range(i + 1, nodes_num)]
# Parameters in QOKit: first gamma, then beta
gamma_cols = [f'gamma{i}' for i in range(p)]
beta_cols = [f'beta{i}' for i in range(p)]
# Add column for number of iterations
columns = adj_vec_cols + gamma_cols + beta_cols + ['iterations']

# Initialize DataFrame
data_collector = pd.DataFrame(columns=columns)

# Create directory if it doesn't exist
os.makedirs("train_datasets", exist_ok=True)

file_name = f"train_datasets/p_{p}_n_{nodes_num}_prob_{prob}_weighted_{weighted_bool}_bounds_{bounds_bool}.csv"

print(f"Starting generation of {dataset_size} samples...")

for i in range(dataset_size):
    if i % 100 == 0:
        print(f"Processed {i}/{dataset_size} samples")
    
    # Generate adjacency vector
    vector_edges = ErdosRenyiFabric.generate_vector_adjacency_elements(
        Data=data_collector,
        edge_prob=prob,
        weighted_bool=weighted_bool,
        num_possible_edges=num_possible_edges,
        adjacency_vec=adj_vec_cols
    )

    # Create graph
    G = ErdosRenyiFabric(vec_adjacancy=vector_edges, num_nodes=nodes_num)

    # Create QAOA object
    qaoa_graph = QAOAMaxCutEngine(Graph=G, reps=p)
    qaoa_graph.determine_initial_point()

    # Optimization
    try:
        if bounds_bool:
            opt_result = qaoa_graph.optimize(optimizer=optimizer, bounds=bounds)
        else:
            opt_result = qaoa_graph.optimize(optimizer=optimizer)
        
        # Extract parameters
        optimized_params = opt_result.x
        
        # Use improved function to get iteration count
        n_iterations = get_iteration_count(opt_result, optimizer)
        
        # Additional debug information
        print(f"Sample {i}: optimizer={optimizer}, iterations={n_iterations}")
        if i < 5:  # Show detailed information only for first 5 samples
            print(f"  opt_result attributes: {[attr for attr in dir(opt_result) if not attr.startswith('_')]}")
            for attr in ['nit', 'nfev', 'njev', 'nhev']:
                if hasattr(opt_result, attr):
                    print(f"  {attr}: {getattr(opt_result, attr)}")
        
        gamma_params = optimized_params[:p]
        beta_params = optimized_params[p:]
        
        # Combine data including number of iterations
        graph_data = np.concatenate([vector_edges, gamma_params, beta_params, [n_iterations]])
        
        # Add to DataFrame
        new_row = pd.DataFrame([graph_data], columns=columns)
        data_collector = pd.concat([data_collector, new_row], ignore_index=True)
        
        # Save every 100 iterations
        if i % 100 == 0:
            data_collector.to_csv(file_name, index=False)
            # Check iteration statistics
            if 'iterations' in data_collector.columns and len(data_collector) > 0:
                iter_stats = data_collector['iterations'].describe()
                print(f"📊 Iteration statistics at sample {i}:")
                print(f"  Mean: {iter_stats['mean']:.2f}")
                print(f"  Median: {iter_stats['50%']:.2f}")
                print(f"  Min: {iter_stats['min']:.0f}")
                print(f"  Max: {iter_stats['max']:.0f}")
            
    except Exception as e:
        print(f"❌ Error during optimization at iteration {i}: {e}")
        continue

# Final save
data_collector.to_csv(file_name, index=False)
print(f"Data saved to {file_name}")
print(f"Total successfully processed {len(data_collector)} samples")

# Final verification of iterations column
if len(data_collector) > 0:
    print(f"Final DataFrame columns: {list(data_collector.columns)}")
    if 'iterations' in data_collector.columns:
        print(f"Iterations column successfully created with {len(data_collector)} samples")
        print(f"Iterations range: min={data_collector['iterations'].min()}, max={data_collector['iterations'].max()}")
    else:
        print("WARNING: Iterations column not found in final DataFrame")
else:
    print("WARNING: No samples were successfully processed")