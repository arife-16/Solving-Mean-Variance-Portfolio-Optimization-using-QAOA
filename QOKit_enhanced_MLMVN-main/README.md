# SmartQAOA: Neural Network-Enhanced Quantum Approximate Optimization Algorithm

A comprehensive implementation of QAOA (Quantum Approximate Optimization Algorithm) enhanced with neural networks for solving MaxCut, MinCut, and portfolio optimization problems.

By SigmaPublishinQ Team https://www.linkedin.com/company/sigma-publishinq


The neural network in this code is integrated with QOKit and trained to predict optimal parameters for the QAOA (Quantum Approximate Optimization Algorithm) applied to the MaxCut problem on random graphs. It takes the graph's adjacency vector as input and outputs parameters (γ and β) used to compute the energy and quality of the graph partition. Training is performed on synthetic data generated for graphs with a specified number of nodes and edge probabilities to enhance QAOA performance compared to default initial parameters, enabling faster identification of effective graph partition solutions.

The purpose of using the neural network is to replace the computationally intensive process of optimizing QAOA parameters, which typically involves multiple evaluations of the objective function, with a rapid prediction from a pre-trained neural network. The advantage lies in significantly speeding up the parameter acquisition process, improving solution quality (closer to the optimal cut), and reducing computational costs, particularly for large graphs where classical optimization becomes prohibitively slow.

## 🚀 Project Overview

SmartQAOA consists of two main components:

### 1. short_SmartQAOA ⚛️
An optimized QAOA implementation that uses neural networks to predict initial QAOA parameters (γ, β), reducing iterations to one and improving the approximation ratio by **5.27%** compared to numerical optimization (COBYLA).

### 2. full-featured SmartQAOA with MLMVN ⚛️ ⚛️ ⚛️
A scalable QAOA extension supporting MaxCut, MinCut, and portfolio optimization with portfolio cuts and advanced features including multilayer neural network based on multi-valued neurons (Aizenberg MLMVN), recursive QAOA (RQAOA), and spectral analysis.

## 📊 Key Features

- **Neural Parameter Prediction**: uses neural networks to predict optimal QAOA parameters
- **Multiple Problem Types**: supports MaxCut, MinCut, and portfolio optimization
- **Significant Speedup**: ~15-20x faster execution with single iteration
- **Quantum-Ready**: portable to quantum devices via pytket
- **Modular Design**: clean, extensible architecture

## 🛠️ Technologies

- **QOKit Library**: core QAOA implementation
- **Multilayer neural network based on Multi-Valued Neurons (Aizenberg MLMVN)**: core neural network implementation
- **PyTorch**: neural network framework
- **NetworkX**: graph operations
- **Qiskit**: quantum computing framework
- **pytket**: quantum device integration
- **Google Colab**: development environment

## 📋 Requirements

### Environment
- Google Colab (CPU, 12 GB RAM, Ubuntu) or local Python environment
- Python 3.7+

### Dependencies
```bash
pip install urllib3==1.26.6 qiskit==1.0.2 qiskit-aer==0.14.1 networkx tqdm pytket pytket-quantinuum pytket-qiskit torch pandas
```

## 🚀 Quick Start

### short_SmartQAOA

1. **Clone the Repository**
   ```bash
   # In Google Colab
   # Copy to: /content/drive/MyDrive/QOKit_enhanced_MLMVN/
   ```

2. **Install Dependencies**
   ```bash
   pip install urllib3==1.26.6 qiskit==1.0.2 qiskit-aer==0.14.1 networkx tqdm pytket pytket-quantinuum pytket-qiskit torch pandas
   ```

3. **Run the Benchmark**
   ```bash
   # Find benchmark_short.ipynb in https://github.com/Serhii-Barskyi/QOKit_enhanced_MLMVN/tree/main/short_smart_qaoa_example 
   # Open benchmark_short.ipynb in Google Colab
   # Execute all cells sequentially
   ```

### Expected Results
- **Approximation Ratio**: 0.8159 (5.27% improvement over COBYLA)
- **Iteration Count**: 1 (vs. multiple iterations in traditional QAOA)
- **Speedup**: ~15-20x faster execution
- **Neural Network Training Time**: ~2.23 seconds

## 📁 Project Structure

### short_SmartQAOA
```
├── benchmark_short.ipynb          # Main benchmark file
├── maxcut_data_forge.py          # Training data generation
├── neural_qaoa_trainer.py        # Neural network training
├── iteration_analyzer.py         # Performance analysis
├── cut_analyzer.py               # Cut quality evaluation
├── qaoa_maxcut_engine.py         # Core QAOA simulation
├── neuro_maxcut_tests.py         # Unit tests
└── erdos_renyi_fabric.py         # Graph generation

```

### full-featured SmartQAOA with MLMVN
```
smart_qaoa.py                     # Main Smart QAOA module with neural optimization
├── mlmvn_network.py              # Multilayer neural network (MLMVN) for QAOA
│   └── complex_mvn.py            # Complex-valued neurons (MVN)
├── adamw_optimizer.py            # AdamW optimizer with cosine scheduler
├── rqaoa_agent_main.py           # RL-agent for recursive QAOA (RQAOA, reinforcement learning)
├── computational_core.py         # Expectation and Hamiltonian calculations
├── graph.py                      # Graph operations (generation, vertex elimination)
├── scaling_analyzer.py           # QAOA scalability analysis
├── adaptive_scaling.py           # Adaptive scaling (entropy, topology)
├── spectral_core.py              # Spectral operations (Laplacian, adjacency matrix)
├── cache_manager.py              # Cache management for spectral data
├── graph_spectral_fft.py         # Spectral FFT encoding for graphs
├── config.py                     # Centralized configuration
├── rqaoa_algorithms.py           # Classical RQAOA algorithm
└── utils.py                      # Utilities (random weights, file operations)
```

## 📈 Performance Metrics

| Metric | short_SmartQAOA | Traditional QAOA |
|--------|-----------------|------------------|
| Approximation Ratio | 0.8159 | 0.7752 |
| Iteration Count | 1 | 10-50 |
| Speedup | 15-20x | 1x |
| Training Time | ~2.23s | N/A |

## 🧪 Testing

### Input Data
- Dataset of 1000 Erdős-Rényi graphs (N=10, p=3)
- Test graphs and predictions in `neural_maxcut_results_p3_n10.csv`

### Output Data
- Neural network model: `net_p3_n10.pth`
- Performance metrics and analysis results

### Verification
Results including approximation ratio and execution time will appear in the notebook output after running `benchmark_short.ipynb`.

## 🔬 Advanced Features (full-featured SmartQAOA)

### Multilayer neural network based on Multi-Valued Neurons (Aizenberg MLMVN)
- Predicts optimal QAOA parameters with complex weights
- Adapts to graph topology using least squares method with soft margins
- Enhanced stability with AdamW optimizer

### Recursive QAOA (RQAOA)
- Implements recursive vertex removal for MinCut optimization
- Supports dynamic graph updates for portfolio optimization
- Enables scalable solutions for larger problem instances

### Spectral Analysis
- Analyzes graph's spectral gap for improved scalability
- Provides theoretical insights into problem hardness
- Guides parameter optimization strategies

## 🎯 Applications

### MaxCut Problem
Maximizing the sum of edge weights between graph partitions - fundamental in:
- Network analysis
- Image segmentation
- Circuit design optimization

### MinCut Problem
Minimizing graph cuts for:
- Network flow optimization
- Clustering algorithms
- Resource allocation

### Portfolio Optimization with Portfolio cuts
Dynamic asset clustering to minimize financial risk:
- Risk minimization strategies
- Asset correlation analysis
- Dynamic portfolio rebalancing

## 🔮 Future Prospects

- **Scalability**: expand to larger graphs (N > 16)
- **Quantum Integration**: full deployment on quantum devices via pytket
- **Enhanced Neural Networks**: broader QAOA applications
- **Real-world Applications**: financial portfolio optimization at scale
- **Hybrid Classical-Quantum**: optimal resource utilization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Professor Igor Aizenberg https://scholar.google.com/citations?hl=en&user=ZjfN_9AAAAAJ&view_op=list_works&sortby=pubdate
- QOKit library developers https://github.com/jpmorganchase/QOKit
- Quantum computing community
- Google Colab platform
- Potomac Quantum Innovation Center https://www.pqic.org/
- Aqora https://aqora.io/
- Connected DMV https://www.connecteddmv.org/
- Quantum Australia https://quantum-australia.com/
- qBraid https://www.qbraid.com/



## 📞 Contact
**Serhii Barskyi** 
Reinforcement Learning Engineer / Quantum Optimization Specialist |ψ⟩⊗|φ⟩ QAOA | QUBO: SK, PO, MaxCut | Quantum Simulators DevOps |⇅〉 Quantum RL | Django REST | Inverse/Bayesian DRL for Next Best Action ⇒ Qiskit | QOKit

**https://www.linkedin.com/in/serhii-barskyi/**

**https://www.linkedin.com/company/sigma-publishinq**



---

**Note**: This project is optimized for Google Colab CPU environment but can be adapted for local execution and quantum hardware deployment.
