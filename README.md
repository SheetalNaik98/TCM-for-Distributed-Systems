# Transactive Cognitive Memory for Distributed Systems

## Overview

This repository contains the implementation of a Transactive Cognitive Memory (TCM) system for multi-agent artificial intelligence, developed as part of Master's research in Data Analytics Engineering at Northeastern University.

## Abstract

Transactive memory is a socio-cognitive mechanism wherein members of a group remember "who knows what" and coordinate accordingly. This implementation operationalizes that concept for multi-agent AI systems through a Beta-Bernoulli trust model with Thompson Sampling for probability-matching delegation among agents. The system achieved a delegation rate of 66.7% while maintaining 100% task success rate, demonstrating superior specialization efficiency compared to baseline memory architectures.

## Research Motivation

Current multi-agent AI systems face significant inefficiencies in knowledge management:
- Redundant storage across agents leading to 40% wasted computational resources
- Inefficient information retrieval with 60% increased latency
- Static expertise assignment requiring manual configuration

This research addresses these challenges by implementing dynamic expertise discovery through probabilistic trust modeling.

## System Architecture

The TCM system comprises three specialized agents:

```python
agents = {
    "planner": PlannerAgent(),      # Decomposes queries into actionable strategies
    "researcher": ResearcherAgent(), # Synthesizes information from distributed memory stores
    "verifier": VerifierAgent()      # Validates claims and updates trust parameters
}
```

Four memory backend architectures are implemented for comparative analysis:

| Backend | Strategy | Description |
|---------|----------|-------------|
| Isolated | Private stores | Per-agent private memory with no cross-agent access |
| Shared | Global pool | Single memory store accessible to all agents |
| Selective | Rule-based | Static filtering based on predefined expertise |
| TCM | Trust-weighted | Dynamic routing via Thompson Sampling |

## Theoretical Foundation

### Beta-Bernoulli Trust Model

Each agent i maintains trust parameters (αi, βi) for each topic domain. Trust evolves through Bayesian updates:

```
Trust(agent_i, topic_j) ~ Beta(α_ij, β_ij)
α_ij ← α_ij + success
β_ij ← β_ij + failure
```

### Thompson Sampling

Delegation decisions employ Thompson Sampling for optimal exploration-exploitation balance:

```
θ_i ~ Beta(α_i, β_i) for each agent
delegate_to = argmax_i(θ_i)
```

## Experimental Results

### Performance Metrics (Iteration 3)

| Backend | Success Rate | Delegation Rate | Memory Reads | Latency |
|---------|-------------|-----------------|--------------|---------|
| TCM | 100% | 66.7% | 1 | 6.3s |
| Isolated | 100% | 0% | 1 | 6.8s |
| Shared | 100% | 0% | 1 | 5.8s |
| Selective | 100% | 0% | 2 | 7.0s |

### Trust Score Convergence

The system demonstrated rapid expertise discovery within 10-15 interactions:
- Planner expertise in planning tasks: 0.95
- Researcher expertise in NLP/ML domains: 0.82
- Verifier expertise in validation: 0.91

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)
- OpenAI or Anthropic API key (optional for LLM integration)

### Setup Instructions

Clone the repository:
```bash
git clone https://github.com/SheetalNaik98/TCM-for-Distributed-Systems.git
cd TCM-for-Distributed-Systems
```

Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Configure environment variables:
```bash
cp .env.example .env
# Edit .env to add API keys if using LLM providers
```

## Usage

### Running Experiments

Test system configuration:
```bash
python app.py test
```

Execute single experiment:
```bash
python app.py run-experiment --memory-backend tcm --task exploratory_synthesis
```

Run comparative analysis across all backends:
```bash
python app.py run-all
```

View experiment results:
```bash
python app.py list-runs
```

### Dashboard Visualization

Launch the Streamlit dashboard for real-time monitoring:
```bash
streamlit run dashboard/streamlit_app.py
```

## Project Structure

```
TCM-for-Distributed-Systems/
├── app.py                  # Main CLI application
├── tcm_lab/
│   ├── agents/            # Agent implementations
│   │   ├── base.py        # Base agent class
│   │   ├── planner.py     # Planning agent
│   │   ├── researcher.py  # Research agent
│   │   └── verifier.py    # Verification agent
│   ├── memory/
│   │   ├── base.py        # Memory interface
│   │   ├── baselines.py   # Baseline implementations
│   │   ├── tcm.py         # TCM implementation
│   │   └── vector_store.py # Vector storage
│   ├── eval/              # Evaluation framework
│   │   ├── tasks.py       # Task definitions
│   │   ├── harness.py     # Experiment harness
│   │   └── metrics.py     # Metrics calculation
│   ├── infra/             # Infrastructure
│   │   ├── config.py      # Configuration management
│   │   └── event_log.py   # Event logging
│   └── llm/               # LLM integration
│       └── provider.py    # Provider abstraction
├── dashboard/             # Visualization
│   └── streamlit_app.py   # Dashboard application
└── runs/                  # Experiment outputs
```

## Implementation Details

### TCM Delegation Algorithm

```python
def delegate_with_tcm(query, agents, trust_params):
    """
    Algorithm 1: TCM Delegation with Beta-Bernoulli Trust
    """
    # Step 1: Thompson Sampling
    samples = {}
    for agent_id, agent in agents.items():
        theta = np.random.beta(
            trust_params[agent_id]['alpha'],
            trust_params[agent_id]['beta']
        )
        samples[agent_id] = theta
    
    # Step 2: Select best agent
    best_agent_id = max(samples, key=samples.get)
    
    # Step 3: Process query
    result = agents[best_agent_id].process(query)
    
    # Step 4: Update trust based on verification
    if result.verdict == "SUPPORTED":
        trust_params[best_agent_id]['alpha'] += 1
    else:
        trust_params[best_agent_id]['beta'] += 1
    
    return result, best_agent_id
```

### Memory Management Strategy

```python
class TransactiveCognitiveMemory:
    def __init__(self, agents, topics):
        self.agent_stores = {agent: VectorStore() for agent in agents}
        self.trust_params = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        self.delegation_matrix = defaultdict(int)
        
    def write(self, content, topic, agent_id):
        # Determine best owner via Thompson Sampling
        best_owner = self.get_best_owner(topic, agent_id)
        
        # Track delegation
        if best_owner != agent_id:
            self.delegation_matrix[(agent_id, best_owner, topic)] += 1
        
        # Store in best owner's memory
        entry_id = self.agent_stores[best_owner].add(content)
        return entry_id
```

## Key Findings

1. **Specialization Efficiency**: TCM achieved 66.7% delegation rate while other backends showed 0% delegation, demonstrating superior task routing to domain experts.

2. **Maintained Accuracy**: All backends achieved 100% task success rate in final iteration, confirming that delegation does not compromise accuracy.

3. **Rapid Convergence**: Trust parameters stabilized within 10-15 interactions, enabling quick expertise discovery.

4. **Scalable Design**: System complexity scales linearly with agent count, supporting expansion to larger multi-agent systems.

## Distributed Systems Applications

### Edge-Cloud Architecture

The TCM framework is particularly suited for distributed deployments:
- Cloud agents handle computationally intensive research tasks
- Edge agents provide low-latency planning decisions
- Verification can be distributed based on criticality

### Fault Tolerance

Trust parameters naturally adapt to agent failures:
- Failed delegations reduce trust scores
- System automatically routes around unreliable agents
- No manual reconfiguration required

### Privacy Preservation

Category-bounded memory approaches can be integrated:
- Sensitive data remains on edge devices
- Only metadata and trust scores synchronized
- Compliant with privacy-preserving evaluation frameworks

## Related Work

This research builds upon:

1. Thompson, W.R. (1933). On the Likelihood That One Unknown Probability Exceeds Another. Biometrika, 25, 285-294.

2. Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A Tutorial on Thompson Sampling. Foundations and Trends in Machine Learning, 11(1), 1-96.

3. Zulfikar, W., Chan, S., & Maes, P. (2024). Memoro: Using Large Language Models to Realize a Concise Interface for Real-Time Memory Augmentation. CHI '24.

4. Kirmayr, J., et al. (2025). CarMem: Enhancing Long-Term Memory in LLM Voice Assistants through Category-Bounding. arXiv:2501.09645.

5. PIN AI Team, et al. (2025). GOD model: Privacy Preserved AI School for Personal Assistant. arXiv:2502.18527.

## Citation

```bibtex
@mastersthesis{naik2025tcm,
  title={Transactive Cognitive Memory for Multi-Agent AI and Distributed Systems},
  author={Naik, Sheetal},
  school={Northeastern University},
  year={2025}
}
```

## Future Work

- Hierarchical trust models for multi-level agent organizations
- Asynchronous message passing for true distributed operation
- Context-aware Thompson Sampling for domain-specific routing
- Integration with federated learning frameworks

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Northeastern University, MS Data Analytics Engineering Program
- Contributors to the multi-agent AI research community

## Contact

Sheetal Naik  
Master of Science in Data Analytics Engineering  
Northeastern University  
GitHub: https://github.com/SheetalNaik98
