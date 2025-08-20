# TCM Lab - Transactive Cognitive Memory System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

TCM Lab implements a **Transactive Cognitive Memory (TCM)** system for multi-agent AI, enabling efficient knowledge distribution and retrieval across specialized agents. The system uses Beta distribution-based trust models to dynamically route information to the most qualified agents.

## Key Features

- **Multi-Agent Architecture**: Three specialized agents (Planner, Researcher, Verifier)
- **Dynamic Trust Model**: Beta distribution-based expertise tracking
- **Memory Backends**: Compare TCM against baseline approaches (Isolated, Shared, Selective)
- **Comprehensive Evaluation**: Built-in tasks and metrics for systematic comparison
- **Real-time Dashboard**: Streamlit-based visualization of experiments
- **LLM Agnostic**: Support for OpenAI and Anthropic APIs

## Installation

### Prerequisites
- Python 3.9+
- OpenAI or Anthropic API key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/tcm-system.git
cd tcm-system
