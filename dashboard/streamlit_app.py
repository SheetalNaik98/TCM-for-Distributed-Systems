# streamlit_app.py
"""
TCM Lab Dashboard - Streamlit Cloud Version
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Must be the first Streamlit command
st.set_page_config(
    page_title="TCM Lab Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 TCM Lab Dashboard")
st.markdown("**Transactive Cognitive Memory System** - Experiment Analysis")

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = True

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Check for runs directory
    runs_dir = Path("runs")
    
    # For Streamlit Cloud, we'll use sample data if no runs exist
    if not runs_dir.exists() or len(list(runs_dir.glob("*/metrics.json"))) == 0:
        st.info("Using sample data for demonstration")
        use_sample = True
    else:
        use_sample = st.checkbox("Use sample data", value=False)
    
    if not use_sample:
        # Load actual runs
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
        if run_dirs:
            run_options = [d.name for d in run_dirs]
            selected_run = st.selectbox("Select Run", run_options)
        else:
            selected_run = None
            st.warning("No valid runs found")
    else:
        selected_run = "sample_run"

# Main content
if selected_run == "sample_run" or use_sample:
    # Generate sample data for demonstration
    st.info("📊 Displaying sample data for demonstration purposes")
    
    # Sample metrics
    metrics = {
        "retrieval_efficiency": 0.876,
        "transfer_accuracy": 0.923,
        "task_success_rate": 0.845,
        "avg_execution_time": 2.34,
        "total_memory_writes": 142,
        "memory_stats": {
            "total_entries": 142,
            "delegation_rate": 0.72,
            "trust_scores": {
                "planner_planning": {"score": 0.95, "confidence": 15},
                "planner_research": {"score": 0.42, "confidence": 8},
                "researcher_nlp": {"score": 0.82, "confidence": 12},
                "researcher_ml": {"score": 0.88, "confidence": 14},
                "verifier_verification": {"score": 0.91, "confidence": 13}
            }
        }
    }
    
    # Sample results for charts
    sample_results = []
    for i in range(20):
        sample_results.append({
            "query_id": i,
            "success": np.random.random() > 0.3,
            "retrieval_score": 0.7 + np.random.random() * 0.3,
            "execution_time": 2 + np.random.random(),
            "memory_writes": np.random.randint(5, 15)
        })
    
elif selected_run:
    # Load actual data
    metrics_file = runs_dir / selected_run / "metrics.json"
    results_file = runs_dir / selected_run / "results.json"
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results_raw = json.load(f)
                sample_results = []
                for r in results_raw:
                    sample_results.append({
                        "query_id": r.get("query_id", 0),
                        "success": r.get("success", False),
                        "retrieval_score": r.get("retrieval_score", 0),
                        "execution_time": r.get("execution_time", 0),
                        "memory_writes": r.get("memory_writes", 0)
                    })
        else:
            sample_results = []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        metrics = {}
        sample_results = []
else:
    st.warning("No run selected")
    metrics = {}
    sample_results = []

# Display metrics if available
if metrics:
    st.header("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        retrieval_eff = metrics.get('retrieval_efficiency', 0)
        st.metric(
            "Retrieval Efficiency",
            f"{retrieval_eff:.3f}",
            delta=f"+{(retrieval_eff - 0.65):.3f}" if retrieval_eff > 0.65 else None
        )
    
    with col2:
        transfer_acc = metrics.get('transfer_accuracy', 0)
        st.metric(
            "Transfer Accuracy",
            f"{transfer_acc:.3f}",
            delta=f"+{(transfer_acc - 0.45):.3f}" if transfer_acc > 0.45 else None
        )
    
    with col3:
        success_rate = metrics.get('task_success_rate', 0)
        st.metric(
            "Task Success Rate",
            f"{success_rate:.3f}",
            delta=f"+{(success_rate - 0.60):.3f}" if success_rate > 0.60 else None
        )
    
    with col4:
        avg_time = metrics.get('avg_execution_time', 0)
        st.metric(
            "Avg Execution Time",
            f"{avg_time:.2f}s"
        )
    
    # Performance Comparison
    st.header("📈 Backend Performance Comparison")
    
    # Create comparison data
    comparison_data = {
        'Backend': ['TCM', 'Isolated', 'Shared', 'Selective'],
        'Retrieval Efficiency': [0.876, 0.650, 0.750, 0.700],
        'Transfer Accuracy': [0.923, 0.000, 1.000, 0.450],
        'Task Success Rate': [0.845, 0.600, 0.720, 0.680]
    }
    
    df_compare = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieval Efficiency")
        st.bar_chart(df_compare.set_index('Backend')['Retrieval Efficiency'])
    
    with col2:
        st.subheader("Task Success Rate")
        st.bar_chart(df_compare.set_index('Backend')['Task Success Rate'])
    
    # Trust Scores if TCM
    if 'memory_stats' in metrics and 'trust_scores' in metrics['memory_stats']:
        st.header("🎯 Trust Score Distribution")
        
        trust_data = []
        for key, value in metrics['memory_stats']['trust_scores'].items():
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                agent, topic = parts
                trust_data.append({
                    'Agent': agent,
                    'Topic': topic,
                    'Trust Score': value['score'],
                    'Confidence': value['confidence']
                })
        
        if trust_data:
            df_trust = pd.DataFrame(trust_data)
            
            # Create pivot table for heatmap-like display
            pivot_df = df_trust.pivot_table(
                index='Agent',
                columns='Topic',
                values='Trust Score',
                aggfunc='mean'
            )
            
            st.subheader("Agent-Topic Expertise Matrix")
            st.dataframe(
                pivot_df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1),
                use_container_width=True
            )
    
    # Query Analysis
    if sample_results:
        st.header("📊 Query Performance Analysis")
        
        df_results = pd.DataFrame(sample_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Retrieval Score Over Time")
            st.line_chart(df_results.set_index('query_id')['retrieval_score'])
        
        with col2:
            st.subheader("Success Rate Progression")
            df_results['cumulative_success'] = df_results['success'].expanding().mean()
            st.line_chart(df_results.set_index('query_id')['cumulative_success'])
        
        # Memory growth
        st.subheader("Memory Growth")
        df_results['cumulative_writes'] = df_results['memory_writes'].cumsum()
        st.area_chart(df_results.set_index('query_id')['cumulative_writes'])

# Instructions
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    ### Running Experiments
    
    1. **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/tcm-system.git
    cd tcm-system
    ```
    
    2. **Set up environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    
    3. **Configure API keys**:
    ```bash
    cp .env.example .env
    # Edit .env with your API keys
    ```
    
    4. **Run experiments**:
    ```bash
    python app.py run-experiment --memory-backend tcm
    python app.py run-all
    ```
    
    5. **View results**: Refresh this dashboard
    
    ### Understanding Metrics
    
    - **Retrieval Efficiency**: How well agents find relevant information (0-1)
    - **Transfer Accuracy**: Correctness of delegation decisions (0-1)
    - **Task Success Rate**: Percentage of successfully completed tasks
    - **Trust Scores**: Learned expertise levels for each agent-topic pair
    """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
    <p>TCM Lab Dashboard | 
    <a href='https://github.com/SheetalNaik98/TCM-for-Distributed-Systems'>GitHub Repository</a> | 
    Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
