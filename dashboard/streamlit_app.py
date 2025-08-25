# dashboard/streamlit_app.py


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="TCM Lab Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🧠 TCM Lab Dashboard")
st.markdown("**Transactive Cognitive Memory System** - Experiment Analysis & Monitoring")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # Check if runs directory exists
    runs_dir = Path("runs")
    if not runs_dir.exists():
        st.warning("No runs directory found. Run experiments first:")
        st.code("python app.py run-experiment")
        runs_available = False
    else:
        # Get all run directories
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        runs_available = len(run_dirs) > 0
    
    if runs_available:
        # Sort runs by modification time
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Create run options
        run_options = [d.name for d in run_dirs]
        selected_run = st.selectbox("Select Experiment Run", run_options)
        
        # Load button
        if st.button("Load Run Data"):
            st.session_state['current_run'] = selected_run
    else:
        st.warning("No experiment runs found")
        selected_run = None
    
    st.divider()
    
    # Quick actions
    st.header("Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    with col2:
        if st.button("📊 Compare All"):
            st.session_state['compare_mode'] = True

# Main content area
if runs_available and selected_run:
    # Load metrics and results
    metrics_file = runs_dir / selected_run / "metrics.json"
    results_file = runs_dir / selected_run / "results.json"
    events_file = runs_dir / selected_run / "events.jsonl"
    
    # Check if files exist
    if not metrics_file.exists():
        st.error(f"Metrics file not found for run: {selected_run}")
    else:
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Display key metrics
        st.header("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            retrieval_eff = metrics.get('retrieval_efficiency', 0)
            st.metric(
                "Retrieval Efficiency",
                f"{retrieval_eff:.3f}",
                delta=f"+{(retrieval_eff - 0.65):.3f} vs Isolated" if retrieval_eff > 0 else None
            )
        
        with col2:
            transfer_acc = metrics.get('transfer_accuracy', 0)
            st.metric(
                "Transfer Accuracy", 
                f"{transfer_acc:.3f}",
                delta=f"+{(transfer_acc - 0.45):.3f} vs Selective" if transfer_acc > 0 else None
            )
        
        with col3:
            success_rate = metrics.get('task_success_rate', 0)
            st.metric(
                "Task Success Rate",
                f"{success_rate:.3f}",
                delta=f"+{(success_rate - 0.60):.3f} vs Baseline" if success_rate > 0 else None
            )
        
        with col4:
            avg_time = metrics.get('avg_execution_time', 0)
            st.metric(
                "Avg Execution Time",
                f"{avg_time:.2f}s"
            )
        
        # Memory Statistics
        if 'memory_stats' in metrics:
            st.header("💾 Memory Statistics")
            memory_stats = metrics['memory_stats']
            
            # Check memory backend type
            if 'trust_scores' in memory_stats:
                # TCM Backend - Show trust scores
                st.subheader("Trust Score Distribution (TCM)")
                
                trust_data = []
                for key, value in memory_stats.get('trust_scores', {}).items():
                    agent, topic = key.rsplit('_', 1)
                    trust_data.append({
                        'Agent': agent,
                        'Topic': topic,
                        'Trust Score': value['score'],
                        'Confidence': value['confidence']
                    })
                
                if trust_data:
                    df_trust = pd.DataFrame(trust_data)
                    
                    # Create heatmap
                    fig = px.scatter(df_trust, 
                                   x='Trust Score', 
                                   y='Confidence',
                                   color='Agent',
                                   hover_data=['Topic'],
                                   title='Agent Expertise Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show expertise matrix
                    st.subheader("Expertise Matrix")
                    pivot_df = df_trust.pivot_table(
                        index='Agent', 
                        columns='Topic', 
                        values='Trust Score',
                        aggfunc='mean'
                    ).round(3)
                    st.dataframe(pivot_df, use_container_width=True)
            
            elif 'entries_per_agent' in memory_stats:
                # Isolated Backend
                st.subheader("Memory Distribution (Isolated)")
                entries = memory_stats.get('entries_per_agent', {})
                
                df_entries = pd.DataFrame(
                    list(entries.items()),
                    columns=['Agent', 'Entries']
                )
                
                fig = px.bar(df_entries, x='Agent', y='Entries',
                           title='Memory Entries per Agent')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show general stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entries", memory_stats.get('total_entries', 0))
            with col2:
                if 'delegation_rate' in memory_stats:
                    st.metric("Delegation Rate", f"{memory_stats['delegation_rate']:.2%}")
            with col3:
                if 'graph_density' in memory_stats:
                    st.metric("Graph Density", f"{memory_stats['graph_density']:.3f}")
        
        # Load and display results if available
        if results_file.exists():
            st.header("📈 Query Analysis")
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Convert to DataFrame for analysis
            results_data = []
            for r in results:
                results_data.append({
                    'Query ID': r['query_id'],
                    'Success': r['success'],
                    'Retrieval Score': r['retrieval_score'],
                    'Execution Time': r['execution_time'],
                    'Memory Writes': r['memory_writes']
                })
            
            df_results = pd.DataFrame(results_data)
            
            # Plot success over time
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(df_results, 
                            x='Query ID', 
                            y='Retrieval Score',
                            title='Retrieval Performance Over Time',
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Calculate cumulative success rate
                df_results['Cumulative Success Rate'] = df_results['Success'].expanding().mean()
                
                fig = px.line(df_results,
                            x='Query ID',
                            y='Cumulative Success Rate',
                            title='Cumulative Success Rate',
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Memory growth analysis
            st.subheader("Memory Growth Analysis")
            df_results['Cumulative Writes'] = df_results['Memory Writes'].cumsum()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=df_results['Query ID'], 
                         y=df_results['Cumulative Writes'],
                         name='Cumulative Memory Writes',
                         line=dict(color='blue')),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=df_results['Query ID'],
                         y=df_results['Retrieval Score'],
                         name='Retrieval Score',
                         line=dict(color='orange')),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Query ID")
            fig.update_yaxes(title_text="Memory Writes", secondary_y=False)
            fig.update_yaxes(title_text="Retrieval Score", secondary_y=True)
            fig.update_layout(title="Memory Growth vs Retrieval Performance")
            
            st.plotly_chart(fig, use_container_width=True)

# Comparison Mode
if st.session_state.get('compare_mode', False):
    st.header("🔄 Backend Comparison")
    
    # Collect all metrics from different runs
    comparison_data = []
    
    if runs_available:
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                metrics_file = run_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Extract backend and task from run name
                    parts = run_dir.name.split('_')
                    backend = "unknown"
                    task = "unknown"
                    
                    for part in parts:
                        if part in ["tcm", "isolated", "shared", "selective"]:
                            backend = part
                        elif "synthesis" in part:
                            task = "synthesis"
                        elif "problem" in part:
                            task = "problem_solving"
                        elif "reasoning" in part:
                            task = "reasoning"
                    
                    comparison_data.append({
                        'Backend': backend,
                        'Task': task,
                        'Retrieval Efficiency': metrics.get('retrieval_efficiency', 0),
                        'Transfer Accuracy': metrics.get('transfer_accuracy', 0),
                        'Task Success Rate': metrics.get('task_success_rate', 0),
                        'Avg Time': metrics.get('avg_execution_time', 0)
                    })
    
    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        
        # Group by backend for overall comparison
        backend_avg = df_compare.groupby('Backend').mean().round(3)
        
        st.subheader("Average Performance by Backend")
        st.dataframe(backend_avg, use_container_width=True)
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(backend_avg.reset_index(),
                        x='Backend',
                        y='Retrieval Efficiency',
                        title='Retrieval Efficiency Comparison',
                        color='Backend')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(backend_avg.reset_index(),
                        x='Backend',
                        y='Task Success Rate',
                        title='Task Success Rate Comparison',
                        color='Backend')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown by task
        st.subheader("Performance by Task Type")
        
        for task in df_compare['Task'].unique():
            if task != "unknown":
                st.write(f"**{task.replace('_', ' ').title()}**")
                task_data = df_compare[df_compare['Task'] == task]
                st.dataframe(task_data[['Backend', 'Retrieval Efficiency', 
                                       'Transfer Accuracy', 'Task Success Rate']], 
                           use_container_width=True)
    else:
        st.info("No comparison data available. Run experiments with different backends first.")

else:
    if not runs_available:
        # Show instructions
        st.info("👋 Welcome to Orchestrix for Distributed Systems")
        st.markdown("""
        ### Getting Started
        
        1. **Run an experiment:**
        ```bash
        python app.py run-experiment --memory-backend tcm --task exploratory_synthesis
        ```
        
        2. **Run all comparisons:**
        ```bash
        python app.py run-all
        ```
        
        3. **Refresh this dashboard** to see results
        
        ### Available Commands
        - `python app.py test` - Test setup
        - `python app.py list-runs` - List all experiments
        - `python app.py clean` - Clean all runs
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
TCM Lab v0.1.0 | Built with Streamlit
</div>
""", unsafe_allow_html=True)
