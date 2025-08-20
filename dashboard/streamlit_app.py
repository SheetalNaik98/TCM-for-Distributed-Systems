# dashboard/streamlit_app.py
"""
TCM Lab Dashboard - Real-time experiment visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="TCM Lab Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    h1 {color: #1f77b4;}
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_runs():
    """Load all experiment runs"""
    runs_dir = Path("runs")
    runs = []
    
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                metrics_file = run_dir / "metrics.json"
                events_file = run_dir / "events.jsonl"
                
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                        
                    runs.append({
                        "run_id": run_dir.name,
                        "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime),
                        "metrics": metrics,
                        "has_events": events_file.exists()
                    })
                    
    return sorted(runs, key=lambda x: x["timestamp"], reverse=True)


@st.cache_data
def load_events(run_id: str):
    """Load events for a specific run"""
    events_file = Path(f"runs/{run_id}/events.jsonl")
    events = []
    
    if events_file.exists():
        with open(events_file) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    continue
                    
    return events


def plot_metrics_comparison(runs):
    """Create comparison plots for multiple runs"""
    
    if not runs:
        return None
        
    # Extract data for plotting
    data = []
    for run in runs:
        metrics = run["metrics"]
        run_parts = run["run_id"].split("_")
        
        # Extract memory backend and task from run_id
        memory_backend = "unknown"
        task = "unknown"
        
        for part in run_parts:
            if part in ["isolated", "shared", "selective", "tcm"]:
                memory_backend = part
            elif "synthesis" in part or "problem" in part or "reasoning" in part:
                task = part
                
        data.append({
            "run_id": run["run_id"],
            "memory_backend": memory_backend,
            "task": task,
            "retrieval_efficiency": metrics.get("retrieval_efficiency", 0),
            "transfer_accuracy": metrics.get("transfer_accuracy", 0),
            "task_success_rate": metrics.get("task_success_rate", 0),
            "avg_execution_time": metrics.get("avg_execution_time", 0)
        })
        
    df = pd.DataFrame(data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Retrieval Efficiency", "Transfer Accuracy", 
                       "Task Success Rate", "Execution Time"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Retrieval Efficiency
    fig.add_trace(
        go.Bar(x=df["memory_backend"], y=df["retrieval_efficiency"], 
               name="Retrieval", marker_color="#1f77b4"),
        row=1, col=1
    )
    
    # Transfer Accuracy
    fig.add_trace(
        go.Bar(x=df["memory_backend"], y=df["transfer_accuracy"],
               name="Transfer", marker_color="#ff7f0e"),
        row=1, col=2
    )
    
    # Task Success Rate
    fig.add_trace(
        go.Bar(x=df["memory_backend"], y=df["task_success_rate"],
               name="Success", marker_color="#2ca02c"),
        row=2, col=1
    )
    
    # Execution Time
    fig.add_trace(
        go.Bar(x=df["memory_backend"], y=df["avg_execution_time"],
               name="Time", marker_color="#d62728"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False,
                     title_text="Memory Backend Performance Comparison")
    fig.update_xaxes(title_text="Memory Backend")
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="Rate", row=2, col=1)
    fig.update_yaxes(title_text="Time (s)", row=2, col=2)
    
    return fig


def plot_trust_evolution(events):
    """Plot trust score evolution for TCM"""
    
    trust_events = [e for e in events if e.get("event_type") == "verifier_response"]
    
    if not trust_events:
        return None
        
    # Extract trust updates
    trust_data = []
    for i, event in enumerate(trust_events):
        if event["data"].get("trust_updated"):
            trust_data.append({
                "event_num": i,
                "timestamp": event["timestamp"],
                "trust_updated": True
            })
            
    if not trust_data:
        return None
        
    df = pd.DataFrame(trust_data)
    
    fig = px.scatter(df, x="event_num", y=[1]*len(df), 
                    title="Trust Updates Over Time",
                    labels={"event_num": "Event Number", "y": "Trust Update"})
    fig.update_yaxes(visible=False)
    fig.update_traces(marker=dict(size=12, color="#1f77b4"))
    
    return fig


def plot_memory_growth(events):
    """Plot memory growth over time"""
    
    query_events = [e for e in events if e.get("event_type") == "query_complete"]
    
    if not query_events:
        return None
        
    data = []
    cumulative_writes = 0
    
    for event in query_events:
        cumulative_writes += event["data"].get("memory_writes", 0)
        data.append({
            "query_id": event["data"]["query_id"],
            "memory_writes": event["data"]["memory_writes"],
            "cumulative_writes": cumulative_writes,
            "retrieval_score": event["data"]["retrieval_score"]
        })
        
    df = pd.DataFrame(data)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=df["query_id"], y=df["cumulative_writes"],
                  name="Cumulative Writes", line=dict(color="#1f77b4")),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df["query_id"], y=df["retrieval_score"],
                  name="Retrieval Score", line=dict(color="#ff7f0e")),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_xaxes(title_text="Query ID")
    fig.update_yaxes(title_text="Memory Writes", secondary_y=False)
    fig.update_yaxes(title_text="Retrieval Score", secondary_y=True)
    fig.update_layout(title_text="Memory Growth and Retrieval Performance")
    
    return fig


# Main app
def main():
    st.title("🧠 TCM Lab Dashboard")
    st.markdown("**Transactive Cognitive Memory System** - Experiment Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Load runs
        runs = load_runs()
        
        if runs:
            run_options = [f"{r['run_id']} ({r['timestamp'].strftime('%Y-%m-%d %H:%M')})" 
                          for r in runs]
            selected_run_idx = st.selectbox("Select Run", range(len(run_options)), 
                                           format_func=lambda x: run_options[x])
            selected_run = runs[selected_run_idx]
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto-refresh", value=False)
            if auto_refresh:
                st.rerun()
                
        else:
            st.warning("No experiment runs found")
            selected_run = None
            
        st.divider()
        
        # Quick actions
        st.header("Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh"):
                st.cache_data.clear()
                st.rerun()
                
        with col2:
            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                
    # Main content
    if selected_run:
        # Metrics overview
        st.header("📊 Metrics Overview")
        
        metrics = selected_run["metrics"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Retrieval Efficiency",
                f"{metrics.get('retrieval_efficiency', 0):.3f}",
                delta=None
            )
            
        with col2:
            st.metric(
                "Transfer Accuracy",
                f"{metrics.get('transfer_accuracy', 0):.3f}",
                delta=None
            )
            
        with col3:
            st.metric(
                "Success Rate",
                f"{metrics.get('task_success_rate', 0):.3f}",
                delta=None
            )
            
        with col4:
            st.metric(
                "Avg Time",
                f"{metrics.get('avg_execution_time', 0):.2f}s",
                delta=None
            )
            
        # Comparison plots
        st.header("📈 Performance Comparison")
        
        comparison_fig = plot_metrics_comparison(runs[:10])  # Compare last 10 runs
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
            
        # Event analysis
        if selected_run["has_events"]:
            st.header("🔍 Event Analysis")
            
            events = load_events(selected_run["run_id"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                memory_fig = plot_memory_growth(events)
                if memory_fig:
                    st.plotly_chart(memory_fig, use_container_width=True)
                    
            with col2:
                trust_fig = plot_trust_evolution(events)
                if trust_fig:
                    st.plotly_chart(trust_fig, use_container_width=True)
                    
        # Memory statistics
        if "memory_stats" in metrics:
            st.header("💾 Memory Statistics")
            
            memory_stats = metrics["memory_stats"]
            
            # Display based on memory type
            if "trust_scores" in memory_stats:  # TCM
                st.subheader("Trust Scores")
                
                trust_df = pd.DataFrame([
                    {"Agent-Topic": k, "Score": v["score"], "Confidence": v["confidence"]}
                    for k, v in memory_stats.get("trust_scores", {}).items()
                ])
                
                if not trust_df.empty:
                    fig = px.scatter(trust_df, x="Score", y="Confidence", 
                                   hover_data=["Agent-Topic"],
                                   title="Trust Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif "entries_per_agent" in memory_stats:  # Isolated
                st.subheader("Entries per Agent")
                
                entries_df = pd.DataFrame([
                    {"Agent": k, "Entries": v}
                    for k, v in memory_stats.get("entries_per_agent", {}).items()
                ])
                
                if not entries_df.empty:
                    fig = px.bar(entries_df, x="Agent", y="Entries",
                                title="Memory Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
        # Raw data
        with st.expander("📋 Raw Metrics"):
            st.json(metrics)
            
    else:
        # No runs available
        st.info("No experiment runs available. Run experiments using:")
        st.code("python app.py run-experiment --memory-backend tcm --task exploratory_synthesis")
        
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        TCM Lab v0.1.0 | Built with Streamlit | 
        <a href='https://github.com/yourusername/tcm-lab'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
