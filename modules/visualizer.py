"""
Cost Visualizer - Creates interactive Plotly charts
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


class CostVisualizer:
    def __init__(self, cost_data, recommendations):
        self.cost_data = cost_data
        self.recommendations = recommendations
    
    def create_service_pie_chart(self):
        """Create pie chart of costs by service"""
        by_service = self.cost_data.get('by_service', {})
        if not by_service:
            return go.Figure()
        
        # Top 5 + Others
        sorted_services = sorted(by_service.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_services[:5]
        others = sum(cost for _, cost in sorted_services[5:])
        
        labels = [svc for svc, _ in top_5]
        values = [cost for _, cost in top_5]
        
        if others > 0:
            labels.append('Others')
            values.append(others)
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title="Cost Distribution by Service",
            height=400
        )
        
        return fig
    
    def create_savings_bar_chart(self):
        """Create bar chart of savings by category"""
        if not self.recommendations:
            return go.Figure()
        
        # Group by category
        savings_by_category = {}
        for rec in self.recommendations:
            category = rec['category']
            savings_by_category[category] = savings_by_category.get(category, 0) + rec['estimated_savings']
        
        categories = list(savings_by_category.keys())
        savings = list(savings_by_category.values())
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=savings,
            marker_color='steelblue',
            text=[f'${s:,.0f}' for s in savings],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Potential Savings by Category",
            xaxis_title="Category",
            yaxis_title="Monthly Savings ($)",
            height=400
        )
        
        return fig
    
    def create_cost_trend_chart(self):
        """Create line chart of cost trends"""
        time_series = self.cost_data.get('time_series', [])
        if len(time_series) < 2:
            return go.Figure()
        
        dates = [record.get('date') for record in time_series]
        costs = [record.get('cost', 0) for record in time_series]
        
        fig = go.Figure()
        
        # Actual costs
        fig.add_trace(go.Scatter(
            x=dates,
            y=costs,
            mode='lines+markers',
            name='Actual Cost',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8)
        ))
        
        # Trend line
        import numpy as np
        x = np.arange(len(costs))
        z = np.polyfit(x, costs, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=p(x),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Cost Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Daily Cost ($)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
