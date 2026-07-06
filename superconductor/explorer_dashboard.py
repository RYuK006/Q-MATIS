import logging
import os
from collections import Counter
from .materials_lake import MaterialsLake

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """
    Generates a static HTML dashboard from the Knowledge Graph using Plotly.
    """
    @staticmethod
    def generate_html_summary(lake: MaterialsLake, output_path: str = "data/qmatis_explorer.html"):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Fetch summary stats from lake
            materials = lake.execute_read("SELECT is_rejected, generation_strategy, parent_id FROM materials")
            decisions = lake.execute_read("SELECT reason FROM decision_history WHERE action = 'Rejected'")
            
            rejection_reasons = Counter([d["reason"] for d in decisions if d["reason"]])
            
            total = len(materials)
            rejected = sum(1 for m in materials if m["is_rejected"])
            accepted = total - rejected
            
            strategies = Counter([m["generation_strategy"] for m in materials if m["generation_strategy"]])
            
            fig = make_subplots(
                rows=2, cols=2, 
                specs=[[{"type": "domain"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]], 
                subplot_titles=["Accepted vs Rejected", "Rejection Reasons", "Strategies Used", "Total Materials"]
            )
            
            # Top Left: Pie
            fig.add_trace(go.Pie(labels=["Accepted", "Rejected"], values=[accepted, rejected], name="Status"), row=1, col=1)
            
            # Top Right: Rejections
            if rejection_reasons:
                fig.add_trace(go.Bar(x=list(rejection_reasons.keys()), y=list(rejection_reasons.values()), name="Rejections"), row=1, col=2)
                
            # Bottom Left: Strategies
            if strategies:
                fig.add_trace(go.Bar(x=list(strategies.keys()), y=list(strategies.values()), name="Strategies"), row=2, col=1)
                
            # Bottom Right: Total Counter
            fig.add_trace(go.Indicator(
                mode="number",
                value=total,
                title={"text": "Total Entities in Graph"}
            ), row=2, col=2)
            
            fig.update_layout(title_text="Q-MATIS Materials Knowledge Graph Explorer", height=800)
            
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Dashboard successfully written to {output_path}")
            
        except ImportError:
            logger.warning("plotly is not installed. Generating a simple HTML table instead.")
            # fallback
            html = f"<html><body><h1>Q-MATIS Explorer</h1><p>Total Materials: {len(lake.execute_read('SELECT id FROM materials'))}</p></body></html>"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
