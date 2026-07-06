import json
import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CandidateReporter:
    @staticmethod
    def generate_json_report(candidate: Dict[str, Any], output_dir: str = "data/reports"):
        os.makedirs(output_dir, exist_ok=True)
        cand_id = candidate.get("id", "UNKNOWN")
        file_path = os.path.join(output_dir, f"{cand_id}_report.json")
        
        # We need to ensure the structure object is serializable, or just export metadata
        report_data = {
            "candidate_id": cand_id,
            "formula": candidate.get("generated_formula", ""),
            "parent_material": candidate.get("parent_material", ""),
            "strategy": candidate.get("strategy", ""),
            "predicted_tc": candidate.get("predicted_tc", None),
            "uncertainty": candidate.get("uncertainty", None),
            "physics_score": candidate.get("physics_score", 1.0),
            "rejection_history": candidate.get("rejected_filters", ""),
            "crystal_system": candidate.get("metadata", {}).get("crystal_system", ""),
            "space_group": candidate.get("metadata", {}).get("spacegroup", ""),
            "lattice_parameters": candidate.get("metadata", {}).get("lattice_parameters", {})
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        logger.debug(f"Generated JSON report for {cand_id}")

class DashboardGenerator:
    @staticmethod
    def generate_html_summary(analytics: Dict[str, int], output_path: str = "data/candidate_generation_summary.html"):
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            labels = list(analytics.keys())
            values = list(analytics.values())
            
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "bar"}]], 
                                subplot_titles=["Rejection Reasons", "Rejection Counts"])
                                
            fig.add_trace(go.Pie(labels=labels, values=values, name="Rejections"), row=1, col=1)
            fig.add_trace(go.Bar(x=labels, y=values, name="Counts"), row=1, col=2)
            
            fig.update_layout(title_text="Candidate Generation Analytics")
            fig.write_html(output_path)
            logger.info(f"Dashboard successfully written to {output_path}")
            
        except ImportError:
            logger.warning("plotly is not installed. Generating a simple HTML table instead.")
            html = "<html><body><h1>Candidate Generation Analytics</h1><table border='1'><tr><th>Reason</th><th>Count</th></tr>"
            for k, v in analytics.items():
                html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            html += "</table></body></html>"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Fallback dashboard written to {output_path}")
