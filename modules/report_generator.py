"""
Report Generator - Creates reports in multiple formats
"""

import pandas as pd
from datetime import datetime

# PDF generation is optional
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class ReportGenerator:
    def __init__(self, cost_data, recommendations, ai_insights=None):
        self.cost_data = cost_data
        self.recommendations = recommendations
        self.ai_insights = ai_insights
    
    def generate(self, output_name, format_type='excel'):
        """Generate report in specified format"""
        if format_type == 'excel':
            return self._generate_excel(output_name)
        elif format_type == 'csv':
            return self._generate_csv(output_name)
        elif format_type == 'html':
            return self._generate_html(output_name)
        elif format_type == 'pdf':
            return self._generate_pdf(output_name)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_excel(self, output_name):
        """Generate Excel report"""
        output_file = f"{output_name}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary
            summary_data = {
                'Metric': ['Total Cost', 'Potential Savings', 'Recommendations', 'Optimization %'],
                'Value': [
                    f"${self.cost_data['summary']['total_cost']:,.2f}",
                    f"${sum(r['estimated_savings'] for r in self.recommendations):,.2f}",
                    len(self.recommendations),
                    f"{(sum(r['estimated_savings'] for r in self.recommendations) / self.cost_data['summary']['total_cost'] * 100):.1f}%"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Recommendations
            if self.recommendations:
                df_recs = pd.DataFrame(self.recommendations)
                df_recs.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Service Costs
            if self.cost_data['by_service']:
                service_df = pd.DataFrame(
                    list(self.cost_data['by_service'].items()),
                    columns=['Service', 'Cost']
                ).sort_values('Cost', ascending=False)
                service_df.to_excel(writer, sheet_name='Service Costs', index=False)
        
        return output_file
    
    def _generate_csv(self, output_name):
        """Generate CSV report"""
        output_file = f"{output_name}.csv"
        df = pd.DataFrame(self.recommendations)
        df.to_csv(output_file, index=False)
        return output_file
    
    def _generate_html(self, output_name):
        """Generate HTML dashboard"""
        output_file = f"{output_name}.html"
        
        total_cost = self.cost_data['summary']['total_cost']
        total_savings = sum(r['estimated_savings'] for r in self.recommendations)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Cloud Cost Optimization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric h3 {{ margin: 0; color: #666; font-size: 14px; }}
        .metric .value {{ font-size: 32px; font-weight: bold; color: #333; margin-top: 10px; }}
        table {{ width: 100%; background: white; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f9fafb; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>☁️ Cloud Cost Optimization Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Cost</h3>
            <div class="value">${total_cost:,.2f}</div>
        </div>
        <div class="metric">
            <h3>Potential Savings</h3>
            <div class="value" style="color: #10b981;">${total_savings:,.2f}</div>
        </div>
        <div class="metric">
            <h3>Recommendations</h3>
            <div class="value">{len(self.recommendations)}</div>
        </div>
        <div class="metric">
            <h3>Optimization %</h3>
            <div class="value" style="color: #10b981;">{(total_savings/total_cost*100):.1f}%</div>
        </div>
    </div>
    
    <h2>Recommendations</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Service</th>
            <th>Priority</th>
            <th>Savings</th>
            <th>Description</th>
        </tr>
"""
        
        for rec in self.recommendations:
            html += f"""
        <tr>
            <td>{rec['category']}</td>
            <td>{rec['service']}</td>
            <td>{rec['priority']}</td>
            <td>${rec['estimated_savings']:,.2f}</td>
            <td>{rec['description']}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_file
    
    def _generate_pdf(self, output_name):
        """Generate PDF report"""
        if not PDF_AVAILABLE:
            raise ImportError("reportlab not installed. Install with: pip install reportlab")
        
        output_file = f"{output_name}.pdf"
        
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("Cloud Cost Optimization Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Summary
        summary_text = f"""
        Total Cost: ${self.cost_data['summary']['total_cost']:,.2f}<br/>
        Potential Savings: ${sum(r['estimated_savings'] for r in self.recommendations):,.2f}<br/>
        Recommendations: {len(self.recommendations)}
        """
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Recommendations table
        table_data = [['Category', 'Service', 'Savings', 'Priority']]
        for rec in self.recommendations[:10]:  # Top 10
            table_data.append([
                rec['category'],
                rec['service'],
                f"${rec['estimated_savings']:,.0f}",
                rec['priority']
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        return output_file
