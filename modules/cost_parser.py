"""
Cost Data Parser - Handles multiple file formats
"""

import pandas as pd
import re
import json
from datetime import datetime


class CostDataParser:
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'pdf', 'json']
    
    def parse_csv(self, df):
        """Parse CSV/Excel data"""
        # Normalize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Find cost column
        cost_col = self._find_column(df, ['cost', 'charge', 'amount', 'total'])
        service_col = self._find_column(df, ['service', 'product', 'product_name'])
        date_col = self._find_column(df, ['date', 'usage_date', 'billing_period'])
        
        # Parse dates if present
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        return self._structure_data(df, cost_col, service_col, date_col)
    
    def parse_pdf_text(self, text):
        """Parse PDF text content"""
        # Extract cost information using regex
        cost_pattern = r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        service_pattern = r'(EC2|RDS|S3|EBS|Lambda|CloudFront|ELB|VPC|CloudWatch|DynamoDB)'
        
        costs = re.findall(cost_pattern, text)
        services = re.findall(service_pattern, text, re.IGNORECASE)
        
        # Create structured data
        data = {
            'raw_text': text,
            'extracted_costs': [float(c.replace(',', '')) for c in costs],
            'extracted_services': list(set(services)),
            'summary': {
                'total_cost': sum(float(c.replace(',', '')) for c in costs) if costs else 0,
                'num_services': len(set(services))
            }
        }
        
        # Try to create service breakdown
        by_service = {}
        for service in set(services):
            # Find costs near service mentions
            service_costs = []
            for match in re.finditer(service, text, re.IGNORECASE):
                pos = match.start()
                nearby_text = text[max(0, pos-100):min(len(text), pos+100)]
                nearby_costs = re.findall(cost_pattern, nearby_text)
                if nearby_costs:
                    service_costs.extend([float(c.replace(',', '')) for c in nearby_costs])
            
            if service_costs:
                by_service[service] = sum(service_costs) / len(service_costs)
        
        data['by_service'] = by_service
        return data
    
    def parse_json(self, json_data):
        """Parse JSON data"""
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
            return self.parse_csv(df)
        elif isinstance(json_data, dict):
            # Handle nested JSON
            if 'ResultsByTime' in json_data:  # AWS Cost Explorer format
                return self._parse_aws_cost_explorer_json(json_data)
            else:
                # Generic JSON
                df = pd.json_normalize(json_data)
                return self.parse_csv(df)
    
    def _parse_aws_cost_explorer_json(self, data):
        """Parse AWS Cost Explorer API response"""
        results = data.get('ResultsByTime', [])
        
        records = []
        for result in results:
            period_start = result.get('TimePeriod', {}).get('Start')
            groups = result.get('Groups', [])
            
            for group in groups:
                service = group.get('Keys', ['Unknown'])[0]
                amount = float(group.get('Metrics', {}).get('UnblendedCost', {}).get('Amount', 0))
                
                records.append({
                    'date': period_start,
                    'service': service,
                    'cost': amount
                })
        
        df = pd.DataFrame(records)
        return self.parse_csv(df)
    
    def merge_datasets(self, datasets):
        """Merge multiple parsed datasets"""
        merged = {
            'summary': {'total_cost': 0, 'num_records': 0},
            'by_service': {},
            'time_series': [],
            'raw_data': []
        }
        
        for data in datasets:
            # Merge summaries
            if 'summary' in data:
                merged['summary']['total_cost'] += data['summary'].get('total_cost', 0)
                merged['summary']['num_records'] += data['summary'].get('num_records', 0)
            
            # Merge service costs
            if 'by_service' in data:
                for service, cost in data['by_service'].items():
                    merged['by_service'][service] = merged['by_service'].get(service, 0) + cost
            
            # Merge time series
            if 'time_series' in data:
                merged['time_series'].extend(data['time_series'])
            
            # Merge raw data
            if 'raw_data' in data and hasattr(data['raw_data'], 'to_dict'):
                merged['raw_data'].append(data['raw_data'])
        
        # Calculate averages
        if merged['summary']['num_records'] > 0:
            merged['summary']['avg_daily_cost'] = merged['summary']['total_cost'] / merged['summary']['num_records']
        
        return merged
    
    def prepare_text_for_rag(self, cost_data, recommendations):
        """Prepare text content for RAG analysis"""
        text_parts = []
        
        # Summary
        text_parts.append(f"COST SUMMARY\n")
        text_parts.append(f"Total Cost: ${cost_data['summary']['total_cost']:,.2f}\n")
        
        # Services
        text_parts.append(f"\nSERVICES BREAKDOWN\n")
        for service, cost in sorted(cost_data['by_service'].items(), key=lambda x: x[1], reverse=True):
            text_parts.append(f"{service}: ${cost:,.2f}\n")
        
        # Recommendations
        text_parts.append(f"\nOPTIMIZATION RECOMMENDATIONS\n")
        for i, rec in enumerate(recommendations, 1):
            text_parts.append(f"{i}. {rec['category']} - {rec['service']}\n")
            text_parts.append(f"   Savings: ${rec['estimated_savings']:,.2f}\n")
            text_parts.append(f"   Description: {rec['description']}\n")
            text_parts.append(f"   Action: {rec['action']}\n\n")
        
        return "".join(text_parts)
    
    def _find_column(self, df, possible_names):
        """Find column by possible names"""
        for col in df.columns:
            if any(name in col.lower() for name in possible_names):
                return col
        return None
    
    def _structure_data(self, df, cost_col, service_col, date_col):
        """Structure parsed data"""
        data = {
            'raw_data': df,
            'summary': {},
            'by_service': {},
            'time_series': []
        }
        
        if cost_col:
            data['summary'] = {
                'total_cost': df[cost_col].sum(),
                'avg_daily_cost': df[cost_col].mean(),
                'max_daily_cost': df[cost_col].max(),
                'min_daily_cost': df[cost_col].min(),
                'num_records': len(df)
            }
        
        if service_col and cost_col:
            data['by_service'] = df.groupby(service_col)[cost_col].sum().to_dict()
        
        if date_col and cost_col:
            time_series = df.groupby(date_col)[cost_col].sum().reset_index()
            data['time_series'] = time_series.to_dict('records')
        
        return data
