"""
Recommendation Engine - Generates cost optimization recommendations
"""

import numpy as np


class RecommendationEngine:
    def __init__(self, cost_data, threshold=10.0):
        self.cost_data = cost_data
        self.threshold = threshold
        self.recommendations = []
    
    def generate_recommendations(self):
        """Generate all recommendations"""
        self.recommendations = []
        
        self._analyze_rightsizing()
        self._analyze_reserved_instances()
        self._analyze_scheduling()
        self._analyze_storage()
        self._analyze_idle_resources()
        self._analyze_data_transfer()
        self._detect_anomalies()
        
        # Filter and sort
        self.recommendations = [r for r in self.recommendations if r['estimated_savings'] >= self.threshold]
        self.recommendations.sort(key=lambda x: x['estimated_savings'], reverse=True)
        
        return self.recommendations
    
    def _analyze_rightsizing(self):
        """EC2/RDS rightsizing opportunities"""
        by_service = self.cost_data.get('by_service', {})
        
        compute_services = {
            'EC2': 0.25,
            'Amazon Elastic Compute Cloud': 0.25,
            'RDS': 0.20,
            'Amazon Relational Database Service': 0.20
        }
        
        for service, savings_rate in compute_services.items():
            cost = by_service.get(service, 0)
            if cost > 100:
                savings = cost * savings_rate
                self.recommendations.append({
                    'category': 'Rightsizing',
                    'service': service.split()[0] if ' ' in service else service,
                    'description': f'Rightsize overprovisioned {service.split()[0]} instances',
                    'estimated_savings': savings,
                    'priority': 'High',
                    'effort': 'Medium',
                    'action': 'Review utilization metrics and downsize instances with <40% CPU/memory usage'
                })
    
    def _analyze_reserved_instances(self):
        """Reserved instance opportunities"""
        by_service = self.cost_data.get('by_service', {})
        
        ri_opportunities = [
            ('EC2', ['EC2', 'Amazon Elastic Compute Cloud'], 0.35),
            ('RDS', ['RDS', 'Amazon Relational Database Service'], 0.40),
            ('ElastiCache', ['ElastiCache', 'Amazon ElastiCache'], 0.35),
        ]
        
        for name, service_keys, savings_rate in ri_opportunities:
            cost = sum(by_service.get(key, 0) for key in service_keys)
            if cost > 150:
                savings = cost * savings_rate
                self.recommendations.append({
                    'category': 'Reserved Instances',
                    'service': name,
                    'description': f'Purchase 1-year Reserved Instances for {name}',
                    'estimated_savings': savings,
                    'priority': 'High',
                    'effort': 'Low',
                    'action': f'Analyze usage patterns and purchase RIs for steady-state {name} workloads'
                })
    
    def _analyze_scheduling(self):
        """Scheduling opportunities"""
        by_service = self.cost_data.get('by_service', {})
        
        total_compute = sum(
            cost for service, cost in by_service.items()
            if any(s in service.lower() for s in ['ec2', 'compute', 'rds', 'database'])
        )
        
        if total_compute > 100:
            # 30% dev/test, 65% savings by running only business hours
            savings = total_compute * 0.30 * 0.65
            self.recommendations.append({
                'category': 'Scheduling',
                'service': 'EC2/RDS',
                'description': 'Schedule non-production resources for business hours only',
                'estimated_savings': savings,
                'priority': 'Medium',
                'effort': 'Low',
                'action': 'Use AWS Instance Scheduler or Lambda to stop dev/test instances outside 9am-6pm'
            })
    
    def _analyze_storage(self):
        """Storage optimization"""
        by_service = self.cost_data.get('by_service', {})
        
        # S3 lifecycle policies
        s3_cost = sum(by_service.get(key, 0) for key in ['S3', 'Amazon Simple Storage Service'])
        if s3_cost > 50:
            savings = s3_cost * 0.30
            self.recommendations.append({
                'category': 'Storage Optimization',
                'service': 'S3',
                'description': 'Implement S3 lifecycle policies for infrequent data',
                'estimated_savings': savings,
                'priority': 'Medium',
                'effort': 'Low',
                'action': 'Move data >90 days old to S3-IA, >180 days to Glacier'
            })
        
        # EBS optimization
        ebs_cost = sum(by_service.get(key, 0) for key in ['EBS', 'Amazon Elastic Block Store'])
        if ebs_cost > 30:
            savings = ebs_cost * 0.25
            self.recommendations.append({
                'category': 'Storage Optimization',
                'service': 'EBS',
                'description': 'Clean up unattached EBS volumes and old snapshots',
                'estimated_savings': savings,
                'priority': 'High',
                'effort': 'Low',
                'action': 'Delete unattached volumes and snapshots older than 90 days'
            })
    
    def _analyze_idle_resources(self):
        """Detect idle resources"""
        by_service = self.cost_data.get('by_service', {})
        
        idle_checks = [
            (['ELB', 'Elastic Load Balancing'], 'Load Balancers', 0.40),
            (['NAT Gateway', 'VPC'], 'NAT Gateways', 0.30),
            (['Elastic IP'], 'Elastic IPs', 0.50),
        ]
        
        for service_keys, name, savings_rate in idle_checks:
            cost = sum(by_service.get(key, 0) for key in service_keys)
            if cost > 20:
                savings = cost * savings_rate
                self.recommendations.append({
                    'category': 'Idle Resources',
                    'service': name,
                    'description': f'Remove idle {name} with no traffic',
                    'estimated_savings': savings,
                    'priority': 'High',
                    'effort': 'Low',
                    'action': f'Audit and delete unused {name}'
                })
    
    def _analyze_data_transfer(self):
        """Data transfer optimization"""
        by_service = self.cost_data.get('by_service', {})
        
        transfer_cost = sum(
            cost for service, cost in by_service.items()
            if any(s in service.lower() for s in ['data transfer', 'cloudfront', 'bandwidth'])
        )
        
        if transfer_cost > 100:
            savings = transfer_cost * 0.20
            self.recommendations.append({
                'category': 'Data Transfer',
                'service': 'Network',
                'description': 'Optimize data transfer and use CloudFront CDN',
                'estimated_savings': savings,
                'priority': 'Medium',
                'effort': 'Medium',
                'action': 'Enable CloudFront caching and review inter-region data transfer'
            })
    
    def _detect_anomalies(self):
        """Detect cost anomalies"""
        time_series = self.cost_data.get('time_series', [])
        
        if len(time_series) > 7:
            costs = [record.get('cost', 0) for record in time_series]
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            
            recent_costs = costs[-3:]
            if any(cost > mean_cost + 2 * std_cost for cost in recent_costs):
                spike_amount = max(recent_costs) - mean_cost
                self.recommendations.append({
                    'category': 'Anomaly Detection',
                    'service': 'Multiple',
                    'description': 'Unusual cost spike detected in recent period',
                    'estimated_savings': spike_amount * 0.5,
                    'priority': 'Critical',
                    'effort': 'High',
                    'action': 'Investigate recent cost spike and identify root cause immediately'
                })
