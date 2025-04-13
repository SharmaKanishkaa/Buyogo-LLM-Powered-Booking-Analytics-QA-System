import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HotelBookingPipeline:
    def __init__(self, data_path: str):
        """Initialize with data path and constants"""
        self.raw_data = pd.read_csv("C:/Users/Kanishka/Documents/Buyogo/hotel_bookings.csv")
        self.processed_data = None
        self.analytics = {}
        
        # Constants
        self.MEAL_MAP = {
            'BB': "Breakfast",
            'FB': "Full Board", 
            'HB': "Half Board",
            'SC': "No meal",
            'Undefined': "No meal"
        }
        self.COUNTRY_MAP = {
            'PRT': 'Portugal', 'GBR': 'UK', 'FRA': 'France',
            'ESP': 'Spain', 'DEU': 'Germany', 'ITA': 'Italy',
            'IRL': 'Ireland', 'BEL': 'Belgium', 'BRA': 'Brazil',
            'NLD': 'Netherlands'
        }
        self.MONTH_ORDER = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute full processing pipeline"""
        self._handle_missing_data()
        self._transform_features()
        self._calculate_derived_features()
        self._generate_analytics()
        self._generate_visualizations()

        self.analytics["raw_data"] = self.raw_data
        return self.analytics

    def _handle_missing_data(self) -> None:
        """Clean and impute missing values"""
        self.raw_data['agent'] = self.raw_data['agent'].fillna(0)
        self.raw_data['company'] = self.raw_data['company'].fillna(0)
        self.raw_data = self.raw_data.dropna(subset=['children'])
        self.raw_data['country'] = self.raw_data['country'].fillna('Unknown')

    def _transform_features(self) -> None:
        """Convert and enrich raw features"""
        self.raw_data["meal"] = self.raw_data["meal"].replace(self.MEAL_MAP).astype('category')
        
        # Create proper datetime field
        self.raw_data['arrival_date'] = pd.to_datetime(
            self.raw_data['arrival_date_year'].astype(str) + '-' +
            self.raw_data['arrival_date_month'] + '-' +
            self.raw_data['arrival_date_day_of_month'].astype(str))
        
        self.raw_data['reservation_status_date'] = pd.to_datetime(
            self.raw_data['reservation_status_date']
        )

    def _calculate_derived_features(self) -> None:
        """Create new calculated features"""
        df = self.raw_data
        df['total_guests'] = df['adults'] + df['children']
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['total_revenue'] = df['adr'] * df['total_nights']
        self.processed_data = df[df['total_guests'] > 0]

    def _generate_analytics(self) -> None:
        """Precompute key analytics"""
        df = self.processed_data
        lead_bins = [0, 7, 30, 90, 365, 737]
        lead_labels = ['0-7d', '7-30d', '30-90d', '90-365d', '365d+']
        df['lead_time_group'] = pd.cut(df['lead_time'], bins=lead_bins, labels=lead_labels)
        
        self.analytics = {
            'summary_stats': {
                'total_bookings': len(df),
                'cancellation_rate': df['is_canceled'].mean(),
                'avg_lead_time': df['lead_time'].mean()
            },
            'monthly_metrics': self._monthly_adr_analysis(df),
            'cancellation_analysis': {
                'by_country': df.groupby('country')['is_canceled'].mean().sort_values(ascending=False).head(10).to_dict(),
                'by_lead_time': df.groupby('lead_time_group')['is_canceled'].mean().to_dict()
            },
            'top_countries': df['country'].value_counts().head(10).to_dict(),
            'guest_distribution': df['total_guests'].value_counts().sort_index().to_dict()
        }

    def _monthly_adr_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly metrics"""
        monthly = df.groupby('arrival_date_month')['adr'].mean().reset_index()
        monthly['arrival_date_month'] = pd.Categorical(
            monthly['arrival_date_month'], 
            categories=self.MONTH_ORDER, 
            ordered=True
        )
        return {
            'monthly_adr': monthly.sort_values('arrival_date_month').set_index('arrival_date_month')['adr'].to_dict(),
            'monthly_revenue': df.groupby('arrival_date_month')['total_revenue'].sum().to_dict()
        }

    def _generate_visualizations(self) -> None:
        """Generate and save visualizations"""
        os.makedirs('static/visualizations', exist_ok=True)
        
        # Monthly ADR Plot
        plt.figure(figsize=(12, 6))
        monthly_data = pd.DataFrame({
            'month': list(self.analytics['monthly_metrics']['monthly_adr'].keys()),
            'adr': list(self.analytics['monthly_metrics']['monthly_adr'].values())
        })
        sns.barplot(data=monthly_data, x='month', y='adr', palette='viridis')
        plt.title('Average Daily Rate by Month')
        plt.xticks(rotation=45)
        plt.savefig('static/visualizations/monthly_adr.png')
        plt.close()
        
        # Cancellation Rates by Country
        plt.figure(figsize=(12, 6))
        cancel_data = pd.DataFrame({
            'country': [self.COUNTRY_MAP.get(k, k) for k in self.analytics['cancellation_analysis']['by_country'].keys()],
            'rate': list(self.analytics['cancellation_analysis']['by_country'].values())
        })
        sns.barplot(data=cancel_data, x='rate', y='country', palette='rocket')
        plt.title('Top Cancellation Rates by Country')
        plt.savefig('static/visualizations/cancellation_by_country.png')
        plt.close()

    def get_visualization_paths(self) -> Dict[str, str]:
        """Get paths to generated visualizations"""
        return {
            'monthly_adr': 'static/visualizations/monthly_adr.png',
            'cancellation_by_country': 'static/visualizations/cancellation_by_country.png'
        }