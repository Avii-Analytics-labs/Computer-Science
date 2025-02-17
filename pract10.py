import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Generate sample e-commerce dataset
np.random.seed(42)
n_records = 1000

# Generate dates for the past year
end_date = datetime(2024, 2, 17)
start_date = end_date - timedelta(days=365)
dates = [start_date + timedelta(days=x) for x in range(366)]

# Generate sales data
data = {
    'Date': np.random.choice(dates, n_records),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home & Garden'], n_records),
    'Sales_Amount': np.random.normal(500, 150, n_records),
    'Customer_Age': np.random.normal(35, 12, n_records),
    'Customer_Gender': np.random.choice(['Male', 'Female'], n_records),
    'Customer_Location': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'Customer_Type': np.random.choice(['New', 'Returning'], n_records, p=[0.3, 0.7]),
}

# Create DataFrame
df = pd.DataFrame(data)

# Clean and prepare data
df['Sales_Amount'] = df['Sales_Amount'].clip(min=0)
df['Customer_Age'] = df['Customer_Age'].clip(min=18, max=90).astype(int)
df['Month'] = df['Date'].dt.strftime('%Y-%m')

def create_sales_analysis(df):
    """
    Create comprehensive sales analysis visualizations
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Monthly Sales Trend
    ax1 = plt.subplot(321)
    monthly_sales = df.groupby('Month')['Sales_Amount'].sum().reset_index()
    sns.lineplot(data=monthly_sales, x='Month', y='Sales_Amount', marker='o')
    plt.xticks(rotation=45)
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales ($)')
    
    # 2. Sales by Category
    ax2 = plt.subplot(322)
    category_sales = df.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=True)
    category_sales.plot(kind='barh')
    plt.title('Sales by Product Category')
    plt.xlabel('Total Sales ($)')
    
    # 3. Customer Age Distribution by Gender
    ax3 = plt.subplot(323)
    sns.boxplot(data=df, x='Customer_Gender', y='Customer_Age', hue='Customer_Type')
    plt.title('Customer Age Distribution by Gender and Type')
    
    # 4. Sales Distribution by Location
    ax4 = plt.subplot(324)
    location_stats = df.groupby('Customer_Location').agg({
        'Sales_Amount': ['mean', 'count']
    }).reset_index()
    location_stats.columns = ['Location', 'Avg_Sales', 'Count']
    
    sns.scatterplot(data=location_stats, x='Count', y='Avg_Sales', s=200)
    for i, row in location_stats.iterrows():
        plt.annotate(row['Location'], (row['Count'], row['Avg_Sales']))
    plt.title('Sales Distribution by Location')
    plt.xlabel('Number of Customers')
    plt.ylabel('Average Sales ($)')
    
    # 5. Customer Type Analysis
    ax5 = plt.subplot(325)
    customer_type_sales = df.groupby(['Customer_Type', 'Product_Category'])['Sales_Amount'].mean().unstack()
    sns.heatmap(customer_type_sales, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Average Sales by Customer Type and Product Category')
    
    # 6. Age Group Analysis
    ax6 = plt.subplot(326)
    df['Age_Group'] = pd.cut(df['Customer_Age'], 
                            bins=[0, 25, 35, 45, 55, 100],
                            labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    age_category_sales = df.groupby(['Age_Group', 'Product_Category'])['Sales_Amount'].mean().unstack()
    sns.heatmap(age_category_sales, annot=True, fmt='.0f', cmap='viridis')
    plt.title('Average Sales by Age Group and Product Category')
    
    plt.tight_layout()
    plt.show()
    
    return monthly_sales, category_sales, location_stats

def generate_insights_report(monthly_sales, category_sales, location_stats):
    """
    Generate a comprehensive insights report
    """
    print("\nKey Business Insights Report")
    print("=" * 50)
    
    # Sales Trends
    print("\n1. Sales Trends:")
    print(f"- Total Sales: ${df['Sales_Amount'].sum():,.2f}")
    print(f"- Average Order Value: ${df['Sales_Amount'].mean():,.2f}")
    print(f"- Best Performing Month: {monthly_sales.iloc[monthly_sales['Sales_Amount'].argmax()]['Month']}")
    
    # Product Categories
    print("\n2. Product Category Analysis:")
    for category, sales in category_sales.items():
        print(f"- {category}: ${sales:,.2f} ({(sales/category_sales.sum()*100):.1f}%)")
    
    # Customer Demographics
    print("\n3. Customer Demographics:")
    print(f"- Average Customer Age: {df['Customer_Age'].mean():.1f} years")
    print(f"- Gender Distribution: {(df['Customer_Gender'].value_counts(normalize=True)*100).round(1).to_dict()}")
    print(f"- New vs Returning Customers: {(df['Customer_Type'].value_counts(normalize=True)*100).round(1).to_dict()}")
    
    # Regional Performance
    print("\n4. Regional Performance:")
    for _, row in location_stats.iterrows():
        print(f"- {row['Location']}: {row['Count']} customers, ${row['Avg_Sales']:,.2f} avg. sales")
    
    # Key Recommendations
    print("\n5. Key Recommendations:")
    top_category = category_sales.index[-1]
    print(f"- Focus on expanding {top_category} category, which shows highest sales")
    best_location = location_stats.sort_values('Avg_Sales', ascending=False).iloc[0]
    print(f"- Investigate success factors in {best_location['Location']} region for replication")
    
    return None

# Create visualizations and generate insights
print("Creating sales analysis visualizations...")
monthly_sales, category_sales, location_stats = create_sales_analysis(df)

print("\nGenerating insights report...")
generate_insights_report(monthly_sales, category_sales, location_stats)

# Additional Analysis: Customer Segmentation
def analyze_customer_segments(df):
    """
    Create customer segmentation analysis
    """
    plt.figure(figsize=(15, 5))
    
    # Customer Value Segments
    df['Value_Segment'] = pd.qcut(df['Sales_Amount'], q=3, labels=['Low', 'Medium', 'High'])
    
    # 1. Value Segment Distribution
    plt.subplot(131)
    df['Value_Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Customer Value Segment Distribution')
    
    # 2. Age Distribution by Value Segment
    plt.subplot(132)
    sns.boxplot(data=df, x='Value_Segment', y='Customer_Age')
    plt.title('Age Distribution by Value Segment')
    
    # 3. Value Segment by Location
    plt.subplot(133)
    segment_location = pd.crosstab(df['Value_Segment'], df['Customer_Location'], normalize='columns')
    segment_location.plot(kind='bar', stacked=True)
    plt.title('Value Segments by Location')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

print("\nAnalyzing customer segments...")
analyze_customer_segments(df)
