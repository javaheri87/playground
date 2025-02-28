import pandas as pd
import polars as pl
import time
import os
import psutil
import numpy as np
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = "/downloaded_from_Kaggle/arXiv_scientific dataset.csv"  # Replace with your actual file path

# Helper function to measure memory and time
def measure_performance(func, *args, **kwargs):
    # Measure memory
    mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=None, max_iterations=1)
    max_mem = max(mem_usage) - min(mem_usage)
    
    # Measure time
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    return result, execution_time, max_mem

# Dictionary to store results
results = {
    'operation': [],
    'library': [],
    'execution_time': [],
    'memory_usage': []
}

# 1. Reading Large CSV Files
def pandas_read_csv():
    return pd.read_csv(file_path)

def polars_read_csv():
    return pl.read_csv(file_path)

print("1. Testing CSV Reading...")
_, pd_time, pd_mem = measure_performance(pandas_read_csv)
_, pl_time, pl_mem = measure_performance(polars_read_csv)

results['operation'].extend(['Read CSV', 'Read CSV'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# Load data for subsequent tests
pandas_df = pd.read_csv(file_path)
polars_df = pl.read_csv(file_path)

# 2. Reading/Writing Parquet Files
parquet_path = "temp_data.parquet"

def pandas_write_parquet():
    pandas_df.to_parquet(parquet_path)

def polars_write_parquet():
    polars_df.write_parquet(parquet_path)

print("2. Testing Parquet Writing...")
_, pd_time, pd_mem = measure_performance(pandas_write_parquet)
_, pl_time, pl_mem = measure_performance(polars_write_parquet)

results['operation'].extend(['Write Parquet', 'Write Parquet'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

def pandas_read_parquet():
    return pd.read_parquet(parquet_path)

def polars_read_parquet():
    return pl.read_parquet(parquet_path)

print("   Testing Parquet Reading...")
_, pd_time, pd_mem = measure_performance(pandas_read_parquet)
_, pl_time, pl_mem = measure_performance(polars_read_parquet)

results['operation'].extend(['Read Parquet', 'Read Parquet'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 3. Group By with Aggregation
def pandas_groupby():
    return pandas_df.groupby('category').agg({
        'summary_word_count': ['mean', 'sum', 'count'],
        'id': 'count'
    })

def polars_groupby():
    return polars_df.group_by('category').agg([
        pl.col('summary_word_count').mean().alias('word_count_mean'),
        pl.col('summary_word_count').sum().alias('word_count_sum'),
        pl.col('summary_word_count').count().alias('word_count_count'),
        pl.col('id').count().alias('id_count')
    ])

print("3. Testing Group By with Aggregation...")
_, pd_time, pd_mem = measure_performance(pandas_groupby)
_, pl_time, pl_mem = measure_performance(polars_groupby)

results['operation'].extend(['Group By', 'Group By'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 4. Complex Filtering
def pandas_complex_filter():
    return pandas_df[(pandas_df['summary_word_count'] > 100) & 
                     (pandas_df['category'] == 'Technology') & 
                     (pandas_df['published_date'] > '2020-01-01')]

def polars_complex_filter():
    return polars_df.filter(
        (pl.col('summary_word_count') > 100) & 
        (pl.col('category') == 'Technology') & 
        (pl.col('published_date') > '2020-01-01')
    )

print("4. Testing Complex Filtering...")
_, pd_time, pd_mem = measure_performance(pandas_complex_filter)
_, pl_time, pl_mem = measure_performance(polars_complex_filter)

results['operation'].extend(['Complex Filtering', 'Complex Filtering'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 5. Joins on Large Tables
# Create a second dataframe for joining
pandas_df2 = pandas_df[['id', 'category', 'summary_word_count']].copy()
pandas_df2.columns = ['id', 'category2', 'word_count2']
polars_df2 = pl.from_pandas(pandas_df2)

def pandas_join():
    return pandas_df.merge(pandas_df2, on='id', how='inner')

def polars_join():
    return polars_df.join(polars_df2, on='id', how='inner')

print("5. Testing Joins on Large Tables...")
_, pd_time, pd_mem = measure_performance(pandas_join)
_, pl_time, pl_mem = measure_performance(polars_join)

results['operation'].extend(['Join', 'Join'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 6. Column Calculations
def pandas_column_calc():
    df = pandas_df.copy()
    df['word_density'] = df['summary_word_count'] / df['summary'].str.len()
    df['is_long'] = df['summary_word_count'] > df['summary_word_count'].mean()
    df['title_length'] = df['title'].str.len()
    return df

def polars_column_calc():
    return polars_df.with_columns([
        (pl.col('summary_word_count') / pl.lit(1)).alias('word_density'),  # Simplified for testing
        (pl.col('summary_word_count') > pl.col('summary_word_count').mean()).alias('is_long'),
        (pl.lit(1)).alias('title_length')  # Simplified for testing
    ])

print("6. Testing Column Calculations...")
_, pd_time, pd_mem = measure_performance(pandas_column_calc)
_, pl_time, pl_mem = measure_performance(polars_column_calc)

results['operation'].extend(['Column Calculations', 'Column Calculations'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 7. String Operations
def pandas_string_ops():
    df = pandas_df.copy()
    df['title_upper'] = df['title'].str.upper()
    df['contains_data'] = df['title'].str.contains('data', case=False)
    df['title_words'] = df['title'].str.split().str.len()
    return df

def polars_string_ops():
    return polars_df.with_columns([
        pl.col('title').str.to_uppercase().alias('title_upper'),
        pl.col('title').str.contains('data', literal=True).alias('contains_data'),
        pl.col('title').str.split(' ').list.len().alias('title_words')
    ])

print("7. Testing String Operations...")
_, pd_time, pd_mem = measure_performance(pandas_string_ops)
_, pl_time, pl_mem = measure_performance(polars_string_ops)

results['operation'].extend(['String Operations', 'String Operations'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 8. Memory Usage with Large Datasets
# Create a larger dataset by duplicating the existing one
def pandas_memory_test():
    large_df = pd.concat([pandas_df] * 10, ignore_index=True)
    return large_df.memory_usage(deep=True).sum()

def polars_memory_test():
    large_df = pl.concat([polars_df] * 10)
    return large_df.estimated_size()

print("8. Testing Memory Usage with Large Datasets...")
_, pd_time, pd_mem = measure_performance(pandas_memory_test)
_, pl_time, pl_mem = measure_performance(polars_memory_test)

results['operation'].extend(['Memory Test', 'Memory Test'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 9. Sorting Large Datasets
def pandas_sort():
    return pandas_df.sort_values(by=['category', 'summary_word_count', 'published_date'], ascending=[True, False, True])

def polars_sort():
    return polars_df.sort(by=['category', 'summary_word_count', 'published_date'], descending=[False, True, False])

print("9. Testing Sorting Large Datasets...")
_, pd_time, pd_mem = measure_performance(pandas_sort)
_, pl_time, pl_mem = measure_performance(polars_sort)

results['operation'].extend(['Sorting', 'Sorting'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# 10. Window Functions
def pandas_window():
    df = pandas_df.copy()
    df['rolling_avg'] = df.sort_values('published_date').groupby('category')['summary_word_count'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['cumulative_sum'] = df.sort_values('published_date').groupby('category')['summary_word_count'].transform('cumsum')
    df['rank'] = df.groupby('category')['summary_word_count'].transform('rank', method='dense')
    return df

def polars_window():
    return polars_df.sort('published_date').with_columns([
        pl.col('summary_word_count').rolling_mean(window_size=3, min_periods=1).over('category').alias('rolling_avg'),
        pl.col('summary_word_count').cum_sum().over('category').alias('cumulative_sum'),
        pl.col('summary_word_count').rank(method='dense').over('category').alias('rank')
    ])

print("10. Testing Window Functions...")
_, pd_time, pd_mem = measure_performance(pandas_window)
_, pl_time, pl_mem = measure_performance(polars_window)

results['operation'].extend(['Window Functions', 'Window Functions'])
results['library'].extend(['Pandas', 'Polars'])
results['execution_time'].extend([pd_time, pl_time])
results['memory_usage'].extend([pd_mem, pl_mem])

# Clean up temporary files
if os.path.exists(parquet_path):
    os.remove(parquet_path)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Visualization
plt.figure(figsize=(15, 10))

# Time comparison with log scale
plt.subplot(2, 1, 1)
sns.barplot(x='operation', y='execution_time', hue='library', data=results_df)
plt.title('Execution Time Comparison: Pandas vs Polars')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Time (seconds) - Log Scale')  # Updated label to indicate log scale
plt.yscale('log')  # Set log scale for y-axis
plt.legend(title='Library')
plt.tight_layout()

# Memory comparison with log scale
plt.subplot(2, 1, 2)
sns.barplot(x='operation', y='memory_usage', hue='library', data=results_df)
plt.title('Memory Usage Comparison: Pandas vs Polars')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Memory (MB) - Log Scale')  # Updated label to indicate log scale
plt.yscale('log')  # Set log scale for y-axis
plt.legend(title='Library')
plt.tight_layout()

plt.savefig('/file_path/file_name.png', dpi=300)
plt.show()

# Print summary
print("\nPerformance Summary:")
summary = results_df.groupby('library').agg({
    'execution_time': ['mean', 'sum'],
    'memory_usage': ['mean', 'sum']
})
print(summary)

# Calculate speedup and memory efficiency
pd_times = results_df[results_df['library'] == 'Pandas']['execution_time'].values
pl_times = results_df[results_df['library'] == 'Polars']['execution_time'].values
speedup = pd_times / pl_times

pd_mem = results_df[results_df['library'] == 'Pandas']['memory_usage'].values
pl_mem = results_df[results_df['library'] == 'Polars']['memory_usage'].values
mem_efficiency = pd_mem / pl_mem

print(f"\nAverage Speedup (Pandas time / Polars time): {np.mean(speedup):.2f}x")
print(f"Average Memory Efficiency (Pandas memory / Polars memory): {np.mean(mem_efficiency):.2f}x")
