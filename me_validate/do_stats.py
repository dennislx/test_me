"""
Restructured Statistics Module

This module calculates columnwise descriptive statistics and makes comparisons
between PCDS and AWS data sources. Refactored for better modularity, testability,
and maintainability.
"""

import os
import re
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from loguru import logger
from confection import Config
from tqdm import tqdm

import utils
import utils_type as ut

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning, message='.*pandas only supports SQLAlchemy connectable.*')
filterwarnings("ignore", category=FutureWarning, message='.*concatenation with empty or all-NA entries is deprecated.*')
filterwarnings("ignore", category=FutureWarning, message='.*Call result\.infer\_objects\(copy\=False\) instead.*')


# ===============================
# Constants and Configuration
# ===============================

SEP = '; '
WHERE_SQL = r"\n.+?where .* not in \(.*\)"


# ===============================
# Data Classes
# ===============================

@dataclass
class ComparisonResult:
    """Result of comparing PCDS and AWS statistics."""
    has_mismatch: bool = False
    mismatched_columns: Set[str] = field(default_factory=set)
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV output."""
        return {
            'Column Stats UnMatch': self.has_mismatch,
            'Stats UnMatch Details': self.details or '; '.join(self.mismatched_columns)
        }


@dataclass
class PartitionStatistics:
    """Statistics for a single partition."""
    pcds_stats: pd.DataFrame
    pcds_mismatches: List[str]
    pcds_name: str
    aws_stats: pd.DataFrame
    aws_mismatches: List[str]
    aws_name: str


# ===============================
# Utility Functions
# ===============================

class DateUtils:
    """Utilities for date handling and processing."""
    
    @staticmethod
    def extend_excludes(exclude_list: List[str], until_date: str) -> pd.DatetimeIndex:
        """Extend exclude list with dates from until_date to now."""
        days = pd.date_range(pd.to_datetime(until_date), dt.datetime.now(), freq='d')
        combined = list(set(exclude_list) | set(days))
        return pd.to_datetime([x for x in combined if x])
    
    @staticmethod
    def get_aws_until_condition(date_var: str, date_type: str, date_until: str) -> List[str]:
        """Generate AWS until condition for date filtering."""
        if date_type.startswith('date') or date_type.startswith('time'):
            date_var = f"DATE_FORMAT({date_var}, '%Y-%m-%d')"
        return [f"{date_var} <= '{date_until}'"]
    
    @staticmethod
    def get_aws_where_condition(date_var: str, date_type: str, date_partition: str) -> str:
        """Generate AWS where condition based on partition."""
        if date_partition == 'full':
            return ''
        
        _date_var, date_range = date_partition.split('=')
        assert date_var == _date_var, "Date Variable Should Match"
        
        if 'char' in date_type:
            date_var = f"DATE_PARSE({date_var}, '%Y-%m-%d')"
        
        if re.match(r'\d{4}$', date_range):
            return f"DATE_FORMAT({date_var}, '%Y') = '{date_range}'"
        elif re.match(r'\d{4}-\d{2}$', date_range):
            return f"DATE_FORMAT({date_var}, '%Y-%m') = '{date_range}'"
        
        return ''


class ValidationUtils:
    """Utilities for data validation."""
    
    @staticmethod
    def get_duplicates(items: List[str]) -> List[str]:
        """Get duplicate items from a list."""
        return [item for item, count in Counter(items).items() if count > 1]
    
    @staticmethod
    def validate_column_mappings(meta_info: ut.MetaJSON) -> bool:
        """Validate that column mappings exist."""
        return meta_info.aws.column_mapping != {'': ''}


# ===============================
# AWS Query Handler
# ===============================

class AWSQueryHandler:
    """Handles AWS Athena queries with caching."""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
    
    def query_athena(self, columns: str, database: str, table: str, 
                    where_conditions: List[str], s3_cache: str) -> pd.DataFrame:
        """Execute Athena query with optional caching."""
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        aws_query = f"""
            SELECT {columns} 
            FROM {database}.{table} 
            {where_clause}
        """.strip()
        
        if self.use_cache and utils.s3_exist(s3_cache):
            logger.info(f"\t\t\tAWS data is already stored in {s3_cache}")
            return utils.s3_load_df(s3_cache)
        
        logger.info(f"\t\t\tStart pulling AWS data from {database}.{table}")
        df_aws = utils.SQLengine('AWS').execute(aws_query)
        utils.s3_save_df(df_aws, s3_cache)
        return df_aws


# ===============================
# Statistics Calculator
# ===============================

class StatsDataframe:
    """Calculates descriptive statistics for dataframes."""
    
    # Type mappings for different platforms
    TYPE_MAPPINGS = {
        'continuous': {
            'PCDS': ['NUMBER'],
            'AWS': ['decimal', 'double', 'bigint']
        },
        'categorical': {
            'PCDS': ['VARCHAR', 'CHAR'],
            'AWS': ['varchar']
        },
        'date': {
            'PCDS': ['DATE', 'TIMESTAMP'],
            'AWS': ['date', 'timestamp']
        }
    }
    
    DATE_FORMAT = '%Y-%m-%d'
    
    def __init__(self, platform: str, col2type: Dict[str, str], index_column: str):
        """Initialize statistics calculator for a platform."""
        self.platform = platform
        self.col2type = utils.UDict(col2type)
        self.index_column = index_column
        
        # Categorize columns by type
        self.continuous_cols = []
        self.categorical_cols = []
        self.date_cols = []
        self.unknown_types = []
        self.index_col = ""
        
        self._categorize_columns()
    
    def _categorize_columns(self):
        """Categorize columns based on their data types."""
        for var, data_type in self.col2type.items():
            if var.lower() == self.index_column.lower():
                self.index_col = var
            
            if self._is_type(data_type, 'categorical'):
                self.categorical_cols.append(var)
            elif self._is_type(data_type, 'continuous'):
                self.continuous_cols.append(var)
            elif self._is_type(data_type, 'date'):
                self.date_cols.append(var)
            else:
                self.unknown_types.append(data_type)
    
    def _is_type(self, data_type: str, type_category: str) -> bool:
        """Check if data type belongs to a category."""
        type_list = self.TYPE_MAPPINGS[type_category][self.platform]
        return any(data_type.startswith(t) for t in type_list)
    
    def transform(self, df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Transform dataframe and calculate statistics."""
        # Process date columns
        df_date = df[self.date_cols].apply(
            lambda c: pd.to_datetime(c, errors='coerce').dt.strftime(self.DATE_FORMAT)
        )
        
        # Process categorical columns
        df_categorical = df[self.categorical_cols].apply(lambda c: c.str.strip())
        
        # Process continuous columns
        df_continuous = df[self.continuous_cols]
        
        # Handle missing values based on configuration
        if hasattr(utils, 'config') and getattr(utils.config.output, 'drop_na', False):
            df_categorical = df_categorical.fillna('NULL')
            df_continuous = df_continuous.fillna(0)
        
        # Combine categorical and date data
        df_categorical = pd.concat([df_date, df_categorical], axis=1)
        
        # Calculate statistics
        stat_categorical = self._describe_categorical(df_categorical)
        stat_continuous = self._describe_continuous(df_continuous)
        
        # Combine statistics
        df_stats = pd.concat([stat_categorical, stat_continuous])
        df_stats['Type'] = df_stats.index.map(self.col2type)
        
        # Apply column mapping if provided
        if column_map is not None:
            df_stats.index = df_stats.index.map(column_map)
        
        return df_stats
    
    @staticmethod
    def _describe_continuous(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate descriptive statistics for continuous variables."""
        description = pd.DataFrame()
        description['N_Total'] = df.apply(len, axis=0)
        description['N_Unique'] = df.apply(lambda s: s.nunique(), axis=0)
        description['N_Missing'] = description['N_Total'] - df.count(axis=0)
        description['Min'] = df.apply(np.min, axis=0)
        description['Max'] = df.apply(np.max, axis=0)
        description['Mean'] = df.apply(lambda x: x.astype('float').mean(), axis=0)
        description['Std'] = df.apply(lambda x: x.astype('float').std(), axis=0)
        description['Freq'] = pd.NA
        description = description.join(StatsDataframe._describe_quantiles(df))
        return description
    
    @staticmethod
    def _describe_quantiles(df: pd.DataFrame, 
                           quantiles: List[float] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) -> pd.DataFrame:
        """Calculate quantile statistics."""
        column_names = [f'P_{int(100*x):02}' for x in quantiles]
        quantile_df = df.astype('float').quantile(quantiles)
        
        if isinstance(quantile_df, pd.Series):
            quantile_df.index = column_names
        else:
            quantile_df = quantile_df.T
            quantile_df.columns = column_names
        
        return quantile_df
    
    @staticmethod
    def _describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate descriptive statistics for categorical variables."""
        description = pd.DataFrame()
        description['N_Total'] = df.apply(len, axis=0)
        description['N_Unique'] = df.apply(lambda s: len(s.unique()), axis=0)
        description['N_Missing'] = description['N_Total'] - df.count(axis=0)
        
        stats = {}
        for col in df:
            freq = df[col].value_counts(dropna=False)
            stats[col] = {
                'Min': freq.min() if len(freq) > 0 else np.nan,
                'Max': freq.max() if len(freq) > 0 else np.nan,
                'Mean': freq.mean() if len(freq) > 0 else np.nan,
                'Std': freq.std() if len(freq) > 0 else np.nan,
                'Freq': freq.sort_values().to_dict() if len(freq) > 0 else {},
            }
            if len(freq) > 0:
                stats[col].update(**StatsDataframe._describe_quantiles(freq).to_dict())
        
        description = description.join(pd.DataFrame(stats).T)
        return description
    
    def __repr__(self) -> str:
        """String representation of the statistics calculator."""
        return (
            f'{self.platform}\n'
            f'  Continuous:  {"; ".join(self.continuous_cols)}\n'
            f'  Categorical: {"; ".join(self.categorical_cols)}\n'
            f'  Date(Cat):   {"; ".join(self.date_cols)}\n'
            f'  Unknown: {"; ".join(self.unknown_types)}'
        )


# ===============================
# Statistics Comparator
# ===============================

class StatisticsComparator:
    """Compares statistics between PCDS and AWS data."""
    
    @staticmethod
    def compare_statistics(df_pcds: pd.DataFrame, df_aws: pd.DataFrame) -> ComparisonResult:
        """Compare PCDS and AWS statistics and identify mismatches."""
        def custom_diff(a, b) -> bool:
            """Custom difference function handling NaN and different data types."""
            if pd.isna(a) and pd.isna(b):
                return False
            if pd.isna(a) ^ pd.isna(b):
                return True
            if isinstance(a, dict) and isinstance(b, dict):
                return a != b
            try:
                a_float, b_float = float(a), float(b)
                return not np.isclose(a_float, b_float, atol=1e-6)
            except (ValueError, TypeError):
                return a != b
        
        def format_dict_diff(dict_a: dict, dict_b: dict) -> str:
            """Format dictionary differences for display."""
            result = '; '.join(
                f'{key}({dict_a.get(key)})' 
                for key in dict_a 
                if dict_a.get(key) != dict_b.get(key)
            )
            return 'Different Encoding' if len(result) > 10_000 else result
        
        compare_fn = np.vectorize(custom_diff)
        mismatched_columns = set()
        
        # Create mask of differences
        mask = pd.DataFrame(
            compare_fn(df_aws.values, df_pcds.values),
            index=df_aws.index, 
            columns=df_aws.columns
        )
        
        # Process mismatches
        for column, row in mask.iterrows():
            for stat, has_difference in row.items():
                if stat == 'Type':
                    continue
                
                if stat == 'Freq' and has_difference:
                    StatisticsComparator._handle_frequency_difference(
                        df_pcds, df_aws, column, stat
                    )
                elif not has_difference and stat == 'Freq':
                    # No difference, clear freq display
                    df_pcds.loc[column, 'Freq'] = ''
                    df_aws.loc[column, 'Freq'] = ''
                
                if has_difference:
                    if stat != "Freq":
                        logger.warning(
                            f'Mismatch on {stat}({column}) where '
                            f'PCDS ({df_pcds.loc[column, "Type"]}) has {df_pcds.loc[column, stat]} and '
                            f'AWS ({df_aws.loc[column, "Type"]}) has {df_aws.loc[column, stat]}'
                        )
                    mismatched_columns.add(column)
        
        return ComparisonResult(
            has_mismatch=len(mismatched_columns) > 0,
            mismatched_columns=mismatched_columns
        )
    
    @staticmethod
    def _handle_frequency_difference(df_pcds: pd.DataFrame, df_aws: pd.DataFrame, 
                                   column: str, stat: str):
        """Handle frequency column differences."""
        def format_dict_diff(dict_a: dict, dict_b: dict) -> str:
            result = '; '.join(
                f'{key}({dict_a.get(key)})' 
                for key in dict_a 
                if dict_a.get(key) != dict_b.get(key)
            )
            return 'Different Encoding' if len(result) > 10_000 else result
        
        pcds_freq = df_pcds.loc[column, stat]
        aws_freq = df_aws.loc[column, stat]
        
        if not pd.isna(pcds_freq) and not pd.isna(aws_freq):
            df_pcds.loc[column, 'Freq'] = format_dict_diff(pcds_freq, aws_freq)
            df_aws.loc[column, 'Freq'] = format_dict_diff(aws_freq, pcds_freq)
        elif pd.isna(pcds_freq):
            df_pcds.loc[column, 'Freq'] = 'NULL'
            df_aws.loc[column, 'Freq'] = ';'.join(f'{k}({v})' for k, v in aws_freq.items())
        elif pd.isna(aws_freq):
            df_pcds.loc[column, 'Freq'] = ';'.join(f'{k}({v})' for k, v in pcds_freq.items())
            df_aws.loc[column, 'Freq'] = 'NULL'


# ===============================
# File Operations
# ===============================

class FileManager:
    """Manages file operations and S3 interactions."""
    
    def __init__(self, config: ut.StatConfig):
        self.config = config
    
    def upload_results_to_s3(self):
        """Upload results to S3."""
        s3_root = utils.urljoin(f'{self.config.output.s3_config.run}/', self.config.input.name)
        
        for root, _, files in os.walk(self.config.output.folder):
            for file in files:
                if file.startswith(self.config.input.step):
                    s3_url = utils.urljoin(f'{s3_root}/', file)
                    local_path = os.path.join(root, file)
                    utils.s3_upload(local_path, s3_url)
    
    def download_meta_files(self):
        """Download meta files if they don't exist locally."""
        meta_file = os.path.join(self.config.output.folder, 'step_meta.json')
        if not os.path.exists(meta_file):
            utils.download_froms3(
                utils.urljoin(f'{self.config.output.s3_config.run}/', self.config.input.name),
                self.config.output.folder
            )
    
    def get_s3_key(self, s3_url: str, partition: str) -> str:
        """Generate S3 key for caching."""
        s3_url = s3_url.lstrip(f'{self.config.output.s3_config.data}/')
        s3_url, _ = os.path.splitext(s3_url)
        
        if partition == 'full':
            return f'{s3_url}_None'
        
        return '_'.join(s3_url.split('/'))


# ===============================
# Main Statistics Processor
# ===============================

class StatisticsProcessor:
    """Main processor for statistics comparison workflow."""
    
    def __init__(self, config: ut.StatConfig):
        self.config = config
        self.file_manager = FileManager(config)
        self.query_handler = AWSQueryHandler(config.output.reuse_aws_data)
        self.stats_data = defaultdict(dict)
    
    def process_all_tables(self) -> Dict[str, Dict[str, PartitionStatistics]]:
        """Process all tables and calculate statistics."""
        logger.info('Configuration:\n' + self.config.to_log_string())
        
        # Setup
        self.file_manager.download_meta_files()
        meta_csv = self._load_and_filter_meta_csv()
        meta_json = utils.read_meta_json(self.config.input.json_config['meta'])
        last_json = utils.s3_load_json(
            utils.urljoin(f'{self.config.output.s3_config.data}/', 'info.json')
        )
        
        # Process each table
        self._setup_csv_output()
        
        start_row, end_row = self.config.input.processing_range
        total = len(meta_csv)
        
        for i, row in tqdm(meta_csv.iterrows(), desc='Processing ...', total=total):
            i = i + 1
            if start_row <= i <= end_row:
                self._process_single_table(row, meta_json, last_json)
        
        return dict(self.stats_data)
    
    def _load_and_filter_meta_csv(self) -> pd.DataFrame:
        """Load and filter meta CSV based on configuration."""
        meta_csv = pd.read_csv(self.config.input.csv_file)
        
        if self.config.input.select_rows:
            select_rows = (
                meta_csv['Consumer Loans Data Product']
                .str.lower()
                .isin(self.config.input.select_rows)
            )
            meta_csv = meta_csv[select_rows]
        
        return meta_csv.iloc[::-1]  # Reverse order
    
    def _setup_csv_output(self):
        """Setup CSV output file."""
        output_file = self.config.output.csv_config.file
        if os.path.exists(output_file):
            os.remove(output_file)
    
    def _process_single_table(self, row: pd.Series, meta_json: Dict, last_json: Dict):
        """Process a single table for statistics comparison."""
        table_name = row.get('PCDS Table Details with DB Name')
        logger.info(f">>> Start {table_name}")
        
        meta_info = meta_json.get(table_name)
        if not meta_info:
            logger.warning(f"No meta info found for {table_name}")
            return
        
        overall_result = ComparisonResult()
        
        # Process each partition
        for partition, s3_url in utils.s3_walk(
            self.config.output.s3_config.data, 
            prefix=f'PCDS_{table_name.upper()}_'
        ):
            
            if not ValidationUtils.validate_column_mappings(meta_info):
                logger.warning(f"No Columns to Compare for {table_name.upper()}")
                overall_result.has_mismatch = True
                overall_result.details = 'No Column Mappings or Matched Columns Provided'
                break
            
            partition_result = self._process_partition(
                table_name, partition, s3_url, meta_info, last_json
            )
            
            if partition_result:
                overall_result.has_mismatch |= partition_result.has_mismatch
                overall_result.mismatched_columns.update(partition_result.mismatched_columns)
        
        # Write results to CSV
        overall_result.details = '; '.join(overall_result.mismatched_columns)
        self._write_csv_row(row, overall_result)
        
        logger.info(">>> Finish Comparing column statistics")
        utils.separator()
    
    def _process_partition(self, table_name: str, partition: str, s3_url: str, 
                          meta_info: ut.MetaJSON, last_json: Dict) -> Optional[ComparisonResult]:
        """Process a single partition of a table."""
        last_pcds_download = last_json[
            self.file_manager.get_s3_key(s3_url, partition)
        ].get('ctime')[:10]
        
        # Setup AWS conditions
        aws_conditions = DateUtils.get_aws_until_condition(
            meta_info.aws.row_variable,
            meta_info.aws.column_types[meta_info.aws.row_variable],
            last_pcds_download
        )
        
        if partition != 'full':
            logger.info(f"\t\tCompare table where {partition}")
            partition_condition = DateUtils.get_aws_where_condition(
                meta_info.aws.row_variable,
                meta_info.aws.column_types[meta_info.aws.row_variable],
                partition
            )
            if partition_condition:
                aws_conditions.append(partition_condition)
        else:
            logger.info("\t\tCompare the entire table")
        
        # Load data
        df_pcds = pd.read_parquet(s3_url)
        
        db_aws, tbl_aws = meta_info.aws_table.split('.', maxsplit=1)
        df_aws = self.query_handler.query_athena(
            columns='*',
            database=db_aws,
            table=tbl_aws,
            where_conditions=[c for c in aws_conditions if c],
            s3_cache=s3_url.replace('PCDS', 'AWS')
        )
        
        if len(df_aws) == 0:
            logger.info("\t\tEmpty AWS table - skipping comparison")
            return ComparisonResult(has_mismatch=False)
        
        # Process and compare statistics
        return self._compare_partition_statistics(
            df_pcds, df_aws, meta_info, table_name, partition, last_pcds_download
        )
    
    def _compare_partition_statistics(self, df_pcds: pd.DataFrame, df_aws: pd.DataFrame,
                                    meta_info: ut.MetaJSON, table_name: str, 
                                    partition: str, last_pcds_download: str) -> ComparisonResult:
        """Compare statistics for a specific partition."""
        # Filter data by excluded dates
        exclude_dates = DateUtils.extend_excludes(
            meta_info.time_excludes, last_pcds_download
        )
        
        # Update AWS metadata for comparison
        meta_aws = meta_info.aws
        meta_aws.column_mapping = {v: k for k, v in meta_info.pcds.column_mapping.items()}
        meta_aws.column_types = {
            k: v for k, v in meta_aws.column_types.items()
            if k in meta_info.pcds.column_mapping.values()
        }
        
        # Filter AWS data
        dt_aws = pd.to_datetime(df_aws[meta_aws.row_variable])
        df_aws = df_aws[~dt_aws.isin(exclude_dates)][list(meta_aws.column_mapping)]
        
        # Filter PCDS data
        dt_pcds = pd.to_datetime(df_pcds[meta_info.pcds.row_variable])
        df_pcds = df_pcds[~dt_pcds.isin(exclude_dates)]
        
        # Handle row count mismatch
        if len(df_pcds) != len(df_aws):
            meta_info.time_excludes.append('2025-03-20')
        
        # Calculate statistics
        logger.info("\t\tCalculate column statistics and Compare")
        
        stats_aws = StatsDataframe(
            'AWS', meta_aws.column_types, meta_aws.row_variable
        ).transform(df_aws)
        logger.info("\t\t\tAWS column summary statistics is calculated")
        
        stats_pcds = StatsDataframe(
            'PCDS', meta_info.pcds.column_types, meta_info.pcds.row_variable
        ).transform(df_pcds, column_map=meta_info.pcds.column_mapping)
        logger.info("\t\t\tPCDS column summary statistics is calculated")
        
        # Align statistics
        stats_pcds, stats_aws = stats_pcds.sort_index(), stats_aws.sort_index()
        
        # Handle shape mismatches
        if stats_pcds.shape != stats_aws.shape:
            logger.warning(f"Table {table_name.upper()} has shape mismatch")
            aws_duplicates = ValidationUtils.get_duplicates(
                list(meta_info.pcds.column_mapping.values())
            )
            
            if aws_duplicates:
                logger.warning(
                    f"One AWS -> Multiple PCDS columns: "
                    f"{SEP.join(f'{k}->{meta_aws.column_mapping[k]}' for k in aws_duplicates)}"
                )
                stats_aws = stats_aws.query('~index.isin(@aws_duplicates)')
                stats_pcds = stats_pcds.query('~index.isin(@aws_duplicates)')
        
        # Compare statistics
        comparison_result = StatisticsComparator.compare_statistics(stats_pcds, stats_aws)
        mismatched_columns = list(comparison_result.mismatched_columns)
        
        # Store partition statistics
        _, tbl_pcds = meta_info.pcds_table.split('.', maxsplit=1)
        self.stats_data[table_name][partition] = PartitionStatistics(
            pcds_stats=stats_pcds.set_index(
                stats_pcds.index.map(meta_aws.column_mapping)
            ),
            pcds_mismatches=[meta_aws.column_mapping.get(x) for x in mismatched_columns],
            pcds_name=tbl_pcds.lower(),
            aws_stats=stats_aws,
            aws_mismatches=mismatched_columns,
            aws_name=tbl_aws.lower()
        )
        
        return comparison_result
    
    def _write_csv_row(self, row: pd.Series, result: ComparisonResult):
        """Write a row to the CSV output file."""
        with open(self.config.output.csv_config.file, 'a+', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=self.config.output.csv_config.columns)
            
            # Write header if file is empty
            if fp.tell() == 0:
                writer.writeheader()
            
            writer.writerow({**row.to_dict(), **result.to_dict()})


# ===============================
# CLI and Main Function
# ===============================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Calculate Columnwise Descriptive Statistics And Make Comparisons'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Config file for calculating column statistics',
        default=r'files\inputs\config_stat_0505.cfg'
    )
    return parser


def main():
    """Main function to run the statistics comparison process."""
    # Parse arguments and load configuration
    args = create_argument_parser().parse_args()
    config = Config().from_disk(args.config)
    config = ut.StatConfig(**config)
    
    # Setup environment and AWS credentials
    from dotenv import load_dotenv
    load_dotenv(config.input.env_file)
    utils.aws_creds_renew()
    
    # Setup logging
    logger.add(**config.output.log_config.to_dict())
    
    try:
        utils.start_run()
        
        # Process statistics
        processor = StatisticsProcessor(config)
        stats_data = processor.process_all_tables()
        
        # Save results
        with open(config.output.pickle_file, 'wb') as fp:
            pickle.dump(stats_data, fp)
        
        # Upload to S3
        processor.file_manager.upload_results_to_s3()
        
        logger.info("Statistics processing completed successfully")
        
    except Exception as e:
        logger.error("Error in processing ... Stopping")
        logger.exception(e)
        raise
    finally:
        utils.end_run()


if __name__ == "__main__":
    main()