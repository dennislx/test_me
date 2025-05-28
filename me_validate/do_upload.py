"""
Restructured Upload Module

This module handles uploading PCDS table data to S3 with support for 
partitioning, progress tracking, and comprehensive error handling.
"""

import argparse
import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime as dt

import pandas as pd
import pandas.io.sql as psql
from loguru import logger
from confection import Config
from tqdm import tqdm
from dotenv import load_dotenv

import utils
import utils_type as ut


# ===============================
# Configuration and Data Classes
# ===============================

@dataclass
class UploadMetrics:
    """Metrics tracking for upload operations."""
    group: str = ""
    name: str = ""
    visited: set = field(default_factory=set)
    row_count: int = 0
    column_count: int = 0
    memory_size: int = 0
    compressed_size: int = 0
    pull_time: int = 0
    upload_time: int = 0
    creation_time: str = ""
    s3_address: str = ""
    
    @classmethod
    def from_table_row(cls, table_row) -> 'UploadMetrics':
        """Create metrics from table row data."""
        group, name, *_ = table_row
        return cls(group=group, name=name, visited=set())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for CSV output."""
        return {
            'Consumer Loans Data Product': self.group,
            'PCDS Table Details with DB Name': self.name,
            'Number of Rows': self.row_count,
            'Number of Columns': self.column_count,
            'Memory Size': utils.DataUtils.get_memory_size(self.memory_size, as_dataset=False),
            'Compress Size': utils.DataUtils.get_memory_size(self.compressed_size, as_dataset=False),
            'Pull Time': utils.Timer.format_duration(self.pull_time),
            'Upload Time': utils.Timer.format_duration(self.upload_time),
            'S3 Address': self.s3_address,
            'Last Modified': self.creation_time
        }
    
    def update_from_time_data(self, time_data: Dict[str, Any]) -> None:
        """Update metrics from time tracking data."""
        for key, data in time_data.items():
            if not key.lower().startswith(f'pcds_{self.name}'):
                continue
            if key in self.visited:
                continue
                
            for metric_key, value in data.items():
                if metric_key == 'ctime':
                    self.creation_time = max(self.creation_time, value)
                elif metric_key == 's3addr':
                    self.s3_address = value
                elif hasattr(self, metric_key):
                    current_value = getattr(self, metric_key)
                    setattr(self, metric_key, value + current_value)
            
            self.visited.add(key)


# ===============================
# Database Connection Management
# ===============================

class PCDSConnectionManager:
    """Manages PCDS database connections with proper resource handling."""
    
    def __init__(self):
        self.service_mappings = {
            'p_uscb_cnsmrlnd_svc': '21P',
            'p_uscb_rft_svc': '30P',
            'pcds_svc': '00P'
        }
    
    def get_connection(self, service_name: str, 
                      ldap_service: str = 'ldap://oid.barcapint.com:4050'):
        """Create and return PCDS database connection."""
        import oracledb
        
        if service_name not in self.service_mappings:
            raise ValueError(f"Unknown service name: {service_name}")
        
        dns_tns = utils.solve_ldap(
            f'{ldap_service}/{service_name},cn=OracleContext,dc=barcapint,dc=com'
        )
        pcds_pwd = f'PCDS_{self.service_mappings[service_name]}'
        
        try:
            usr = os.environ['PCDS_USR']
            pwd = os.environ[pcds_pwd]
        except KeyError as e:
            raise ValueError(f"Missing environment variable: {e}")
        
        return oracledb.connect(user=usr, password=pwd, dsn=dns_tns)


# ===============================
# Query Building
# ===============================

class QueryBuilder:
    """Builds SQL queries for PCDS data extraction."""
    
    @staticmethod
    def build_select_query(table: str, columns: str = '*', where_clause: str = '') -> str:
        """Build a SELECT query with optional WHERE clause."""
        query = f"SELECT {columns} FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        return query.strip()
    
    @staticmethod
    def build_where_clause(date_var: str, date_type: str, partition_type: str, 
                          date_range: str, date_format: str) -> str:
        """Build WHERE clause for date-based filtering."""
        if 'char' in date_type.lower():
            date_var = f"TO_DATE({date_var}, '{date_format}')"
        
        if partition_type == 'year':
            return f"TO_CHAR({date_var}, 'YYYY') = '{date_range}'"
        elif partition_type == 'year_month':
            return f"TO_CHAR({date_var}, 'YYYY-MM') = '{date_range}'"
        else:
            return ""


# ===============================
# Data Upload Operations
# ===============================

class DataUploader:
    """Handles data upload operations to S3."""
    
    def __init__(self, config: ut.PullConfig):
        self.config = config
        self.connection_manager = PCDSConnectionManager()
        self.query_builder = QueryBuilder()
        self.time_tracking = {}
    
    def upload_table_data(self, pcds_info: ut.MetaInfo, s3_base_url: str,
                         s3_basename: str, where_conditions: List[str],
                         should_repull: bool, subfolder: Optional[str] = None) -> None:
        """Upload PCDS table data to S3 with optional partitioning."""
        
        service, table = pcds_info.info_string.split('.', maxsplit=1)
        
        # Construct S3 URL
        main_url = utils.urljoin(f'{s3_base_url}/', s3_basename)
        if subfolder:
            full_url = utils.urljoin(f'{main_url}/', subfolder)
        else:
            full_url = main_url
        full_url = full_url + '.pq'
        
        # Check if data already exists and we don't need to repull
        try:
            if not should_repull and utils.S3Manager.object_exists(full_url):
                logger.info(f"\t\t\tPCDS data already exists at {full_url}")
                return
        except Exception:
            pass  # Continue with upload if check fails
        
        logger.info(f"\t\t\tUploading PCDS data from {table} -> {full_url}")
        
        # Build and execute query
        where_clause = ' AND '.join(condition for condition in where_conditions if condition)
        query = self.query_builder.build_select_query(
            table=table,
            columns='*',
            where_clause=where_clause
        )
        
        # Execute query and measure performance
        with self.connection_manager.get_connection(service) as conn:
            with utils.Timer() as pull_timer:
                df_pcds = psql.read_sql_query(query, conn)
                pull_time = pull_timer.pause()
        
        # Validate row count
        nrow, ncol = df_pcds.shape
        if hasattr(pcds_info, 'row_count') and nrow != pcds_info.row_count:
            logger.warning(
                f"Row count mismatch: DataFrame({nrow}) vs Expected({pcds_info.row_count})"
            )
        
        # Calculate memory usage and upload
        memory_size = int(df_pcds.memory_usage(deep=True).sum())
        
        with utils.Timer() as upload_timer:
            compressed_size = utils.S3Manager.save_dataframe(df_pcds, full_url)
            upload_time = upload_timer.pause()
        
        # Get S3 metadata
        creation_time = ""
        try:
            for _, info in utils.S3Manager.get_metadata(full_url).items():
                creation_time = utils.DateTimeUtils.correct_s3_time(info['LastModified'])
                break
        except Exception:
            creation_time = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(
            f"\t\t\tUploaded PCDS data "
            f"({utils.DataUtils.get_memory_size(memory_size, as_dataset=False)}) to "
            f"{full_url} ({utils.DataUtils.get_memory_size(compressed_size, as_dataset=False)})"
        )
        
        # Store timing data
        time_key = f'{s3_basename}_{subfolder}' if subfolder else s3_basename
        self.time_tracking[time_key] = {
            'pull_time': pull_time,
            'upload_time': upload_time,
            'creation_time': creation_time,
            'row_count': nrow,
            'column_count': ncol,
            'memory_size': memory_size,
            'compressed_size': compressed_size,
            's3_address': main_url
        }
    
    def save_timing_data(self) -> None:
        """Save timing data to S3."""
        try:
            s3_url = utils.urljoin(f'{self.config.output.s3_config.data}/', 'info.json')
            utils.S3Manager.save_json(self.time_tracking, s3_url)
            logger.info(f"Timing data saved to {s3_url}")
        except Exception as e:
            logger.error(f"Failed to save timing data: {e}")


# ===============================
# Partition Management
# ===============================

class PartitionManager:
    """Manages data partitioning strategies."""
    
    PARTITION_MODES = {
        'none': 'full',
        'year': 'yyyy',
        'year_month': 'yymm'
    }
    
    def __init__(self, sql_engine: utils.SQLEngine):
        self.sql_engine = sql_engine
    
    def get_partition_ranges(self, table_name: str, date_column: str, 
                           partition_type: str) -> List[str]:
        """Get partition ranges for the specified table and partition type."""
        if partition_type == 'none':
            return ['full']
        
        try:
            return self.sql_engine.get_query_range(table_name, date_column, partition_type)
        except Exception as e:
            logger.warning(f"Failed to get partition ranges: {e}")
            return ['full']  # Fallback to full table
    
    def get_partition_mode(self, partition_type: str) -> str:
        """Get partition mode string for S3 naming."""
        return self.PARTITION_MODES.get(partition_type, 'full')


# ===============================
# Main Upload Processor
# ===============================

class UploadProcessor:
    """Main processor for upload operations."""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.uploader = DataUploader(self.config)
        self.partition_manager = PartitionManager(utils.SQLEngine('AWS'))
        self.csv_writer_initialized = False
    
    def _load_config(self, config_path: Path) -> ut.PullConfig:
        """Load and validate configuration."""
        try:
            config_dict = Config().from_disk(config_path)
            return ut.PullConfig(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_environment(self) -> None:
        """Setup logging, environment, and AWS credentials."""
        # Remove existing output file if starting from beginning
        if os.path.exists(self.config.output.csv_config.file):
            os.remove(self.config.output.csv_config.file)
        
        # Setup logging
        logger.add(**self.config.output.log_config.to_dict())
        logger.info('Configuration:\n' + self.config.to_log_string())
        
        # Load environment variables
        load_dotenv(self.config.input.env_file)
        
        # Initialize AWS credentials
        utils.start_run()
        utils.aws_creds_renew(15 * 60)
        
        # Load existing timing data
        try:
            s3_url = utils.urljoin(f'{self.config.output.s3_config.data}/', 'info.json')
            self.uploader.time_tracking = utils.S3Manager.load_json(s3_url)
        except Exception:
            self.uploader.time_tracking = {}
            logger.info("No existing timing data found, starting fresh")
    
    def _process_single_table(self, table_row, meta_json: Dict) -> UploadMetrics:
        """Process upload for a single table."""
        table_name = table_row.get('PCDS Table Details with DB Name')
        
        # Configure pull settings
        pull_config = self.config.pull_data(table_name)
        metrics = UploadMetrics.from_table_row(table_row)
        
        # Get metadata
        meta_info = meta_json.get(table_name)
        if not meta_info:
            logger.warning(f"No metadata found for table: {table_name}")
            return metrics
        
        meta_pcds = meta_info.pcds
        partition_type = pull_config.partition_type
        
        # Get partition ranges
        partition_ranges = self.partition_manager.get_partition_ranges(
            meta_info.aws_table, meta_info.aws.row_variable, partition_type
        )
        
        # Process each partition
        for time_range in tqdm(partition_ranges, desc=f"Processing {table_name}"):
            where_conditions = [pull_config.get_where_condition('PCDS')]
            subfolder = None
            
            if partition_type == 'none':
                logger.info("\t\tDownloading entire table")
            else:
                # Build date-based WHERE clause
                date_where = self.query_builder.build_where_clause(
                    date_var=meta_pcds.row_variable,
                    date_type=meta_info.pcds.column_types.get(meta_pcds.row_variable, 'DATE'),
                    partition_type=partition_type,
                    date_range=time_range,
                    date_format=pull_config.get_date_format('PCDS')
                )
                
                if date_where:
                    where_conditions.append(date_where)
                
                subfolder = f'{meta_pcds.row_variable.lower()}={time_range}'
                logger.info(f"\t\tDownloading partition: {subfolder}")
            
            # Upload data
            mode = self.partition_manager.get_partition_mode(partition_type)
            s3_basename = '_'.join(['PCDS', table_name.upper(), mode])
            
            try:
                self.uploader.upload_table_data(
                    pcds_info=meta_pcds,
                    s3_base_url=self.config.output.s3_config.data,
                    s3_basename=s3_basename,
                    where_conditions=where_conditions,
                    should_repull=pull_config.should_delete,
                    subfolder=subfolder
                )
            except Exception as e:
                logger.error(f"Failed to upload data for {table_name}: {e}")
                continue
        
        # Update metrics from timing data
        metrics.update_from_time_data(self.uploader.time_tracking)
        return metrics
    
    def _write_results(self, metrics: UploadMetrics) -> None:
        """Write results to CSV file."""
        with open(self.config.output.csv_config.file, 'a+', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=self.config.output.csv_config.columns)
            
            if not self.csv_writer_initialized:
                writer.writeheader()
                self.csv_writer_initialized = True
            
            writer.writerow(metrics.to_dict())
    
    def run(self) -> None:
        """Execute the complete upload process."""
        try:
            self._setup_environment()
            
            # Load input data
            meta_csv = pd.read_csv(self.config.input.csv_file)
            meta_json = utils.FileProcessor.read_meta_json(self.config.input.json_config['meta'])
            
            # Process tables in reverse order (as in original)
            meta_csv = meta_csv.iloc[::-1]
            total = len(meta_csv)
            
            for i, (_, row) in enumerate(tqdm(
                meta_csv.iterrows(), desc='Processing tables...', total=total
            )):
                try:
                    metrics = self._process_single_table(row, meta_json)
                    self._write_results(metrics)
                    utils.separator()
                except Exception as e:
                    logger.error(f"Error processing table {row.get('PCDS Table Details with DB Name', 'unknown')}: {e}")
                    continue
            
        except Exception as e:
            logger.error("Error in upload processing... Stopping")
            logger.exception(e)
            raise
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Cleanup operations after processing."""
        try:
            # Save timing data
            self.uploader.save_timing_data()
            
            # Save local timing data backup
            with open(self.config.output.json_file, 'w') as f:
                json.dump(self.uploader.time_tracking, f, indent=2)
            
            logger.info("Upload processing completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            utils.end_run()


# ===============================
# Command Line Interface
# ===============================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Upload PCDS Table Data to S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python do_upload.py --config config.cfg
  python do_upload.py --config config.cfg --dry-run
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file for uploading PCDS table data',
        default=Path('files/inputs/config_pull.cfg')
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actually uploading data'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for the upload script."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Validate configuration file exists
        if not args.config.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        # Create and run processor
        processor = UploadProcessor(args.config)
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No data will be uploaded")
            # TODO: Implement dry run functionality
            return
        
        processor.run()
        
    except KeyboardInterrupt:
        logger.info("Upload process interrupted by user")
    except Exception as e:
        logger.error(f"Upload process failed: {e}")
        raise


if __name__ == "__main__":
    main()