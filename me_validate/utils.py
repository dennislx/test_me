"""
Restructured Utilities Module

This module provides utility functions and classes for database connections,
file operations, AWS services, data processing, and application management.
"""

import re
import io
import os
import time
import json
import uuid
import pytz
import boto3
import botocore
import botocore.exceptions
import requests
import threading
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import datetime as dt
import awswrangler as aws
import pandas.api.types as ptype

from collections import namedtuple
from urllib.parse import urlparse, urljoin, uses_relative, uses_netloc
from loguru import logger
from joblib import Memory, expires_after
from unittest import mock
from configparser import ConfigParser
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from utils_type import (
    MetaTable, ColumnMap, S3Config, MetaJSON, PLATFORM, 
    TPartition, MetaMatch, Trange
)

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning, 
               message='.*pandas only supports SQLAlchemy connectable.*')


# ===============================
# Configuration and Constants
# ===============================

AppExec = {'excel': 'Excel.Application', 'ppt': 'PowerPoint.Application'}
TODAY = dt.datetime.now().strftime('%Y%m%d')
SESSION = None
WIDTH = 80
MEMORY = Memory('.cache', verbose=1)

# Configure URL parsing for S3
if 's3' not in uses_netloc:
    uses_netloc.append('s3')
    uses_relative.append('s3')


# ===============================
# Logging and Display Utilities
# ===============================

class LoggingUtils:
    """Utilities for logging and display formatting."""
    
    @staticmethod
    def start_run():
        """Log the start of a processing run."""
        logger.info('\n\n' + '=' * WIDTH)
    
    @staticmethod
    def end_run():
        """Log the end of a processing run."""
        logger.info('\n\n' + '=' * WIDTH)
    
    @staticmethod
    def separator():
        """Log a separator line."""
        logger.info('-' * WIDTH)


# ===============================
# Date and Time Utilities
# ===============================

class DateTimeUtils:
    """Utilities for date and time operations."""
    
    @staticmethod
    def get_date_sorted(df: pd.Series, format_str: str = '%Y-%m-%d') -> pd.Series:
        """Sort dates and return formatted strings."""
        if ptype.is_string_dtype(df):
            df = pd.to_datetime(df, errors='coerce')
        assert ptype.is_datetime64_dtype(df), "Series must be datetime-like"
        return df.sort_values().dt.strftime(format_str)
    
    @staticmethod
    def get_date_filter(df: pd.Series, date_list: List[str], 
                       format_str: str = '%Y-%m-%d') -> pd.Series:
        """Filter dates based on a list of date strings."""
        if ptype.is_string_dtype(df):
            df = pd.to_datetime(df, errors='coerce')
        assert ptype.is_datetime64_dtype(df), "Series must be datetime-like"
        return df.dt.strftime(format_str).isin(date_list)
    
    @staticmethod
    def correct_s3_time(time_input: Union[str, dt.datetime], 
                       time_format: str = '%a, %d %b %Y %H:%M:%S %Z') -> str:
        """Convert S3 time to NYC timezone string."""
        if isinstance(time_input, str):
            time_input = dt.datetime.strptime(time_input, time_format).replace(tzinfo=pytz.utc)
        nyc_tz = pytz.timezone('America/New_York')
        return time_input.astimezone(nyc_tz).strftime('%Y-%m-%d %H:%M:%S')


# ===============================
# Data Processing Utilities
# ===============================

class DataUtils:
    """Utilities for data processing and formatting."""
    
    @staticmethod
    def get_default_value(data_dict: Dict, key: str, default_key: str = 'DEFAULT') -> Any:
        """Get value from dictionary with fallback to default key."""
        data_dict = UDict(**data_dict)
        try:
            return data_dict[key]
        except KeyError:
            return data_dict[default_key]
    
    @staticmethod
    def format_abbreviation(value: Union[int, float], precision: int = 2, 
                          percentage: bool = False) -> str:
        """Format numbers with K, M, B, T suffixes."""
        if pd.isna(value):
            return ""
        if percentage:
            return f'{value:,.{precision}%}'
        
        magnitude = 0
        while abs(value) >= 1_000:
            magnitude += 1
            value /= 1_000.0
        
        suffixes = ["", "K", "M", "B", "T", "Q"]
        return f'{np.round(value, precision)}{suffixes[magnitude]}'
    
    @staticmethod
    def get_memory_size(data: Union[pd.DataFrame, int], as_dataset: bool = True) -> str:
        """Format memory size in human-readable format."""
        if as_dataset and isinstance(data, pd.DataFrame):
            bytes_size = data.memory_usage(deep=True).sum()
        else:
            bytes_size = data
        
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1_024:
                return f'{bytes_size:.3f} {unit}'
            bytes_size /= 1_024
        return f'{bytes_size:.3f} PB'
    
    @staticmethod
    def get_disk_usage(drive: str = 'C:/') -> None:
        """Display disk usage information."""
        import shutil
        total, used, free = shutil.disk_usage(drive)
        
        def format_size(size): 
            return f'{size / 1024**3:.3f} GB'
        
        print(f"Total: {format_size(total)}")
        print(f"Used:  {format_size(used)}")
        print(f"Free:  {format_size(free)}")


# ===============================
# Database Connection Classes
# ===============================

class DatabaseConnectionManager:
    """Base class for database connection management."""
    
    def __init__(self):
        self._connections = {}
    
    def get_connection(self, connection_type: str, **kwargs):
        """Get a database connection based on type."""
        if connection_type == 'PCDS':
            return self._get_pcds_connection(**kwargs)
        elif connection_type == 'AWS':
            return self._get_aws_connection(**kwargs)
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")
    
    def _get_pcds_connection(self, service_name: str, 
                           ldap_service: str = 'ldap://oid.barcapint.com:4050'):
        """Create PCDS Oracle database connection."""
        svc2server = {
            'p_uscb_cnsmrlnd_svc': '21P',
            'p_uscb_rft_svc': '30P',
            'pcds_svc': '00P'
        }
        
        import oracledb
        pcds_pwd = f'PCDS_{svc2server.get(service_name)}'
        dns_tns = self._solve_ldap(f'{ldap_service}/{service_name},cn=OracleContext,dc=barcapint,dc=com')
        usr, pwd = os.environ['PCDS_USR'], os.environ[pcds_pwd]
        return oracledb.connect(user=usr, password=pwd, dsn=dns_tns)
    
    def _get_aws_connection(self, database: str, region_name: str = "us-east-1", 
                          work_group: str = "uscb-analytics",
                          s3_staging_dir: str = "s3://355538383407-us-east-1-athena-output/uscb-analytics/"):
        """Create AWS Athena connection."""
        from pyathena import connect
        return connect(
            schema_name=database, region_name=region_name,
            work_group=work_group, s3_staging_dir=s3_staging_dir
        )
    
    @staticmethod
    def _solve_ldap(ldap_dsn: str) -> str:
        """Get TNS connect string from LDAP."""
        from ldap3 import Server, Connection
        pattern = r"^ldap:\/\/(.+)\/(.+)\,(cn=OracleContext.*)$"
        match = re.match(pattern, ldap_dsn)
        
        if not match:
            return None
        
        ldap_server, db, ora_context = match.groups()
        server = Server(ldap_server)
        conn = Connection(server)
        conn.bind()
        conn.search(ora_context, f"(cn={db})", attributes=['orclNetDescString'])
        return conn.entries[0].orclNetDescString.value


# ===============================
# AWS Services Management
# ===============================

class AWSManager:
    """Manager for AWS services and operations."""
    
    @staticmethod
    def renew_credentials(seconds: int = 0, 
                         msg: str = 'AWS Credential Has Been Updated !') -> None:
        """Renew AWS credentials using internal authentication service."""
        global SESSION
        
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        usr, pwd = os.environ['EDP_USR'], os.environ['EDP_PWD']
        token_url = 'https://awsportal.barcapint.com/v1/jwttoken/'
        arn_url = ('https://awsportal.barcapint.com/v1/creds-provider/'
                  'provide-credentials/arn:aws:iam::355538383407:role/'
                  'app-uscb-analytics=fglbuscloudanalytic@I+000')
        
        # Get JWT token
        resp = requests.post(
            token_url, headers={'Accept': '*/*'}, verify=False,
            json={"username": usr, "password": pwd}
        ).json()
        
        # Get credentials
        headers = {"Accept": "*/*", "Authorization": "Bearer " + resp['token']}
        creds = requests.get(arn_url, headers=headers, verify=False).json()['Credentials']
        
        # Set environment variables
        os.environ['AWS_ACCESS_KEY_ID'] = creds['AccessKeyId']
        os.environ['AWS_SECRET_ACCESS_KEY'] = creds['SecretAccessKey']
        os.environ['AWS_SESSION_TOKEN'] = creds['SessionToken']
        os.environ['HTTPS_PROXY'] = f'http://{usr}:{pwd}@35.165.20.1:8080'
        
        if msg:
            logger.info(msg)
        LoggingUtils.separator()
        
        SESSION = boto3.Session(region_name='us-east-1')
        
        if seconds > 0:
            threading.Timer(seconds, AWSManager.renew_credentials, args=(seconds,)).start()


# ===============================
# S3 Operations
# ===============================

class S3Manager:
    """Manager for S3 operations."""
    
    @staticmethod
    def upload_file(local_path: str, s3_url: str, **kwargs) -> None:
        """Upload file to S3."""
        aws.s3.upload(local_path, s3_url, boto3_session=SESSION, 
                     s3_additional_kwargs=kwargs)
        logger.info(f"Success: upload {local_path} to {s3_url}")
    
    @staticmethod
    def delete_object(s3_url: str) -> None:
        """Delete S3 object."""
        aws.s3.delete_objects(s3_url, boto3_session=SESSION)
    
    @staticmethod
    def get_metadata(s3_url: str) -> Dict:
        """Get S3 object metadata."""
        return aws.s3.describe_objects(s3_url, boto3_session=SESSION)
    
    @staticmethod
    def object_exists(s3_url: str) -> bool:
        """Check if S3 object exists."""
        objects = aws.s3.list_objects(s3_url, boto3_session=SESSION)
        return len(objects) > 0
    
    @staticmethod
    def load_dataframe(s3_url: str, folder: bool = False, **kwargs) -> pd.DataFrame:
        """Load DataFrame from S3."""
        if folder:
            return pd.concat(S3Manager.load_dataframe(x) 
                           for x in aws.s3.list_objects(s3_url))
        
        try:
            in_buffer = io.BytesIO()
            aws.s3.download(s3_url, in_buffer, boto3_session=SESSION, **kwargs)
            in_buffer.seek(0)
            return pd.read_parquet(in_buffer, **kwargs)
        except botocore.exceptions.ClientError:
            raise FileNotFoundError(f"S3 object not found: {s3_url}")
    
    @staticmethod
    def load_json(s3_url: str, **kwargs) -> Dict:
        """Load JSON from S3."""
        kwargs['path_suffix'] = '.json'
        return aws.s3.read_json(s3_url, boto3_session=SESSION, **kwargs).to_dict()
    
    @staticmethod
    def save_json(obj: Any, s3_url: str, **kwargs) -> None:
        """Save object as JSON to S3."""
        aws.s3.to_json(df=pd.DataFrame(obj), path=s3_url, 
                      boto3_session=SESSION, **kwargs)
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, s3_url: str, **kwargs) -> int:
        """Save DataFrame to S3 and return compressed size."""
        out_buffer = io.BytesIO()
        df.to_parquet(out_buffer, index=False)
        out_buffer.seek(0)
        S3Manager.upload_file(out_buffer, s3_url)
        return out_buffer.getbuffer().nbytes
    
    @staticmethod
    def download_folder(s3_url: str, local_path: str, prefix: Optional[str] = None) -> None:
        """Download files from S3 folder to local path."""
        logger.info(f'Download files from {s3_url} to {local_path}')
        for s3_file in aws.s3.list_objects(s3_url, boto3_session=SESSION):
            s3_base = os.path.basename(s3_file)
            if prefix and not s3_base.startswith(prefix):
                continue
            local_file = os.path.join(local_path, s3_base)
            aws.s3.download(s3_file, local_file, boto3_session=SESSION)
    
    @staticmethod
    def walk_objects(s3_base_url: str, prefix: str = '') -> List[Tuple[str, str]]:
        """Walk through S3 objects and yield partition and URL pairs."""
        objects = aws.s3.list_objects(s3_base_url, boto3_session=SESSION)
        for obj_url in objects:
            if prefix and not os.path.basename(obj_url).startswith(prefix):
                continue
            # Extract partition info from path
            partition = S3Manager._extract_partition_from_url(obj_url)
            yield partition, obj_url
    
    @staticmethod
    def _extract_partition_from_url(s3_url: str) -> str:
        """Extract partition information from S3 URL."""
        # This is a simplified implementation
        # In practice, you'd parse the URL structure to extract partition info
        path_parts = urlparse(s3_url).path.split('/')
        for part in path_parts:
            if '=' in part:
                return part
        return 'full'


# ===============================
# Query Processing and Caching
# ===============================

class QueryProcessor:
    """Handles query processing with caching capabilities."""
    
    def __init__(self):
        self.memory = MEMORY
    
    @MEMORY.cache(cache_validation_callback=expires_after(hours=120), ignore=['connection'])
    def execute_query(self, query_str: str, connection) -> pd.DataFrame:
        """Execute SQL query with caching."""
        return psql.read_sql_query(query_str, connection)
    
    def execute_athena_query(self, sql_query: str, s3_output: str = '', 
                           max_retry: int = 5, load_data: bool = True, 
                           **kwargs) -> Union[pd.DataFrame, str]:
        """Execute Athena query with S3 caching."""
        s3_url = self._get_random_s3_path(sql_query, s3_output)
        
        if not S3Manager.object_exists(s3_url):
            self._unload_to_s3(sql_query, s3_url, **kwargs)
        
        if not load_data:
            return s3_url
        
        for _ in range(max_retry):
            try:
                return S3Manager.load_dataframe(s3_url)
            except botocore.exceptions.ResponseStreamingError:
                time.sleep(1)
        
        raise Exception(f"Failed to load data after {max_retry} retries")
    
    @MEMORY.cache(cache_validation_callback=expires_after(hours=24))
    def _get_random_s3_path(self, sql: str, s3_bucket: str) -> str:
        """Generate random S3 path for query caching."""
        return urljoin(s3_bucket, str(uuid.uuid4()))
    
    def _unload_to_s3(self, query: str, s3_url: str, **kwargs) -> None:
        """Unload query results to S3."""
        # Extract database from query
        db_match = re.search(r'from\s+([^\s\.]+)', query, re.IGNORECASE)
        if not db_match:
            raise ValueError("Cannot extract database from query")
        
        database = db_match.groups()[0]
        
        unload_kwargs = dict(
            database=database, file_format='PARQUET', compression='snappy',
            workgroup="uscb-analytics"
        )
        
        query = query.format(**kwargs).strip()
        resp = aws.athena.unload(
            sql=query, path=s3_url, boto3_session=SESSION, **unload_kwargs
        )
        
        if resp.raw_payload['Status']['State'] != 'SUCCEEDED':
            raise Exception("Athena query failed")


# ===============================
# File and Data Processing
# ===============================

class FileProcessor:
    """Handles file processing operations."""
    
    @staticmethod
    def read_meta_json(json_file: str) -> Dict[str, MetaJSON]:
        """Read and parse meta JSON file."""
        with open(json_file, 'r') as fp:
            data = json.load(fp)
        return {k: MetaJSON(**v) for k, v in data.items()}
    
    @staticmethod
    def read_input_excel(meta_table: MetaTable) -> pd.DataFrame:
        """Read and process input Excel file."""
        def extract_name(name):
            if pd.isna(name):
                return pd.NA
            remove_extra = r'\(.*\)'
            return re.sub(remove_extra, '', name).strip()
        
        def merge_pcds_service_table(df):
            cols = [x for x in df.columns if x != 'group']
            df[cols] = df[cols].map(extract_name)
            
            tbl = df.pop('pcds_tbl')
            tbl = tbl.combine_first(df.pop('pcds_backup'))
            svc = df.pop('pcds_svc').fillna('no_server_provided')
            df.loc[:, ['pcds_tbl']] = svc + '.' + tbl.str.lower()
            df['pcds_id'] = df['pcds_id'].str.upper()
            df['aws_id'] = df['aws_id'].str.lower()
        
        df = pd.read_excel(
            meta_table.file,
            sheet_name=meta_table.sheet,
            skiprows=meta_table.skip_rows,
            usecols=list(meta_table.select_cols)
        )
        df = df.rename(columns=meta_table.select_cols)
        
        if len(meta_table.select_rows) > 0:
            df['group'] = df['group'].astype(str)
            group_filter = ', '.join(f'"{x}"' for x in meta_table.select_rows)
            df = df.query(f'group in ({group_filter})')
        
        merge_pcds_service_table(df)
        return df
    
    @staticmethod
    def process_column_mapping(column_map: ColumnMap) -> Dict:
        """Process column mapping from Excel file."""
        def clean_column_name(col):
            if pd.isna(col):
                return 'comment'
            col = col.split('\n')[-1]
            return '_'.join(x.lower() for x in col.split())
        
        def fetch_column_value(row, names, na_str):
            for name in names:
                if name in row and row[name] != na_str:
                    return row[name]
            return pd.NA
        
        if os.path.exists(column_map.to_json) and not column_map.overwrite:
            with open(column_map.to_json, 'r') as fp:
                return json.load(fp)
        
        all_sheets = {}
        workbook = pd.read_excel(column_map.file, sheet_name=None)
        
        for name, sheet in workbook.items():
            if name in column_map.excludes:
                continue
            
            if 'Source' in sheet.columns:
                sheet = pd.DataFrame(sheet.iloc[1:].values, columns=sheet.iloc[0])
            
            sheet = sheet.rename(columns=clean_column_name)
            mapping_dict = {}
            
            for row in sheet.itertuples():
                row_dict = UDict(**row._asdict())
                pcds_col = fetch_column_value(row_dict, column_map.pcds_col, column_map.na_str)
                aws_col = fetch_column_value(row_dict, column_map.aws_col, column_map.na_str)
                
                if not (pd.isna(pcds_col) or pd.isna(aws_col)):
                    pcds_col = pcds_col.upper()
                    if pcds_col in mapping_dict:
                        logger.warning(f'Table {name} has duplicated PCDS column {pcds_col}')
                    mapping_dict[pcds_col] = aws_col.lower()
            
            if len(mapping_dict) == 0:
                logger.info(f"No match key found in {name}")
            
            all_sheets[name.lower()] = mapping_dict
        
        with open(column_map.to_json, 'w') as fp:
            json.dump(all_sheets, fp)
        
        return UDict(**all_sheets)


# ===============================
# SQL Engine
# ===============================

class SQLEngine:
    """SQL execution engine for different platforms."""
    
    AWS_SCHEME = r'from\s+([^\s\.]+)'  # match from (xxx).yyy
    
    def __init__(self, platform: PLATFORM):
        self.platform = platform
        self.db_manager = DatabaseConnectionManager()
        self.query_processor = QueryProcessor()
    
    def execute(self, query: str, **query_kwargs) -> pd.DataFrame:
        """Execute query based on platform."""
        if self.platform == 'PCDS':
            return self._execute_pcds_query(query, **query_kwargs)
        else:
            return self._execute_aws_query(query, **query_kwargs)
    
    def _execute_pcds_query(self, query: str, **query_kwargs) -> pd.DataFrame:
        """Execute PCDS query."""
        service_name = query_kwargs.pop('service_name')
        query = query.format(**query_kwargs).strip()
        
        with self.db_manager.get_connection('PCDS', service_name=service_name) as conn:
            return self.query_processor.execute_query(query, conn)
    
    def _execute_aws_query(self, query: str, **query_kwargs) -> pd.DataFrame:
        """Execute AWS query."""
        database_match = re.search(self.AWS_SCHEME, query, re.IGNORECASE)
        if not database_match:
            raise ValueError("Cannot extract database from query")
        
        database = database_match.groups()[0]
        query = query.format(**query_kwargs).strip()
        
        query_id = aws.athena.start_query_execution(
            query, boto3_session=SESSION, database=database
        )
        return aws.athena.get_query_results(query_id, boto3_session=SESSION)
    
    def get_query_range(self, table_db: str, column: str, 
                       partition: TPartition = 'none') -> List[Union[Trange, str]]:
        """Get query range based on partition type."""
        assert self.platform == 'AWS', "Query range only supported for AWS"
        
        query = f"SELECT min({column}), max({column}) FROM {table_db}"
        result = self.execute(query).iloc[0]
        
        match partition:
            case 'none':
                return [Trange(start_date=result.iloc[0], end_date=result.iloc[1])]
            case 'year':
                date_range = pd.date_range(*result, freq='1YS')
                return date_range.year.tolist()
            case 'year_month':
                date_range = pd.date_range(*result, freq='MS')
                return date_range.strftime('%Y-%m').tolist()


# ===============================
# Application Management
# ===============================

class ApplicationManager:
    """Manager for Windows applications (Excel, PowerPoint)."""
    
    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        self.app_type = self._determine_app_type()
    
    def _determine_app_type(self) -> str:
        """Determine application type from file extension."""
        if self.file_path.endswith('.xlsx'):
            return 'excel'
        elif self.file_path.endswith('.pptx'):
            return 'ppt'
        else:
            raise NotImplementedError(f"Unsupported file type: {self.file_path}")
    
    def _get_application(self, visible: bool = False):
        """Get application instance."""
        import win32com.client as wc
        
        try:
            app = wc.GetActiveObject(AppExec[self.app_type])
        except Exception:
            app = wc.Dispatch(AppExec[self.app_type])
        
        if visible:
            app.Visible = visible
        
        if self.app_type == 'excel':
            return app.Workbooks
        elif self.app_type == 'ppt':
            return app.Presentations
    
    def is_open(self) -> bool:
        """Check if application is currently open."""
        try:
            with open(self.file_path, 'r+b'):
                return False
        except IOError:
            return True
    
    def open_application(self) -> None:
        """Open the application."""
        if self.is_open():
            return
        
        apps = self._get_application(visible=True)
        try:
            apps.Open(self.file_path)
        except Exception:
            raise Exception('Failed to open application - file may be corrupted')
    
    def close_application(self, save: bool = False) -> None:
        """Close the application."""
        if not self.is_open():
            return
        
        apps = self._get_application()
        for app in apps:
            if app.FullName.lower() == self.file_path.lower():
                if self.app_type == 'excel':
                    app.Close(SaveChanges=save)
                elif self.app_type == 'ppt':
                    if save:
                        app.Save()
                    app.Close()


# ===============================
# Utility Classes
# ===============================

class UDict(dict):
    """Case-insensitive dictionary."""
    
    def __getitem__(self, key):
        return super().__getitem__(self._match_key(key))
    
    def __contains__(self, key):
        try:
            self._match_key(key)
            return True
        except KeyError:
            return False
    
    def _match_key(self, key):
        """Match key case-insensitively."""
        for k in self:
            if k.lower() == key.lower():
                return k
        raise KeyError(key)


@dataclass
class Timer:
    """Context manager for timing operations."""
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        pass
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        return time.perf_counter() - self.start
    
    def pause(self) -> float:
        """Pause timer and return elapsed time."""
        elapsed = self.elapsed
        self.start = time.perf_counter()
        return elapsed
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds'


# ===============================
# Configuration Utilities
# ===============================

def get_custom_config_parser(interpolate: bool = True):
    """Get custom configuration parser."""
    from confection import CustomInterpolation
    config = ConfigParser(
        interpolation=CustomInterpolation() if interpolate else None,
        allow_no_value=True,
    )
    config.optionxform = str
    return config


# Mock the configuration parser
mock.patch('confection.get_configparser', wraps=get_custom_config_parser).start()


# ===============================
# URL and Path Utilities
# ===============================

class URLUtils:
    """Utilities for URL and path operations."""
    
    @staticmethod
    def get_s3_info(s3_url: str, filepath: Optional[str] = None, 
                   basename: Optional[str] = None, add_today: bool = False):
        """Get S3 URL information with optional modifications."""
        S3Info = namedtuple('S3Info', ['prefix', 'bucket', 'scheme', 'url', 'ext'])
        
        def get_basename(path): 
            return os.path.basename(path)
        
        if basename:
            s3_url = urljoin(s3_url, basename)
        elif filepath and get_basename(filepath) != get_basename(s3_url):
            s3_url = urljoin(s3_url, get_basename(filepath))
        
        if add_today:
            base, ext = os.path.splitext(s3_url)
            s3_url = f'{base}_{TODAY}{ext}'
        
        obj = urlparse(s3_url)
        prefix = obj.path.lstrip('/')
        ext = os.path.splitext(obj.path)[1]
        
        return S3Info(
            prefix=prefix, 
            bucket=obj.netloc, 
            scheme=obj.scheme, 
            url=s3_url, 
            ext=ext
        )


# ===============================
# Legacy Function Wrappers
# ===============================
# These maintain backward compatibility with the original utils.py

def start_run():
    """Legacy wrapper for LoggingUtils.start_run()"""
    return LoggingUtils.start_run()

def end_run():
    """Legacy wrapper for LoggingUtils.end_run()"""
    return LoggingUtils.end_run()

def seperator():  # Note: keeping original typo for compatibility
    """Legacy wrapper for LoggingUtils.separator()"""
    return LoggingUtils.separator()

def get_datesort(df: pd.Series, format_str: str = '%Y-%m-%d'):
    """Legacy wrapper for DateTimeUtils.get_date_sorted()"""
    return DateTimeUtils.get_date_sorted(df, format_str)

def get_datedrop(df: pd.Series, date_lst: List[str], format_str: str = '%Y-%m-%d'):
    """Legacy wrapper for DateTimeUtils.get_date_filter()"""
    return DateTimeUtils.get_date_filter(df, date_lst, format_str)

def get_default(d: Dict, k: str, default_key: str = 'DEFAULT'):
    """Legacy wrapper for DataUtils.get_default_value()"""
    return DataUtils.get_default_value(d, k, default_key)

def get_abbr(val: Union[int, float], precision: int = 2, percentage: bool = False):
    """Legacy wrapper for DataUtils.format_abbreviation()"""
    return DataUtils.format_abbreviation(val, precision, percentage)

def get_mem(df: Union[pd.DataFrame, int], dataset: bool = True):
    """Legacy wrapper for DataUtils.get_memory_size()"""
    return DataUtils.get_memory_size(df, dataset)

def get_disk_mem(drive: str = 'C:/'):
    """Legacy wrapper for DataUtils.get_disk_usage()"""
    return DataUtils.get_disk_usage(drive)

def solve_ldap(ldap_dsn: str):
    """Legacy wrapper for DatabaseConnectionManager._solve_ldap()"""
    return DatabaseConnectionManager._solve_ldap(ldap_dsn)

def pcds_connect(service_name: str, ldap_service: str = 'ldap://oid.barcapint.com:4050'):
    """Legacy wrapper for DatabaseConnectionManager._get_pcds_connection()"""
    db_manager = DatabaseConnectionManager()
    return db_manager._get_pcds_connection(service_name, ldap_service)

def athena_connect(data_base: str, region_name: str = "us-east-1", 
                  work_group: str = "uscb-analytics",
                  s3_staging_dir: str = "s3://355538383407-us-east-1-athena-output/uscb-analytics/"):
    """Legacy wrapper for DatabaseConnectionManager._get_aws_connection()"""
    db_manager = DatabaseConnectionManager()
    return db_manager._get_aws_connection(data_base, region_name, work_group, s3_staging_dir)

def aws_creds_renew(seconds: int = 0, msg: str = 'AWS Credential Has Been Updated !'):
    """Legacy wrapper for AWSManager.renew_credentials()"""
    return AWSManager.renew_credentials(seconds, msg)

def s3_upload(path: str, s3_url: str, **kwargs):
    """Legacy wrapper for S3Manager.upload_file()"""
    return S3Manager.upload_file(path, s3_url, **kwargs)

def s3_correct_time(time_input, time_format: str = '%a, %d %b %Y %H:%M:%S %Z'):
    """Legacy wrapper for DateTimeUtils.correct_s3_time()"""
    return DateTimeUtils.correct_s3_time(time_input, time_format)

def s3_delete(s3_url: str):
    """Legacy wrapper for S3Manager.delete_object()"""
    return S3Manager.delete_object(s3_url)

def s3_metainfo(s3_url: str):
    """Legacy wrapper for S3Manager.get_metadata()"""
    return S3Manager.get_metadata(s3_url)

def s3_exist(s3_url: str):
    """Legacy wrapper for S3Manager.object_exists()"""
    return S3Manager.object_exists(s3_url)

def s3_load_df(s3_url: str, folder: bool = False, **kwargs):
    """Legacy wrapper for S3Manager.load_dataframe()"""
    return S3Manager.load_dataframe(s3_url, folder, **kwargs)

def s3_load_json(s3_url: str, **kwargs):
    """Legacy wrapper for S3Manager.load_json()"""
    return S3Manager.load_json(s3_url, **kwargs)

def s3_save_json(obj: Any, s3_url: str, **kwargs):
    """Legacy wrapper for S3Manager.save_json()"""
    return S3Manager.save_json(obj, s3_url, **kwargs)

def s3_save_df(df: pd.DataFrame, s3_url: str, **kwargs):
    """Legacy wrapper for S3Manager.save_dataframe()"""
    return S3Manager.save_dataframe(df, s3_url, **kwargs)

def s3_walk(s3_base_url: str, prefix: str = ''):
    """Legacy wrapper for S3Manager.walk_objects()"""
    return S3Manager.walk_objects(s3_base_url, prefix)

def download_froms3(s3_url: str, path: str, prefix: Optional[str] = None):
    """Legacy wrapper for S3Manager.download_folder()"""
    return S3Manager.download_folder(s3_url, path, prefix)

def read_meta_json(json_file: str):
    """Legacy wrapper for FileProcessor.read_meta_json()"""
    return FileProcessor.read_meta_json(json_file)

def read_input_excel(meta_table: MetaTable):
    """Legacy wrapper for FileProcessor.read_input_excel()"""
    return FileProcessor.read_input_excel(meta_table)

def process_mapping(column_map: ColumnMap):
    """Legacy wrapper for FileProcessor.process_column_mapping()"""
    return FileProcessor.process_column_mapping(column_map)

def get_s3url(s3_url: str, filepath: Optional[str] = None, 
             basename: Optional[str] = None, add_today: bool = False):
    """Legacy wrapper for URLUtils.get_s3_info()"""
    return URLUtils.get_s3_info(s3_url, filepath, basename, add_today)

@MEMORY.cache(cache_validation_callback=expires_after(hours=120), ignore=['connection'])
def costly_query(query_str: str, connection):
    """Legacy cached query function"""
    return psql.read_sql_query(query_str, connection)

# Rename App class to maintain compatibility
App = ApplicationManager

# Create SQLengine class for backward compatibility
class SQLengine(SQLEngine):
    """Backward compatibility wrapper for SQLEngine"""
    pass


# ===============================
# Module Exports
# ===============================

__all__ = [
    # New classes
    'LoggingUtils', 'DateTimeUtils', 'DataUtils', 'DatabaseConnectionManager',
    'AWSManager', 'S3Manager', 'QueryProcessor', 'FileProcessor', 'SQLEngine',
    'ApplicationManager', 'URLUtils', 'UDict', 'Timer',
    
    # Legacy functions for backward compatibility
    'start_run', 'end_run', 'seperator', 'get_datesort', 'get_datedrop',
    'get_default', 'get_abbr', 'get_mem', 'get_disk_mem', 'solve_ldap',
    'pcds_connect', 'athena_connect', 'aws_creds_renew', 's3_upload',
    's3_correct_time', 's3_delete', 's3_metainfo', 's3_exist', 's3_load_df',
    's3_load_json', 's3_save_json', 's3_save_df', 's3_walk', 'download_froms3',
    'read_meta_json', 'read_input_excel', 'process_mapping', 'get_s3url',
    'costly_query', 'App', 'SQLengine'
]