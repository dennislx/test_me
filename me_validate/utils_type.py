"""
Restructured Utils Type Definitions

This module contains clean, organized type definitions and data classes
used throughout the application. Only actively used types are included.
"""

import pprint as pp
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Dict, List, Optional, Union, Any
from pathlib import Path


# ===============================
# Type Aliases
# ===============================

PLATFORM = Literal['PCDS', 'AWS']
TPartition = Literal['none', 'year', 'year_month']


# ===============================
# Custom Exceptions
# ===============================

class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass


class NONEXIST_TABLE(DatabaseError):
    """Raised when a database table or view does not exist."""
    pass


class NONEXIST_DATEVAR(DatabaseError):
    """Raised when a date-like variable is not found in the table."""
    pass


# ===============================
# Utility Functions
# ===============================

def parse_string_list(list_str: str, separator: str = '\n') -> List[str]:
    """Parse a delimited string into a list of non-empty strings."""
    if not list_str:
        return []
    return [x.strip() for x in list_str.strip().split(separator) if x.strip()]


def parse_dict_string(dict_str: str, separator: str = '=') -> Dict[str, str]:
    """Parse a string with key=value pairs into a dictionary."""
    if not dict_str:
        return {}
    
    result = {}
    for line in parse_string_list(dict_str):
        if separator in line:
            key, value = line.split(separator, 1)
            result[key.strip()] = value.strip()
    return result


def col2type_mapping(col_str: str, type_str: str, separator: str = '; ') -> Dict[str, str]:
    """Create column to type mapping from delimited strings."""
    if not col_str or not type_str:
        return {}
    
    columns = col_str.split(separator)
    types = type_str.split(separator)
    
    return dict(zip(columns, types))


# ===============================
# Base Classes
# ===============================

@dataclass
class BaseConfig:
    """Base configuration class with logging and validation capabilities."""
    
    def to_log_string(self, indent: int = 1, padding: str = '') -> str:
        """Convert configuration to a formatted log string."""
        def format_value(value, pad):
            if isinstance(value, BaseConfig):
                return value.to_log_string(indent, pad)
            elif isinstance(value, Dict):
                return pp.pformat(value, indent=indent)
            else:
                return repr(value)
        
        class_name = self.__class__.__name__
        padding = padding + '\t' * indent
        
        field_strings = []
        for key, value in vars(self).items():
            formatted_value = format_value(value, padding)
            field_strings.append(f'{padding}{key}={formatted_value}')
        
        return f'{class_name}(\n' + ',\n'.join(field_strings) + '\n)'


# ===============================
# Range and Table Configuration
# ===============================

@dataclass
class ProcessingRange:
    """Defines a range of rows or items to process."""
    start: Optional[int] = None
    end: Optional[int] = None
    
    def __iter__(self):
        """Allow unpacking as start, end = range_obj."""
        return iter([self.start or 1, self.end or float('inf')])


@dataclass
class TableConfig(BaseConfig):
    """Configuration for reading Excel table inputs."""
    file: Path
    sheet: str
    skip_rows: int = 0
    select_cols: Dict[str, str] = field(default_factory=dict)
    select_rows: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.select_cols, str):
            self.select_cols = parse_dict_string(self.select_cols)
        if isinstance(self.select_rows, str):
            self.select_rows = parse_string_list(self.select_rows)


# ===============================
# Output Configuration
# ===============================

@dataclass
class CSVConfig:
    """Configuration for CSV output."""
    file: Path
    columns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.columns, str):
            self.columns = parse_string_list(self.columns)


@dataclass
class LogConfig:
    """Configuration for logging output."""
    level: Literal['debug', 'info', 'warning', 'error'] = 'info'
    format: str = '{time} | {level} | {message}'
    file: Optional[Path] = None
    overwrite: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logger configuration."""
        config = {
            'level': self.level.upper(),
            'format': self.format,
        }
        
        if self.file:
            config['sink'] = self.file
            config['mode'] = 'w' if self.overwrite else 'a'
        
        return config


@dataclass
class S3Config:
    """Configuration for S3 paths and operations."""
    run: Path = ""
    data: Path = ""
    stat: Path = ""
    misc: Path = ""
    temp: Path = ""


# ===============================
# Column Mapping Configuration
# ===============================

@dataclass
class ColumnMapping(BaseConfig):
    """Configuration for column mapping between systems."""
    to_json: Path
    file: Path
    na_string: str = ""
    overwrite: bool = False
    excludes: List[str] = field(default_factory=list)
    pcds_columns: List[str] = field(default_factory=list)
    aws_columns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Parse string inputs if provided
        if isinstance(self.excludes, str):
            self.excludes = parse_string_list(self.excludes)
        if isinstance(self.pcds_columns, str):
            self.pcds_columns = self._transform_column_names(self.pcds_columns)
        if isinstance(self.aws_columns, str):
            self.aws_columns = self._transform_column_names(self.aws_columns)
    
    def _transform_column_names(self, column_string: str) -> List[str]:
        """Transform column string to standardized column names."""
        columns = parse_string_list(column_string)
        return ['_'.join(col.lower().split()) for col in columns]


@dataclass
class MatchingConfig:
    """Configuration for column matching rules."""
    candidates: List[str] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)
    add_columns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.candidates, str):
            self.candidates = parse_string_list(self.candidates)
        # Convert dict inputs to lists if needed
        if isinstance(self.drop_columns, dict):
            self.drop_columns = list(self.drop_columns.keys())
        if isinstance(self.add_columns, dict):
            self.add_columns = list(self.add_columns.keys())


# ===============================
# Input Configuration
# ===============================

@dataclass
class InputConfig(BaseConfig):
    """Base input configuration."""
    name: str
    step: str
    env_file: Path
    processing_range: ProcessingRange = field(default_factory=ProcessingRange)
    
    def __post_init__(self):
        if isinstance(self.processing_range, dict):
            self.processing_range = ProcessingRange(**self.processing_range)


@dataclass
class MetaInputConfig(InputConfig):
    """Input configuration for meta analysis."""
    table_config: TableConfig = field(default_factory=TableConfig)
    clear_cache: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.table_config, dict):
            self.table_config = TableConfig(**self.table_config)


@dataclass
class StatInputConfig(InputConfig):
    """Input configuration for statistics analysis."""
    meta_step: str = ""
    pull_step: str = ""
    csv_file: Path = None
    json_config: Dict[str, Path] = field(default_factory=dict)
    select_rows: List[str] = field(default_factory=list)
    debug_mode: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.select_rows, str):
            self.select_rows = [x.lower() for x in parse_string_list(self.select_rows)]


# ===============================
# Output Configuration
# ===============================

@dataclass
class OutputConfig(BaseConfig):
    """Base output configuration."""
    folder: Path
    pickle_file: Path
    csv_config: CSVConfig = field(default_factory=CSVConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    s3_config: S3Config = field(default_factory=S3Config)
    
    def __post_init__(self):
        if isinstance(self.csv_config, dict):
            self.csv_config = CSVConfig(**self.csv_config)
        if isinstance(self.log_config, dict):
            self.log_config = LogConfig(**self.log_config)
        if isinstance(self.s3_config, dict):
            self.s3_config = S3Config(**self.s3_config)


@dataclass
class MetaOutputConfig(OutputConfig):
    """Output configuration for meta analysis."""
    next_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.next_config, str):
            self.next_config = parse_dict_string(self.next_config)


@dataclass
class StatOutputConfig(OutputConfig):
    """Output configuration for statistics analysis."""
    drop_na_values: bool = False
    reuse_aws_data: bool = False


# ===============================
# Main Configuration Classes
# ===============================

@dataclass
class MetaConfig(BaseConfig):
    """Complete configuration for meta analysis."""
    input: MetaInputConfig
    output: MetaOutputConfig
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    column_maps: ColumnMapping = field(default_factory=ColumnMapping)
    
    def __post_init__(self):
        if isinstance(self.input, dict):
            self.input = MetaInputConfig(**self.input)
        if isinstance(self.output, dict):
            self.output = MetaOutputConfig(**self.output)
        if isinstance(self.matching, dict):
            self.matching = MatchingConfig(**self.matching)
        if isinstance(self.column_maps, dict):
            self.column_maps = ColumnMapping(**self.column_maps)


@dataclass
class StatConfig(BaseConfig):
    """Complete configuration for statistics analysis."""
    input: StatInputConfig
    output: StatOutputConfig
    
    def __post_init__(self):
        if isinstance(self.input, dict):
            self.input = StatInputConfig(**self.input)
        if isinstance(self.output, dict):
            self.output = StatOutputConfig(**self.output)


# ===============================
# Data Transfer Objects
# ===============================

@dataclass
class MetaInfo:
    """Metadata information for a table."""
    column_mapping: Dict[str, str] = field(default_factory=dict)
    column_types: Dict[str, str] = field(default_factory=dict)
    info_string: str = ""
    row_variable: str = ""
    excluded_rows: List[str] = field(default_factory=list)
    row_count: int = 0
    
    def update(self, **kwargs):
        """Update fields with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class MetaJSON:
    """Complete metadata for PCDS and AWS comparison."""
    pcds: MetaInfo = field(default_factory=MetaInfo)
    aws: MetaInfo = field(default_factory=MetaInfo)
    pcds_table: str = ""
    aws_table: str = ""
    time_excludes: List[str] = field(default_factory=list)
    last_modified: str = ""
    
    def __init__(self, 
                 pcds_cols: str = "", pcds_types: str = "", pcds_nrows: int = 0, pcds_id: str = "",
                 aws_cols: str = "", aws_types: str = "", aws_nrows: int = 0, aws_id: str = "",
                 **kwargs):
        """Initialize from string parameters (for backward compatibility)."""
        
        # Handle other fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Parse time excludes
        if hasattr(self, 'time_excludes') and isinstance(self.time_excludes, str):
            self.time_excludes = parse_string_list(self.time_excludes, '; ')
        
        # Create PCDS metadata
        self.pcds = MetaInfo(
            column_mapping=col2type_mapping(pcds_cols, aws_cols),
            column_types=col2type_mapping(pcds_cols, pcds_types),
            info_string=getattr(self, 'pcds_table', ''),
            row_variable=pcds_id,
            excluded_rows=getattr(self, 'time_excludes', []),
            row_count=pcds_nrows
        )
        
        # Create AWS metadata
        self.aws = MetaInfo(
            column_mapping=col2type_mapping(aws_cols, pcds_cols),
            column_types=col2type_mapping(aws_cols, aws_types),
            info_string=getattr(self, 'aws_table', ''),
            row_variable=aws_id,
            excluded_rows=getattr(self, 'time_excludes', []),
            row_count=aws_nrows
        )


@dataclass
class MetaMergeResult:
    """Result of merging PCDS and AWS metadata."""
    unique_pcds: List[str] = field(default_factory=list)
    unique_aws: List[str] = field(default_factory=list)
    column_mapping: Optional[pd.DataFrame] = None
    type_mismatches: str = ""
    uncaptured_mappings: str = ""


@dataclass
class TimeRange:
    """Represents a time range for data processing."""
    start_date: str
    end_date: str
    
    def __str__(self) -> str:
        return f'{self.start_date}_{self.end_date}'


# ===============================
# Pull Configuration (Simplified)
# ===============================

@dataclass
class PullDataConfig:
    """Configuration for data pulling operations."""
    where_conditions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    s3_partitioning: Dict[str, str] = field(default_factory=dict)
    delete_existing: Dict[str, bool] = field(default_factory=dict)
    date_formats: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def __call__(self, table_name: str):
        """Set current table name for configuration lookup."""
        self.current_table = table_name
        return self
    
    @property
    def should_delete(self) -> bool:
        """Check if existing data should be deleted for current table."""
        default = self.delete_existing.get('default', False)
        return self.delete_existing.get(getattr(self, 'current_table', ''), default)
    
    @property
    def partition_type(self) -> TPartition:
        """Get partition type for current table."""
        default = self.s3_partitioning.get('default', 'none')
        return self.s3_partitioning.get(getattr(self, 'current_table', ''), default)
    
    def get_where_condition(self, platform: PLATFORM) -> str:
        """Get WHERE condition for specified platform."""
        platform_conditions = self.where_conditions.get(platform, {})
        default = platform_conditions.get('default', '')
        return platform_conditions.get(getattr(self, 'current_table', ''), default)
    
    def get_date_format(self, platform: PLATFORM) -> str:
        """Get date format for specified platform."""
        platform_formats = self.date_formats.get(platform, {})
        default = platform_formats.get('default', '%Y-%m-%d')
        return platform_formats.get(getattr(self, 'current_table', ''), default)


@dataclass
class PullConfig(BaseConfig):
    """Complete configuration for data pulling operations."""
    input: InputConfig
    output: OutputConfig
    pull_data: PullDataConfig = field(default_factory=PullDataConfig)
    
    def __post_init__(self):
        if isinstance(self.input, dict):
            self.input = InputConfig(**self.input)
        if isinstance(self.output, dict):
            self.output = OutputConfig(**self.output)
        if isinstance(self.pull_data, dict):
            self.pull_data = PullDataConfig(**self.pull_data)


# ===============================
# Legacy Aliases for Backward Compatibility
# ===============================

# Keep old names for backward compatibility
MetaTable = TableConfig
ColumnMap = ColumnMapping
MetaMatch = MatchingConfig
Trange = TimeRange
BaseType = BaseConfig

# Legacy function aliases
read_str_lst = parse_string_list
read_dstr_lst = parse_dict_string


# ===============================
# Module Exports
# ===============================

__all__ = [
    # Type aliases
    'PLATFORM', 'TPartition',
    
    # Exceptions
    'DatabaseError', 'NONEXIST_TABLE', 'NONEXIST_DATEVAR',
    
    # Utility functions
    'parse_string_list', 'parse_dict_string', 'col2type_mapping',
    
    # Base classes
    'BaseConfig', 'ProcessingRange',
    
    # Configuration classes
    'TableConfig', 'CSVConfig', 'LogConfig', 'S3Config',
    'ColumnMapping', 'MatchingConfig',
    'InputConfig', 'MetaInputConfig', 'StatInputConfig',
    'OutputConfig', 'MetaOutputConfig', 'StatOutputConfig',
    'MetaConfig', 'StatConfig', 'PullConfig',
    
    # Data classes
    'MetaInfo', 'MetaJSON', 'MetaMergeResult', 'TimeRange', 'PullDataConfig',
    
    # Legacy aliases
    'MetaTable', 'ColumnMap', 'MetaMatch', 'Trange', 'BaseType',
    'read_str_lst', 'read_dstr_lst'
]