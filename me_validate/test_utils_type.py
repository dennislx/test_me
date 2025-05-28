"""
Unit tests for the restructured utils_type module.
"""

import pytest
import pandas as pd
from pathlib import Path
from dataclasses import asdict

# Import the classes from the restructured module
from utils_type import (
    # Utility functions
    parse_string_list, parse_dict_string, col2type_mapping,
    
    # Exceptions
    NONEXIST_TABLE, NONEXIST_DATEVAR, DatabaseError,
    
    # Configuration classes
    ProcessingRange, TableConfig, CSVConfig, LogConfig, S3Config,
    ColumnMapping, MatchingConfig, InputConfig, MetaInputConfig,
    StatInputConfig, OutputConfig, MetaOutputConfig, StatOutputConfig,
    MetaConfig, StatConfig, PullConfig,
    
    # Data classes
    MetaInfo, MetaJSON, MetaMergeResult, TimeRange, PullDataConfig,
    
    # Base classes
    BaseConfig
)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_parse_string_list_basic(self):
        """Test basic string list parsing."""
        input_str = "item1\nitem2\nitem3"
        result = parse_string_list(input_str)
        assert result == ["item1", "item2", "item3"]
    
    def test_parse_string_list_with_separator(self):
        """Test string list parsing with custom separator."""
        input_str = "item1;item2;item3"
        result = parse_string_list(input_str, separator=';')
        assert result == ["item1", "item2", "item3"]
    
    def test_parse_string_list_empty(self):
        """Test parsing empty string."""
        assert parse_string_list("") == []
        assert parse_string_list(None) == []
    
    def test_parse_string_list_with_whitespace(self):
        """Test parsing with extra whitespace."""
        input_str = "  item1  \n  item2  \n  item3  "
        result = parse_string_list(input_str)
        assert result == ["item1", "item2", "item3"]
    
    def test_parse_dict_string_basic(self):
        """Test basic dictionary string parsing."""
        input_str = "key1=value1\nkey2=value2"
        result = parse_dict_string(input_str)
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_parse_dict_string_with_separator(self):
        """Test dictionary parsing with custom separator."""
        input_str = "key1:value1\nkey2:value2"
        result = parse_dict_string(input_str, separator=':')
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_parse_dict_string_empty(self):
        """Test parsing empty dictionary string."""
        assert parse_dict_string("") == {}
        assert parse_dict_string(None) == {}
    
    def test_parse_dict_string_with_whitespace(self):
        """Test dictionary parsing with whitespace."""
        input_str = "  key1 = value1  \n  key2 = value2  "
        result = parse_dict_string(input_str)
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_col2type_mapping_basic(self):
        """Test column to type mapping."""
        cols = "col1; col2; col3"
        types = "VARCHAR; NUMBER; DATE"
        result = col2type_mapping(cols, types)
        assert result == {"col1": "VARCHAR", "col2": "NUMBER", "col3": "DATE"}
    
    def test_col2type_mapping_empty(self):
        """Test column to type mapping with empty inputs."""
        assert col2type_mapping("", "") == {}
        assert col2type_mapping(None, None) == {}


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_database_error_hierarchy(self):
        """Test exception hierarchy."""
        assert issubclass(NONEXIST_TABLE, DatabaseError)
        assert issubclass(NONEXIST_DATEVAR, DatabaseError)
    
    def test_nonexist_table_exception(self):
        """Test NONEXIST_TABLE exception."""
        with pytest.raises(NONEXIST_TABLE):
            raise NONEXIST_TABLE("Table not found")
    
    def test_nonexist_datevar_exception(self):
        """Test NONEXIST_DATEVAR exception."""
        with pytest.raises(NONEXIST_DATEVAR):
            raise NONEXIST_DATEVAR("Date variable not found")


class TestProcessingRange:
    """Test cases for ProcessingRange class."""
    
    def test_processing_range_default(self):
        """Test ProcessingRange with default values."""
        range_obj = ProcessingRange()
        assert range_obj.start is None
        assert range_obj.end is None
    
    def test_processing_range_with_values(self):
        """Test ProcessingRange with specified values."""
        range_obj = ProcessingRange(start=1, end=100)
        assert range_obj.start == 1
        assert range_obj.end == 100
    
    def test_processing_range_iteration(self):
        """Test ProcessingRange iteration."""
        range_obj = ProcessingRange(start=10, end=20)
        start, end = range_obj
        assert start == 10
        assert end == 20
    
    def test_processing_range_iteration_defaults(self):
        """Test ProcessingRange iteration with defaults."""
        range_obj = ProcessingRange()
        start, end = range_obj
        assert start == 1
        assert end == float('inf')


class TestTableConfig:
    """Test cases for TableConfig class."""
    
    def test_table_config_basic(self):
        """Test basic TableConfig creation."""
        config = TableConfig(
            file=Path("test.xlsx"),
            sheet="Sheet1",
            skip_rows=1
        )
        assert config.file == Path("test.xlsx")
        assert config.sheet == "Sheet1"
        assert config.skip_rows == 1
    
    def test_table_config_with_string_inputs(self):
        """Test TableConfig with string inputs that need parsing."""
        config = TableConfig(
            file=Path("test.xlsx"),
            sheet="Sheet1",
            select_cols="old_col=new_col\nold_col2=new_col2",
            select_rows="row1\nrow2\nrow3"
        )
        assert config.select_cols == {"old_col": "new_col", "old_col2": "new_col2"}
        assert config.select_rows == ["row1", "row2", "row3"]


class TestCSVConfig:
    """Test cases for CSVConfig class."""
    
    def test_csv_config_basic(self):
        """Test basic CSVConfig creation."""
        config = CSVConfig(file=Path("output.csv"))
        assert config.file == Path("output.csv")
        assert config.columns == []
    
    def test_csv_config_with_string_columns(self):
        """Test CSVConfig with string column input."""
        config = CSVConfig(
            file=Path("output.csv"),
            columns="col1\ncol2\ncol3"
        )
        assert config.columns == ["col1", "col2", "col3"]


class TestLogConfig:
    """Test cases for LogConfig class."""
    
    def test_log_config_defaults(self):
        """Test LogConfig with default values."""
        config = LogConfig()
        assert config.level == 'info'
        assert config.overwrite is True
        assert config.file is None
    
    def test_log_config_to_dict(self):
        """Test LogConfig conversion to dictionary."""
        config = LogConfig(
            level='debug',
            format='{time} - {message}',
            file=Path('app.log'),
            overwrite=False
        )
        result = config.to_dict()
        
        expected = {
            'level': 'DEBUG',
            'format': '{time} - {message}',
            'sink': Path('app.log'),
            'mode': 'a'
        }
        assert result == expected
    
    def test_log_config_to_dict_no_file(self):
        """Test LogConfig to_dict without file."""
        config = LogConfig(level='warning')
        result = config.to_dict()
        
        assert 'sink' not in result
        assert 'mode' not in result
        assert result['level'] == 'WARNING'


class TestColumnMapping:
    """Test cases for ColumnMapping class."""
    
    def test_column_mapping_basic(self):
        """Test basic ColumnMapping creation."""
        config = ColumnMapping(
            to_json=Path("mapping.json"),
            file=Path("source.xlsx")
        )
        assert config.to_json == Path("mapping.json")
        assert config.file == Path("source.xlsx")
        assert config.overwrite is False
    
    def test_column_mapping_with_string_inputs(self):
        """Test ColumnMapping with string inputs."""
        config = ColumnMapping(
            to_json=Path("mapping.json"),
            file=Path("source.xlsx"),
            excludes="sheet1\nsheet2",
            pcds_columns="PCDS Column 1\nPCDS Column 2",
            aws_columns="AWS Column 1\nAWS Column 2"
        )
        assert config.excludes == ["sheet1", "sheet2"]
        assert config.pcds_columns == ["pcds_column_1", "pcds_column_2"]
        assert config.aws_columns == ["aws_column_1", "aws_column_2"]


class TestMatchingConfig:
    """Test cases for MatchingConfig class."""
    
    def test_matching_config_basic(self):
        """Test basic MatchingConfig creation."""
        config = MatchingConfig()
        assert config.candidates == []
        assert config.drop_columns == []
        assert config.add_columns == []
    
    def test_matching_config_with_strings(self):
        """Test MatchingConfig with string inputs."""
        config = MatchingConfig(
            candidates="candidate1\ncandidate2",
            drop_columns={"col1": "reason1", "col2": "reason2"},
            add_columns={"col3": "reason3"}
        )
        assert config.candidates == ["candidate1", "candidate2"]
        assert config.drop_columns == ["col1", "col2"]
        assert config.add_columns == ["col3"]


class TestInputConfig:
    """Test cases for InputConfig class."""
    
    def test_input_config_basic(self):
        """Test basic InputConfig creation."""
        config = InputConfig(
            name="test_analysis",
            step="step1",
            env_file=Path(".env")
        )
        assert config.name == "test_analysis"
        assert config.step == "step1"
        assert config.env_file == Path(".env")
        assert isinstance(config.processing_range, ProcessingRange)
    
    def test_input_config_with_range_dict(self):
        """Test InputConfig with range as dictionary."""
        config = InputConfig(
            name="test",
            step="step1",
            env_file=Path(".env"),
            processing_range={"start": 10, "end": 100}
        )
        assert config.processing_range.start == 10
        assert config.processing_range.end == 100


class TestMetaInputConfig:
    """Test cases for MetaInputConfig class."""
    
    def test_meta_input_config_basic(self):
        """Test basic MetaInputConfig creation."""
        config = MetaInputConfig(
            name="meta_analysis",
            step="meta_step",
            env_file=Path(".env")
        )
        assert config.name == "meta_analysis"
        assert isinstance(config.table_config, TableConfig)
        assert config.clear_cache is True
    
    def test_meta_input_config_with_table_dict(self):
        """Test MetaInputConfig with table config as dictionary."""
        config = MetaInputConfig(
            name="meta_analysis",
            step="meta_step",
            env_file=Path(".env"),
            table_config={
                "file": Path("input.xlsx"),
                "sheet": "Data",
                "skip_rows": 2
            }
        )
        assert config.table_config.file == Path("input.xlsx")
        assert config.table_config.sheet == "Data"
        assert config.table_config.skip_rows == 2


class TestStatInputConfig:
    """Test cases for StatInputConfig class."""
    
    def test_stat_input_config_basic(self):
        """Test basic StatInputConfig creation."""
        config = StatInputConfig(
            name="stat_analysis",
            step="stat_step",
            env_file=Path(".env")
        )
        assert config.name == "stat_analysis"
        assert config.debug_mode is False
        assert config.select_rows == []
    
    def test_stat_input_config_with_select_rows(self):
        """Test StatInputConfig with select_rows as string."""
        config = StatInputConfig(
            name="stat_analysis",
            step="stat_step",
            env_file=Path(".env"),
            select_rows="ROW1\nROW2\nROW3"
        )
        assert config.select_rows == ["row1", "row2", "row3"]


class TestOutputConfig:
    """Test cases for OutputConfig class."""
    
    def test_output_config_basic(self):
        """Test basic OutputConfig creation."""
        config = OutputConfig(
            folder=Path("output/"),
            pickle_file=Path("results.pkl")
        )
        assert config.folder == Path("output/")
        assert config.pickle_file == Path("results.pkl")
        assert isinstance(config.csv_config, CSVConfig)
        assert isinstance(config.log_config, LogConfig)
        assert isinstance(config.s3_config, S3Config)
    
    def test_output_config_with_nested_dicts(self):
        """Test OutputConfig with nested configurations as dictionaries."""
        config = OutputConfig(
            folder=Path("output/"),
            pickle_file=Path("results.pkl"),
            csv_config={"file": Path("output.csv"), "columns": "col1\ncol2"},
            log_config={"level": "debug", "file": Path("app.log")},
            s3_config={"run": Path("s3://bucket/run/"), "data": Path("s3://bucket/data/")}
        )
        assert config.csv_config.file == Path("output.csv")
        assert config.csv_config.columns == ["col1", "col2"]
        assert config.log_config.level == "debug"
        assert config.log_config.file == Path("app.log")
        assert config.s3_config.run == Path("s3://bucket/run/")


class TestMetaConfig:
    """Test cases for MetaConfig class."""
    
    def test_meta_config_basic(self):
        """Test basic MetaConfig creation."""
        config = MetaConfig(
            input=MetaInputConfig(
                name="test",
                step="meta",
                env_file=Path(".env")
            ),
            output=MetaOutputConfig(
                folder=Path("output/"),
                pickle_file=Path("meta.pkl")
            )
        )
        assert isinstance(config.input, MetaInputConfig)
        assert isinstance(config.output, MetaOutputConfig)
        assert isinstance(config.matching, MatchingConfig)
        assert isinstance(config.column_maps, ColumnMapping)
    
    def test_meta_config_with_dicts(self):
        """Test MetaConfig with dictionary inputs."""
        config = MetaConfig(
            input={
                "name": "test",
                "step": "meta",
                "env_file": Path(".env")
            },
            output={
                "folder": Path("output/"),
                "pickle_file": Path("meta.pkl")
            },
            matching={
                "candidates": "cand1\ncand2"
            },
            column_maps={
                "to_json": Path("mapping.json"),
                "file": Path("source.xlsx")
            }
        )
        assert config.input.name == "test"
        assert config.output.folder == Path("output/")
        assert config.matching.candidates == ["cand1", "cand2"]
        assert config.column_maps.to_json == Path("mapping.json")


class TestStatConfig:
    """Test cases for StatConfig class."""
    
    def test_stat_config_basic(self):
        """Test basic StatConfig creation."""
        config = StatConfig(
            input=StatInputConfig(
                name="test",
                step="stat",
                env_file=Path(".env")
            ),
            output=StatOutputConfig(
                folder=Path("output/"),
                pickle_file=Path("stat.pkl")
            )
        )
        assert isinstance(config.input, StatInputConfig)
        assert isinstance(config.output, StatOutputConfig)
    
    def test_stat_config_with_dicts(self):
        """Test StatConfig with dictionary inputs."""
        config = StatConfig(
            input={
                "name": "test",
                "step": "stat",
                "env_file": Path(".env"),
                "debug_mode": True
            },
            output={
                "folder": Path("output/"),
                "pickle_file": Path("stat.pkl"),
                "drop_na_values": True
            }
        )
        assert config.input.debug_mode is True
        assert config.output.drop_na_values is True


class TestMetaInfo:
    """Test cases for MetaInfo class."""
    
    def test_meta_info_default(self):
        """Test MetaInfo with default values."""
        info = MetaInfo()
        assert info.column_mapping == {}
        assert info.column_types == {}
        assert info.row_count == 0
    
    def test_meta_info_with_values(self):
        """Test MetaInfo with specified values."""
        info = MetaInfo(
            column_mapping={"col1": "COL1"},
            column_types={"col1": "VARCHAR"},
            row_count=1000
        )
        assert info.column_mapping == {"col1": "COL1"}
        assert info.column_types == {"col1": "VARCHAR"}
        assert info.row_count == 1000
    
    def test_meta_info_update(self):
        """Test MetaInfo update method."""
        info = MetaInfo()
        info.update(row_count=500, info_string="test_table")
        assert info.row_count == 500
        assert info.info_string == "test_table"


class TestMetaJSON:
    """Test cases for MetaJSON class."""
    
    def test_meta_json_basic(self):
        """Test basic MetaJSON creation."""
        meta = MetaJSON()
        assert isinstance(meta.pcds, MetaInfo)
        assert isinstance(meta.aws, MetaInfo)
        assert meta.time_excludes == []
    
    def test_meta_json_with_string_params(self):
        """Test MetaJSON with string parameters (backward compatibility)."""
        meta = MetaJSON(
            pcds_cols="col1; col2",
            pcds_types="VARCHAR; NUMBER",
            pcds_nrows=1000,
            pcds_id="ID_COL",
            aws_cols="col1; col2",
            aws_types="varchar; decimal",
            aws_nrows=1000,
            aws_id="id_col",
            pcds_table="service.table",
            aws_table="db.table",
            time_excludes="2023-01-01; 2023-01-02"
        )
        
        assert meta.pcds.column_mapping == {"col1": "col1", "col2": "col2"}
        assert meta.pcds.column_types == {"col1": "VARCHAR", "col2": "NUMBER"}
        assert meta.pcds.row_count == 1000
        assert meta.pcds.row_variable == "ID_COL"
        
        assert meta.aws.column_mapping == {"col1": "col1", "col2": "col2"}
        assert meta.aws.column_types == {"col1": "varchar", "col2": "decimal"}
        assert meta.aws.row_count == 1000
        assert meta.aws.row_variable == "id_col"
        
        assert meta.time_excludes == ["2023-01-01", "2023-01-02"]


class TestTimeRange:
    """Test cases for TimeRange class."""
    
    def test_time_range_basic(self):
        """Test basic TimeRange creation."""
        time_range = TimeRange(start_date="2023-01-01", end_date="2023-01-31")
        assert time_range.start_date == "2023-01-01"
        assert time_range.end_date == "2023-01-31"
    
    def test_time_range_string_representation(self):
        """Test TimeRange string representation."""
        time_range = TimeRange(start_date="2023-01-01", end_date="2023-01-31")
        assert str(time_range) == "2023-01-01_2023-01-31"


class TestPullDataConfig:
    """Test cases for PullDataConfig class."""
    
    def test_pull_data_config_basic(self):
        """Test basic PullDataConfig creation."""
        config = PullDataConfig()
        assert config.where_conditions == {}
        assert config.s3_partitioning == {}
        assert config.delete_existing == {}
        assert config.date_formats == {}
    
    def test_pull_data_config_call_method(self):
        """Test PullDataConfig call method for table selection."""
        config = PullDataConfig()
        result = config("test_table")
        assert result is config
        assert config.current_table == "test_table"
    
    def test_pull_data_config_should_delete(self):
        """Test should_delete property."""
        config = PullDataConfig(
            delete_existing={
                "default": False,
                "table1": True
            }
        )
        
        config("table1")
        assert config.should_delete is True
        
        config("table2")
        assert config.should_delete is False
    
    def test_pull_data_config_partition_type(self):
        """Test partition_type property."""
        config = PullDataConfig(
            s3_partitioning={
                "default": "none",
                "table1": "year"
            }
        )
        
        config("table1")
        assert config.partition_type == "year"
        
        config("table2")
        assert config.partition_type == "none"
    
    def test_pull_data_config_where_condition(self):
        """Test get_where_condition method."""
        config = PullDataConfig(
            where_conditions={
                "PCDS": {
                    "default": "1=1",
                    "table1": "active = 'Y'"
                }
            }
        )
        
        config("table1")
        assert config.get_where_condition("PCDS") == "active = 'Y'"
        
        config("table2")
        assert config.get_where_condition("PCDS") == "1=1"
    
    def test_pull_data_config_date_format(self):
        """Test get_date_format method."""
        config = PullDataConfig(
            date_formats={
                "AWS": {
                    "default": "%Y-%m-%d",
                    "table1": "%Y/%m/%d"
                }
            }
        )
        
        config("table1")
        assert config.get_date_format("AWS") == "%Y/%m/%d"
        
        config("table2")
        assert config.get_date_format("AWS") == "%Y-%m-%d"


class TestBaseConfig:
    """Test cases for BaseConfig class."""
    
    def test_base_config_to_log_string(self):
        """Test BaseConfig to_log_string method."""
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "test"
            value: int = 42
        
        config = TestConfig()
        log_string = config.to_log_string()
        
        assert "TestConfig(" in log_string
        assert "name='test'" in log_string
        assert "value=42" in log_string
    
    def test_base_config_nested_to_log_string(self):
        """Test BaseConfig with nested configuration."""
        
        @dataclass
        class NestedConfig(BaseConfig):
            nested_value: str = "nested"
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "test"
            nested: NestedConfig = None
        
        nested = NestedConfig()
        config = TestConfig(nested=nested)
        log_string = config.to_log_string()
        
        assert "TestConfig(" in log_string
        assert "NestedConfig(" in log_string
        assert "nested_value='nested'" in log_string


class TestMetaMergeResult:
    """Test cases for MetaMergeResult class."""
    
    def test_meta_merge_result_default(self):
        """Test MetaMergeResult with default values."""
        result = MetaMergeResult()
        assert result.unique_pcds == []
        assert result.unique_aws == []
        assert result.column_mapping is None
        assert result.type_mismatches == ""
        assert result.uncaptured_mappings == ""
    
    def test_meta_merge_result_with_values(self):
        """Test MetaMergeResult with specified values."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        result = MetaMergeResult(
            unique_pcds=["col3", "col4"],
            unique_aws=["col5"],
            column_mapping=df,
            type_mismatches="VARCHAR->varchar",
            uncaptured_mappings="COL1->col1"
        )
        
        assert result.unique_pcds == ["col3", "col4"]
        assert result.unique_aws == ["col5"]
        pd.testing.assert_frame_equal(result.column_mapping, df)
        assert result.type_mismatches == "VARCHAR->varchar"
        assert result.uncaptured_mappings == "COL1->col1"


# Integration tests
class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_meta_config_complete_setup(self):
        """Test complete MetaConfig setup from dictionaries."""
        config_dict = {
            "input": {
                "name": "integration_test",
                "step": "meta_step",
                "env_file": Path(".env"),
                "table_config": {
                    "file": Path("input.xlsx"),
                    "sheet": "Data",
                    "select_cols": "old=new\nold2=new2"
                },
                "processing_range": {"start": 1, "end": 50}
            },
            "output": {
                "folder": Path("output/"),
                "pickle_file": Path("meta.pkl"),
                "csv_config": {"file": Path("results.csv")},
                "log_config": {"level": "debug"}
            },
            "matching": {
                "candidates": "cand1\ncand2"
            },
            "column_maps": {
                "to_json": Path("mapping.json"),
                "file": Path("source.xlsx")
            }
        }
        
        config = MetaConfig(**config_dict)
        
        # Verify nested structure is properly created
        assert config.input.name == "integration_test"
        assert config.input.table_config.select_cols == {"old": "new", "old2": "new2"}
        assert config.input.processing_range.start == 1
        assert config.output.log_config.level == "debug"
        assert config.matching.candidates == ["cand1", "cand2"]
    
    def test_meta_json_backward_compatibility(self):
        """Test MetaJSON backward compatibility with old parameter format."""
        # This tests the interface that existing code expects
        old_params = {
            "pcds_cols": "ID; NAME; DATE_COL",
            "pcds_types": "NUMBER; VARCHAR2(50); DATE",
            "pcds_nrows": 5000,
            "pcds_id": "DATE_COL",
            "aws_cols": "id; name; date_col",
            "aws_types": "decimal; varchar(50); date",
            "aws_nrows": 5000,
            "aws_id": "date_col",
            "pcds_table": "service.my_table",
            "aws_table": "database.my_table",
            "time_excludes": "2023-01-01; 2023-01-02; 2023-01-03",
            "last_modified": "2023-12-01"
        }
        
        meta = MetaJSON(**old_params)
        
        # Verify PCDS configuration
        assert "ID" in meta.pcds.column_types
        assert meta.pcds.column_types["ID"] == "NUMBER"
        assert meta.pcds.row_count == 5000
        assert meta.pcds.row_variable == "DATE_COL"
        
        # Verify AWS configuration
        assert "id" in meta.aws.column_types
        assert meta.aws.column_types["id"] == "decimal"
        assert meta.aws.row_count == 5000
        assert meta.aws.row_variable == "date_col"
        
        # Verify time excludes parsing
        assert len(meta.time_excludes) == 3
        assert "2023-01-01" in meta.time_excludes


# Error handling tests
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_column_mapping_invalid_transform(self):
        """Test ColumnMapping with invalid column string."""
        # Should handle gracefully without raising exceptions
        config = ColumnMapping(
            to_json=Path("test.json"),
            file=Path("test.xlsx"),
            pcds_columns=""  # Empty string should result in empty list
        )
        assert config.pcds_columns == []
    
    def test_meta_json_partial_data(self):
        """Test MetaJSON with missing data."""
        # Should handle missing parameters gracefully
        meta = MetaJSON(
            pcds_cols="col1",
            pcds_types="VARCHAR",
            pcds_nrows=100
            # Missing other required parameters
        )
        
        # Should still create valid MetaInfo objects
        assert isinstance(meta.pcds, MetaInfo)
        assert isinstance(meta.aws, MetaInfo)
        assert meta.pcds.row_count == 100


# Performance tests
class TestPerformance:
    """Performance-related tests."""
    
    def test_large_config_creation(self):
        """Test performance with large configuration objects."""
        import time
        
        # Create a large configuration
        large_select_cols = {f"old_col_{i}": f"new_col_{i}" for i in range(1000)}
        
        start_time = time.time()
        config = TableConfig(
            file=Path("large.xlsx"),
            sheet="Data",
            select_cols=large_select_cols
        )
        creation_time = time.time() - start_time
        
        assert creation_time < 0.1  # Should complete quickly
        assert len(config.select_cols) == 1000
    
    def test_meta_json_large_mappings(self):
        """Test MetaJSON with large column mappings."""
        import time
        
        # Create large column lists
        large_cols = "; ".join([f"col_{i}" for i in range(500)])
        large_types = "; ".join(["VARCHAR"] * 500)
        
        start_time = time.time()
        meta = MetaJSON(
            pcds_cols=large_cols,
            pcds_types=large_types,
            pcds_nrows=1000000,
            aws_cols=large_cols,
            aws_types=large_types,
            aws_nrows=1000000
        )
        creation_time = time.time() - start_time
        
        assert creation_time < 0.5  # Should complete reasonably quickly
        assert len(meta.pcds.column_types) == 500
        assert len(meta.aws.column_types) == 500


# Parametrized tests
@pytest.mark.parametrize("input_str,separator,expected", [
    ("a\nb\nc", '\n', ["a", "b", "c"]),
    ("a;b;c", ';', ["a", "b", "c"]),
    ("a,b,c", ',', ["a", "b", "c"]),
    ("", '\n', []),
    ("  a  \n  b  ", '\n', ["a", "b"]),
])
def test_parse_string_list_parametrized(input_str, separator, expected):
    """Parametrized test for parse_string_list function."""
    result = parse_string_list(input_str, separator)
    assert result == expected


@pytest.mark.parametrize("level,expected_upper", [
    ("debug", "DEBUG"),
    ("info", "INFO"),
    ("warning", "WARNING"),
    ("error", "ERROR"),
])
def test_log_config_level_conversion(level, expected_upper):
    """Parametrized test for LogConfig level conversion."""
    config = LogConfig(level=level)
    result = config.to_dict()
    assert result['level'] == expected_upper


@pytest.mark.parametrize("partition_type", ["none", "year", "year_month"])
def test_pull_data_config_partition_types(partition_type):
    """Parametrized test for PullDataConfig partition types."""
    config = PullDataConfig(
        s3_partitioning={"default": partition_type}
    )
    config("test_table")
    assert config.partition_type == partition_type


if __name__ == '__main__':
    # Run specific test classes or all tests
    pytest.main([
        __file__, 
        '-v',  # verbose output
        '--tb=short',  # shorter traceback format
        '--cov=utils_type',  # coverage report
        '--cov-report=html'  # HTML coverage report
    ])