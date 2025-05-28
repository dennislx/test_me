"""
Unit tests for the restructured upload module.
"""

import pytest
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from dataclasses import asdict
import datetime as dt

# Import the classes from the restructured module
import do_upload
from do_upload import (
    UploadMetrics, PCDSConnectionManager, QueryBuilder, DataUploader,
    PartitionManager, UploadProcessor, create_argument_parser, main
)
import utils_type as ut
import utils


class TestUploadMetrics:
    """Test cases for UploadMetrics class."""
    
    def test_upload_metrics_default(self):
        """Test UploadMetrics with default values."""
        metrics = UploadMetrics()
        assert metrics.group == ""
        assert metrics.name == ""
        assert metrics.row_count == 0
        assert isinstance(metrics.visited, set)
    
    def test_upload_metrics_from_table_row(self):
        """Test creating UploadMetrics from table row."""
        table_row = ("group1", "table1", "extra", "data")
        metrics = UploadMetrics.from_table_row(table_row)
        assert metrics.group == "group1"
        assert metrics.name == "table1"
        assert isinstance(metrics.visited, set)
    
    def test_upload_metrics_to_dict(self):
        """Test converting UploadMetrics to dictionary."""
        metrics = UploadMetrics(
            group="test_group",
            name="test_table",
            row_count=1000,
            column_count=5,
            memory_size=1024*1024,  # 1MB
            compressed_size=512*1024,  # 512KB
            pull_time=30,
            upload_time=15,
            s3_address="s3://bucket/path",
            creation_time="2023-01-01 12:00:00"
        )
        
        result = metrics.to_dict()
        
        assert result['Consumer Loans Data Product'] == "test_group"
        assert result['PCDS Table Details with DB Name'] == "test_table"
        assert result['Number of Rows'] == 1000
        assert result['Number of Columns'] == 5
        assert 'MB' in result['Memory Size']
        assert 'KB' in result['Compress Size']
        assert result['S3 Address'] == "s3://bucket/path"
        assert result['Last Modified'] == "2023-01-01 12:00:00"
    
    def test_upload_metrics_update_from_time_data(self):
        """Test updating metrics from time tracking data."""
        metrics = UploadMetrics(name="test_table")
        
        time_data = {
            'pcds_test_table_001': {
                'pull_time': 30,
                'upload_time': 15,
                'row_count': 1000,
                'memory_size': 1024,
                'ctime': '2023-01-01 12:00:00',
                's3addr': 's3://bucket/path'
            },
            'pcds_test_table_002': {
                'pull_time': 25,
                'upload_time': 10,
                'row_count': 500,
                'memory_size': 512
            },
            'aws_other_table': {  # Should be ignored
                'pull_time': 100
            }
        }
        
        metrics.update_from_time_data(time_data)
        
        assert metrics.pull_time == 55  # 30 + 25
        assert metrics.upload_time == 25  # 15 + 10
        assert metrics.row_count == 1500  # 1000 + 500
        assert metrics.memory_size == 1536  # 1024 + 512
        assert metrics.creation_time == '2023-01-01 12:00:00'
        assert metrics.s3_address == 's3://bucket/path'
        assert len(metrics.visited) == 2


class TestPCDSConnectionManager:
    """Test cases for PCDSConnectionManager class."""
    
    def test_pcds_connection_manager_init(self):
        """Test PCDSConnectionManager initialization."""
        manager = PCDSConnectionManager()
        assert 'p_uscb_cnsmrlnd_svc' in manager.service_mappings
        assert manager.service_mappings['p_uscb_cnsmrlnd_svc'] == '21P'
    
    @patch('oracledb.connect')
    @patch('do_upload.utils.solve_ldap')
    def test_get_connection_success(self, mock_solve_ldap, mock_connect):
        """Test successful database connection."""
        mock_solve_ldap.return_value = 'mock_tns'
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        manager = PCDSConnectionManager()
        
        with patch.dict(os.environ, {
            'PCDS_USR': 'test_user',
            'PCDS_21P': 'test_pwd'
        }):
            result = manager.get_connection('p_uscb_cnsmrlnd_svc')
        
        assert result == mock_connection
        mock_solve_ldap.assert_called_once()
        mock_connect.assert_called_once_with(
            user='test_user',
            password='test_pwd',
            dsn='mock_tns'
        )
    
    def test_get_connection_invalid_service(self):
        """Test connection with invalid service name."""
        manager = PCDSConnectionManager()
        
        with pytest.raises(ValueError, match="Unknown service name"):
            manager.get_connection('invalid_service')
    
    def test_get_connection_missing_env_var(self):
        """Test connection with missing environment variables."""
        manager = PCDSConnectionManager()
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing environment variable"):
                manager.get_connection('p_uscb_cnsmrlnd_svc')


class TestQueryBuilder:
    """Test cases for QueryBuilder class."""
    
    def test_build_select_query_basic(self):
        """Test basic SELECT query building."""
        query = QueryBuilder.build_select_query('my_table')
        assert query == "SELECT * FROM my_table"
    
    def test_build_select_query_with_columns(self):
        """Test SELECT query with specific columns."""
        query = QueryBuilder.build_select_query('my_table', 'col1, col2, col3')
        assert query == "SELECT col1, col2, col3 FROM my_table"
    
    def test_build_select_query_with_where(self):
        """Test SELECT query with WHERE clause."""
        query = QueryBuilder.build_select_query('my_table', '*', 'id > 100')
        assert query == "SELECT * FROM my_table WHERE id > 100"
    
    def test_build_where_clause_year_partition(self):
        """Test WHERE clause for year partitioning."""
        where_clause = QueryBuilder.build_where_clause(
            date_var='date_col',
            date_type='DATE',
            partition_type='year',
            date_range='2023',
            date_format='%Y-%m-%d'
        )
        assert "TO_CHAR(date_col, 'YYYY') = '2023'" == where_clause
    
    def test_build_where_clause_year_month_partition(self):
        """Test WHERE clause for year-month partitioning."""
        where_clause = QueryBuilder.build_where_clause(
            date_var='date_col',
            date_type='DATE',
            partition_type='year_month',
            date_range='2023-01',
            date_format='%Y-%m-%d'
        )
        assert "TO_CHAR(date_col, 'YYYY-MM') = '2023-01'" == where_clause
    
    def test_build_where_clause_char_date(self):
        """Test WHERE clause with character date type."""
        where_clause = QueryBuilder.build_where_clause(
            date_var='date_str',
            date_type='VARCHAR2',
            partition_type='year',
            date_range='2023',
            date_format='YYYY-MM-DD'
        )
        assert "TO_DATE(date_str, 'YYYY-MM-DD')" in where_clause
        assert "'2023'" in where_clause
    
    def test_build_where_clause_no_partition(self):
        """Test WHERE clause with no partitioning."""
        where_clause = QueryBuilder.build_where_clause(
            date_var='date_col',
            date_type='DATE',
            partition_type='none',
            date_range='',
            date_format='%Y-%m-%d'
        )
        assert where_clause == ""


class TestDataUploader:
    """Test cases for DataUploader class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=ut.PullConfig)
        config.output.s3_config.data = "s3://test-bucket/data"
        return config
    
    @pytest.fixture
    def data_uploader(self, mock_config):
        """Create a DataUploader instance for testing."""
        return DataUploader(mock_config)
    
    def test_data_uploader_init(self, mock_config):
        """Test DataUploader initialization."""
        uploader = DataUploader(mock_config)
        assert uploader.config == mock_config
        assert isinstance(uploader.connection_manager, PCDSConnectionManager)
        assert isinstance(uploader.query_builder, QueryBuilder)
        assert uploader.time_tracking == {}
    
    @patch('do_upload.utils.S3Manager.object_exists')
    @patch('do_upload.utils.S3Manager.save_dataframe')
    @patch('do_upload.utils.S3Manager.get_metadata')
    @patch('pandas.io.sql.read_sql_query')
    def test_upload_table_data_existing_file(self, mock_read_sql, mock_get_metadata,
                                           mock_save_df, mock_exists, data_uploader):
        """Test upload when file already exists and repull is False."""
        mock_exists.return_value = True
        
        pcds_info = Mock()
        pcds_info.info_string = "service.table"
        
        # Should return early without uploading
        data_uploader.upload_table_data(
            pcds_info=pcds_info,
            s3_base_url="s3://bucket",
            s3_basename="test_basename",
            where_conditions=[],
            should_repull=False
        )
        
        mock_read_sql.assert_not_called()
        mock_save_df.assert_not_called()
    
    @patch('do_upload.utils.S3Manager.object_exists')
    @patch('do_upload.utils.S3Manager.save_dataframe')
    @patch('do_upload.utils.S3Manager.get_metadata')
    @patch('pandas.io.sql.read_sql_query')
    @patch('do_upload.utils.Timer')
    def test_upload_table_data_new_upload(self, mock_timer, mock_read_sql,
                                        mock_get_metadata, mock_save_df,
                                        mock_exists, data_uploader):
        """Test uploading new data."""
        mock_exists.return_value = False
        
        # Mock DataFrame
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_read_sql.return_value = mock_df
        
        # Mock timing
        mock_timer_instance = Mock()
        mock_timer_instance.pause.return_value = 30  # 30 seconds
        mock_timer.return_value.__enter__.return_value = mock_timer_instance
        
        # Mock S3 operations
        mock_save_df.return_value = 1024  # 1KB compressed
        mock_get_metadata.return_value = {
            'key': {'LastModified': 'Mon, 01 Jan 2023 12:00:00 GMT'}
        }
        
        # Mock connection
        mock_conn = Mock()
        data_uploader.connection_manager.get_connection = Mock(return_value=mock_conn)
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        
        pcds_info = Mock()
        pcds_info.info_string = "service.table"
        pcds_info.row_count = 3
        
        data_uploader.upload_table_data(
            pcds_info=pcds_info,
            s3_base_url="s3://bucket",
            s3_basename="test_basename",
            where_conditions=["condition1", "condition2"],
            should_repull=True,
            subfolder="partition1"
        )
        
        # Verify query execution
        mock_read_sql.assert_called_once()
        called_query = mock_read_sql.call_args[0][0]
        assert "SELECT * FROM table" in called_query
        assert "WHERE condition1 AND condition2" in called_query
        
        # Verify S3 upload
        mock_save_df.assert_called_once()
        
        # Verify timing data stored
        assert 'test_basename_partition1' in data_uploader.time_tracking
        timing_data = data_uploader.time_tracking['test_basename_partition1']
        assert timing_data['pull_time'] == 30
        assert timing_data['upload_time'] == 30
        assert timing_data['row_count'] == 3
        assert timing_data['column_count'] == 2
    
    @patch('do_upload.utils.S3Manager.save_json')
    def test_save_timing_data_success(self, mock_save_json, data_uploader):
        """Test successful timing data save."""
        data_uploader.time_tracking = {'test': 'data'}
        data_uploader.save_timing_data()
        mock_save_json.assert_called_once()
    
    @patch('do_upload.utils.S3Manager.save_json')
    @patch('do_upload.utils.logger')
    def test_save_timing_data_failure(self, mock_logger, mock_save_json, data_uploader):
        """Test timing data save failure handling."""
        mock_save_json.side_effect = Exception("S3 error")
        data_uploader.save_timing_data()
        mock_logger.error.assert_called_once()


class TestPartitionManager:
    """Test cases for PartitionManager class."""
    
    @pytest.fixture
    def sql_engine(self):
        """Create a mock SQL engine."""
        return Mock(spec=do_upload.utils.SQLEngine)
    
    @pytest.fixture
    def partition_manager(self, sql_engine):
        """Create a PartitionManager instance."""
        return PartitionManager(sql_engine)
    
    def test_partition_manager_init(self, sql_engine):
        """Test PartitionManager initialization."""
        manager = PartitionManager(sql_engine)
        assert manager.sql_engine == sql_engine
        assert 'none' in manager.PARTITION_MODES
        assert manager.PARTITION_MODES['year'] == 'yyyy'
    
    def test_get_partition_ranges_none(self, partition_manager):
        """Test getting partition ranges for no partitioning."""
        result = partition_manager.get_partition_ranges('table', 'date_col', 'none')
        assert result == ['full']
        partition_manager.sql_engine.get_query_range.assert_not_called()
    
    def test_get_partition_ranges_year(self, partition_manager):
        """Test getting partition ranges for year partitioning."""
        partition_manager.sql_engine.get_query_range.return_value = ['2022', '2023']
        
        result = partition_manager.get_partition_ranges('table', 'date_col', 'year')
        
        assert result == ['2022', '2023']
        partition_manager.sql_engine.get_query_range.assert_called_once_with(
            'table', 'date_col', 'year'
        )
    
    def test_get_partition_ranges_error_fallback(self, partition_manager):
        """Test fallback when getting partition ranges fails."""
        partition_manager.sql_engine.get_query_range.side_effect = Exception("DB error")
        
        result = partition_manager.get_partition_ranges('table', 'date_col', 'year')
        
        assert result == ['full']  # Should fallback
    
    def test_get_partition_mode(self, partition_manager):
        """Test getting partition mode strings."""
        assert partition_manager.get_partition_mode('none') == 'full'
        assert partition_manager.get_partition_mode('year') == 'yyyy'
        assert partition_manager.get_partition_mode('year_month') == 'yymm'
        assert partition_manager.get_partition_mode('unknown') == 'full'


class TestUploadProcessor:
    """Test cases for UploadProcessor class."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary configuration file."""
        config_data = {
            "input": {
                "name": "test_upload",
                "step": "upload_step",
                "env_file": Path(".env"),
                "csv_file": Path("input.csv"),
                "json_config": {"meta": Path("meta.json")}
            },
            "output": {
                "folder": Path("output/"),
                "pickle_file": Path("upload.pkl"),
                "csv_config": {"file": Path("upload.csv"), "columns": ["col1", "col2"]},
                "log_config": {"level": "info"},
                "s3_config": {"data": "s3://bucket/data"},
                "json_file": Path("timing.json")
            },
            "pull_data": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, default=str)
            temp_file = Path(f.name)
        
        yield temp_file
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
    
    @patch('confection.Config')
    def test_load_config_success(self, mock_config_class, temp_config_file):
        """Test successful configuration loading."""
        mock_config = Mock()
        mock_config.from_disk.return_value = {
            "input": {"name": "test", "step": "upload", "env_file": Path(".env")},
            "output": {"folder": Path("out"), "pickle_file": Path("test.pkl")},
            "pull_data": {}
        }
        mock_config_class.return_value = mock_config
        
        processor = UploadProcessor(temp_config_file)
        assert processor.config is not None
        assert isinstance(processor.uploader, DataUploader)
        assert isinstance(processor.partition_manager, PartitionManager)
    
    @patch('confection.Config')
    def test_load_config_failure(self, mock_config_class, temp_config_file):
        """Test configuration loading failure."""
        mock_config = Mock()
        mock_config.from_disk.side_effect = Exception("Config error")
        mock_config_class.return_value = mock_config
        
        with pytest.raises(Exception):
            UploadProcessor(temp_config_file)
    
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('do_upload.utils.logger')
    @patch('do_upload.load_dotenv')
    @patch('do_upload.utils.start_run')
    @patch('do_upload.utils.aws_creds_renew')
    @patch('do_upload.utils.S3Manager.load_json')
    def test_setup_environment(self, mock_load_json, mock_aws_creds, mock_start_run,
                              mock_load_dotenv, mock_logger, mock_remove, mock_exists,
                              temp_config_file):
        """Test environment setup."""
        mock_exists.return_value = True
        mock_load_json.return_value = {"existing": "data"}
        
        with patch('confection.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.from_disk.return_value = {
                "input": {"name": "test", "step": "upload", "env_file": Path(".env")},
                "output": {
                    "folder": Path("out"), 
                    "pickle_file": Path("test.pkl"),
                    "csv_config": {"file": Path("output.csv")},
                    "log_config": {"level": "info"},
                    "s3_config": {"data": "s3://bucket"}
                },
                "pull_data": {}
            }
            mock_config_class.return_value = mock_config
            
            processor = UploadProcessor(temp_config_file)
            processor._setup_environment()
        
        mock_remove.assert_called_once()
        mock_load_dotenv.assert_called_once()
        mock_start_run.assert_called_once()
        mock_aws_creds.assert_called_once_with(15 * 60)
    
    @patch('pandas.read_csv')
    @patch('do_upload.utils.FileProcessor.read_meta_json')
    def test_process_single_table_success(self, mock_read_meta, mock_read_csv, temp_config_file):
        """Test successful single table processing."""
        # Mock metadata
        mock_meta_info = Mock()
        mock_meta_info.pcds = Mock()
        mock_meta_info.pcds.row_variable = "date_col"
        mock_meta_info.aws_table = "db.table"
        mock_meta_info.aws.row_variable = "date_col"
        mock_meta_json = {"test_table": mock_meta_info}
        mock_read_meta.return_value = mock_meta_json
        
        with patch('confection.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.from_disk.return_value = {
                "input": {"name": "test", "step": "upload", "env_file": Path(".env")},
                "output": {
                    "folder": Path("out"), 
                    "pickle_file": Path("test.pkl"),
                    "s3_config": {"data": "s3://bucket"}
                },
                "pull_data": {}
            }
            mock_config_class.return_value = mock_config
            
            processor = UploadProcessor(temp_config_file)
            
            # Mock dependencies
            processor.partition_manager.get_partition_ranges = Mock(return_value=['full'])
            processor.uploader.upload_table_data = Mock()
            
            table_row = {"PCDS Table Details with DB Name": "test_table"}
            
            result = processor._process_single_table(table_row, mock_meta_json)
            
            assert isinstance(result, UploadMetrics)
            assert result.name == "test_table"
            processor.uploader.upload_table_data.assert_called()
    
    def test_write_results(self, temp_config_file):
        """Test writing results to CSV."""
        with patch('confection.Config') as mock_config_class:
            mock_config = Mock()
            mock_config.from_disk.return_value = {
                "input": {"name": "test", "step": "upload", "env_file": Path(".env")},
                "output": {
                    "folder": Path("out"), 
                    "pickle_file": Path("test.pkl"),
                    "csv_config": {"file": Path("output.csv"), "columns": ["col1", "col2"]}
                },
                "pull_data": {}
            }
            mock_config_class.return_value = mock_config
            
            processor = UploadProcessor(temp_config_file)
            
            metrics = UploadMetrics(group="test_group", name="test_table")
            
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('csv.DictWriter') as mock_writer_class:
                    mock_writer = Mock()
                    mock_writer_class.return_value = mock_writer
                    
                    processor._write_results(metrics)
                    
                    mock_writer.writeheader.assert_called_once()
                    mock_writer.writerow.assert_called_once()
                    processor.csv_writer_initialized = True
                    
                    # Second call shouldn't write header again
                    processor._write_results(metrics)
                    assert mock_writer.writeheader.call_count == 1
                    assert mock_writer.writerow.call_count == 2


class TestCommandLineInterface:
    """Test cases for command line interface."""
    
    def test_create_argument_parser(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        
        # Test default arguments
        args = parser.parse_args([])
        assert args.config == Path('files/inputs/config_pull.cfg')
        assert args.dry_run is False
        assert args.verbose is False
    
    def test_argument_parser_with_args(self):
        """Test argument parser with custom arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            '--config', 'custom_config.cfg',
            '--dry-run',
            '--verbose'
        ])
        
        assert args.config == Path('custom_config.cfg')
        assert args.dry_run is True
        assert args.verbose is True
    
    @patch('do_upload.UploadProcessor')
    def test_main_success(self, mock_processor_class):
        """Test successful main function execution."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        with patch('sys.argv', ['do_upload.py', '--config', 'test.cfg']):
            with patch('pathlib.Path.exists', return_value=True):
                main()
        
        mock_processor_class.assert_called_once()
        mock_processor.run.assert_called_once()
    
    @patch('do_upload.logger')
    def test_main_file_not_found(self, mock_logger):
        """Test main function with missing config file."""
        with patch('sys.argv', ['do_upload.py', '--config', 'missing.cfg']):
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    main()
    
    @patch('do_upload.UploadProcessor')
    @patch('do_upload.logger')
    def test_main_keyboard_interrupt(self, mock_logger, mock_processor_class):
        """Test main function handling keyboard interrupt."""
        mock_processor = Mock()
        mock_processor.run.side_effect = KeyboardInterrupt()
        mock_processor_class.return_value = mock_processor
        
        with patch('sys.argv', ['do_upload.py']):
            with patch('pathlib.Path.exists', return_value=True):
                main()
        
        mock_logger.info.assert_called_with("Upload process interrupted by user")
    
    @patch('do_upload.UploadProcessor')
    @patch('do_upload.logger')
    def test_main_general_exception(self, mock_logger, mock_processor_class):
        """Test main function handling general exceptions."""
        mock_processor = Mock()
        mock_processor.run.side_effect = Exception("Test error")
        mock_processor_class.return_value = mock_processor
        
        with patch('sys.argv', ['do_upload.py']):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(Exception):
                    main()
        
        mock_logger.error.assert_called()


# Integration tests
class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_upload_metrics_integration(self):
        """Test UploadMetrics integration with time data processing."""
        metrics = UploadMetrics(name="integration_test")
        
        # Simulate time tracking data from multiple partitions
        time_data = {
            'pcds_integration_test_part1': {
                'pull_time': 30,
                'upload_time': 15,
                'row_count': 1000,
                'memory_size': 1024*1024,
                'compressed_size': 512*1024,
                'ctime': '2023-01-01 12:00:00',
                's3addr': 's3://bucket/path1'
            },
            'pcds_integration_test_part2': {
                'pull_time': 45,
                'upload_time': 20,
                'row_count': 1500,
                'memory_size': 1536*1024,
                'compressed_size': 768*1024,
                'ctime': '2023-01-02 12:00:00',
                's3addr': 's3://bucket/path2'
            }
        }
        
        metrics.update_from_time_data(time_data)
        result_dict = metrics.to_dict()
        
        # Verify aggregation
        assert metrics.pull_time == 75  # 30 + 45
        assert metrics.upload_time == 35  # 15 + 20
        assert metrics.row_count == 2500  # 1000 + 1500
        
        # Verify formatting in output
        assert 'MB' in result_dict['Memory Size']
        assert 'KB' in result_dict['Compress Size']
        assert 'hours' in result_dict['Pull Time'] or 'minutes' in result_dict['Pull Time']
    
    @patch('utils.S3Manager.save_dataframe')
    @patch('pandas.io.sql.read_sql_query')
    def test_query_builder_data_uploader_integration(self, mock_read_sql, mock_save_df):
        """Test integration between QueryBuilder and DataUploader."""
        # Setup mocks
        mock_df = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
        mock_read_sql.return_value = mock_df
        mock_save_df.return_value = 1024
        
        config = Mock()
        config.output.s3_config.data = "s3://test-bucket"
        
        uploader = DataUploader(config)
        
        # Mock connection
        mock_conn = Mock()
        uploader.connection_manager.get_connection = Mock(return_value=mock_conn)
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        
        pcds_info = Mock()
        pcds_info.info_string = "service.test_table"
        
        with patch('do_upload.utils.S3Manager.object_exists', return_value=False), \
             patch('do_upload.utils.S3Manager.get_metadata', return_value={}), \
             patch('do_upload.utils.Timer') as mock_timer:
            
            mock_timer_instance = Mock()
            mock_timer_instance.pause.return_value = 30
            mock_timer.return_value.__enter__.return_value = mock_timer_instance
            
            uploader.upload_table_data(
                pcds_info=pcds_info,
                s3_base_url="s3://bucket",
                s3_basename="test_basename",
                where_conditions=["active = 'Y'", "created_date > '2023-01-01'"],
                should_repull=True
            )
        
        # Verify the query was built correctly
        mock_read_sql.assert_called_once()
        actual_query = mock_read_sql.call_args[0][0]
        assert "SELECT * FROM test_table" in actual_query
        assert "WHERE active = 'Y' AND created_date > '2023-01-01'" in actual_query


# Error handling tests
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_upload_metrics_invalid_time_data(self):
        """Test UploadMetrics handling invalid time data."""
        metrics = UploadMetrics(name="test_table")
        
        # Invalid time data structure
        invalid_time_data = {
            'pcds_test_table_001': {
                'invalid_key': 'invalid_value',
                'pull_time': 'not_a_number'  # Should be handled gracefully
            }
        }
        
        # Should not raise exception
        metrics.update_from_time_data(invalid_time_data)
        assert metrics.pull_time == 0  # Should remain unchanged
    
    def test_data_uploader_connection_failure(self):
        """Test DataUploader handling connection failures."""
        config = Mock()
        uploader = DataUploader(config)
        
        # Mock connection failure
        uploader.connection_manager.get_connection = Mock(side_effect=Exception("Connection failed"))
        
        pcds_info = Mock()
        pcds_info.info_string = "service.table"
        
        with pytest.raises(Exception, match="Connection failed"):
            uploader.upload_table_data(
                pcds_info=pcds_info,
                s3_base_url="s3://bucket",
                s3_basename="test",
                where_conditions=[],
                should_repull=True
            )
    
    def test_partition_manager_sql_engine_failure(self):
        """Test PartitionManager handling SQL engine failures."""
        sql_engine = Mock()
        sql_engine.get_query_range.side_effect = Exception("SQL error")
        
        manager = PartitionManager(sql_engine)
        
        # Should fallback gracefully
        result = manager.get_partition_ranges("table", "date_col", "year")
        assert result == ['full']


# Performance tests
class TestPerformance:
    """Performance-related tests."""
    
    def test_upload_metrics_large_time_data(self):
        """Test UploadMetrics performance with large time data."""
        import time
        
        metrics = UploadMetrics(name="perf_test")
        
        # Create large time data structure
        large_time_data = {}
        for i in range(1000):
            large_time_data[f'pcds_perf_test_{i:04d}'] = {
                'pull_time': i,
                'upload_time': i * 0.5,
                'row_count': i * 100,
                'memory_size': i * 1024
            }
        
        start_time = time.time()
        metrics.update_from_time_data(large_time_data)
        processing_time = time.time() - start_time
        
        # Should complete quickly
        assert processing_time < 1.0  # Less than 1 second
        assert metrics.row_count == sum(i * 100 for i in range(1000))
        assert len(metrics.visited) == 1000


# Parametrized tests
@pytest.mark.parametrize("partition_type,expected_mode", [
    ("none", "full"),
    ("year", "yyyy"),
    ("year_month", "yymm"),
    ("unknown", "full"),
])
def test_partition_modes(partition_type, expected_mode):
    """Parametrized test for partition mode mapping."""
    sql_engine = Mock()
    manager = PartitionManager(sql_engine)
    assert manager.get_partition_mode(partition_type) == expected_mode


@pytest.mark.parametrize("date_type,partition_type,expected_function", [
    ("DATE", "year", "TO_CHAR"),
    ("VARCHAR2", "year", "TO_DATE"),
    ("TIMESTAMP", "year_month", "TO_CHAR"),
    ("CHAR", "year_month", "TO_DATE"),
])
def test_where_clause_building(date_type, partition_type, expected_function):
    """Parametrized test for WHERE clause building."""
    where_clause = QueryBuilder.build_where_clause(
        date_var="date_col",
        date_type=date_type,
        partition_type=partition_type,
        date_range="2023",
        date_format="YYYY-MM-DD"
    )
    
    if partition_type != "none":
        assert expected_function in where_clause
        assert "2023" in where_clause


@pytest.mark.parametrize("service_name,expected_suffix", [
    ("p_uscb_cnsmrlnd_svc", "21P"),
    ("p_uscb_rft_svc", "30P"),
    ("pcds_svc", "00P"),
])
def test_service_mappings(service_name, expected_suffix):
    """Parametrized test for service name mappings."""
    manager = PCDSConnectionManager()
    assert manager.service_mappings[service_name] == expected_suffix


if __name__ == '__main__':
    # Run specific test classes or all tests
    pytest.main([
        __file__, 
        '-v',  # verbose output
        '--tb=short',  # shorter traceback format
        '--cov=do_upload',  # coverage report
        '--cov-report=html'  # HTML coverage report
    ])