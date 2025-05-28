"""
Unit tests for the restructured utils module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import datetime as dt
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import pytz

# Import the classes from the restructured module
from utils import (
    LoggingUtils, DateTimeUtils, DataUtils, DatabaseConnectionManager,
    AWSManager, S3Manager, QueryProcessor, FileProcessor, SQLEngine,
    ApplicationManager, URLUtils, UDict, Timer
)


class TestLoggingUtils:
    """Test cases for LoggingUtils class."""
    
    @patch('utils.logger')
    def test_start_run(self, mock_logger):
        """Test start_run logging."""
        LoggingUtils.start_run()
        mock_logger.info.assert_called_once()
        assert '=' * 80 in mock_logger.info.call_args[0][0]
    
    @patch('utils.logger')
    def test_end_run(self, mock_logger):
        """Test end_run logging."""
        LoggingUtils.end_run()
        mock_logger.info.assert_called_once()
        assert '=' * 80 in mock_logger.info.call_args[0][0]
    
    @patch('utils.logger')
    def test_separator(self, mock_logger):
        """Test separator logging."""
        LoggingUtils.separator()
        mock_logger.info.assert_called_once()
        assert '-' * 80 in mock_logger.info.call_args[0][0]


class TestDateTimeUtils:
    """Test cases for DateTimeUtils class."""
    
    def test_get_date_sorted_with_datetime(self):
        """Test date sorting with datetime series."""
        dates = pd.to_datetime(['2023-01-03', '2023-01-01', '2023-01-02'])
        result = DateTimeUtils.get_date_sorted(dates)
        expected = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_get_date_sorted_with_strings(self):
        """Test date sorting with string series."""
        dates = pd.Series(['2023-01-03', '2023-01-01', '2023-01-02'])
        result = DateTimeUtils.get_date_sorted(dates)
        expected = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_get_date_filter(self):
        """Test date filtering."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        filter_dates = ['2023-01-01', '2023-01-03']
        result = DateTimeUtils.get_date_filter(dates, filter_dates)
        expected = pd.Series([True, False, True])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_correct_s3_time_with_string(self):
        """Test S3 time correction with string input."""
        time_str = 'Mon, 01 Jan 2023 12:00:00 GMT'
        result = DateTimeUtils.correct_s3_time(time_str)
        assert isinstance(result, str)
        assert '2023-01-01' in result
    
    def test_correct_s3_time_with_datetime(self):
        """Test S3 time correction with datetime input."""
        dt_obj = dt.datetime(2023, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
        result = DateTimeUtils.correct_s3_time(dt_obj)
        assert isinstance(result, str)
        assert '2023-01-01' in result


class TestDataUtils:
    """Test cases for DataUtils class."""
    
    def test_get_default_value_found(self):
        """Test getting default value when key exists."""
        data = {'key1': 'value1', 'DEFAULT': 'default_value'}
        result = DataUtils.get_default_value(data, 'key1')
        assert result == 'value1'
    
    def test_get_default_value_not_found(self):
        """Test getting default value when key doesn't exist."""
        data = {'key1': 'value1', 'DEFAULT': 'default_value'}
        result = DataUtils.get_default_value(data, 'nonexistent')
        assert result == 'default_value'
    
    def test_format_abbreviation_thousands(self):
        """Test number abbreviation for thousands."""
        result = DataUtils.format_abbreviation(1500)
        assert result == '1.5K'
    
    def test_format_abbreviation_millions(self):
        """Test number abbreviation for millions."""
        result = DataUtils.format_abbreviation(2500000)
        assert result == '2.5M'
    
    def test_format_abbreviation_percentage(self):
        """Test percentage formatting."""
        result = DataUtils.format_abbreviation(0.25, percentage=True)
        assert result == '0.25%'
    
    def test_format_abbreviation_nan(self):
        """Test handling of NaN values."""
        result = DataUtils.format_abbreviation(np.nan)
        assert result == ""
    
    def test_get_memory_size_dataframe(self):
        """Test memory size calculation for DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        result = DataUtils.get_memory_size(df, as_dataset=True)
        assert 'B' in result  # Should contain bytes indicator
    
    def test_get_memory_size_bytes(self):
        """Test memory size calculation for raw bytes."""
        result = DataUtils.get_memory_size(1024, as_dataset=False)
        assert result == '1.000 KB'


class TestUDict:
    """Test cases for UDict class."""
    
    def test_case_insensitive_access(self):
        """Test case-insensitive dictionary access."""
        d = UDict({'Key1': 'value1', 'KEY2': 'value2'})
        assert d['key1'] == 'value1'
        assert d['Key2'] == 'value2'
        assert d['KEY1'] == 'value1'
    
    def test_case_insensitive_contains(self):
        """Test case-insensitive contains check."""
        d = UDict({'Key1': 'value1'})
        assert 'key1' in d
        assert 'KEY1' in d
        assert 'Key1' in d
        assert 'nonexistent' not in d
    
    def test_key_error(self):
        """Test KeyError for non-existent keys."""
        d = UDict({'Key1': 'value1'})
        with pytest.raises(KeyError):
            _ = d['nonexistent']


class TestTimer:
    """Test cases for Timer class."""
    
    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        import time
        with Timer() as timer:
            time.sleep(0.01)  # Sleep for 10ms
            elapsed = timer.elapsed
        assert elapsed >= 0.01
    
    def test_timer_pause(self):
        """Test Timer pause functionality."""
        import time
        timer = Timer()
        timer.__enter__()
        time.sleep(0.01)
        elapsed = timer.pause()
        assert elapsed >= 0.01
    
    def test_format_duration(self):
        """Test duration formatting."""
        result = Timer.format_duration(3661)  # 1 hour, 1 minute, 1 second
        assert '1 hours 1 minutes 1 seconds' == result


class TestDatabaseConnectionManager:
    """Test cases for DatabaseConnectionManager class."""
    
    @patch('utils.oracledb')
    @patch.object(DatabaseConnectionManager, '_solve_ldap')
    def test_get_pcds_connection(self, mock_solve_ldap, mock_oracledb):
        """Test PCDS connection creation."""
        mock_solve_ldap.return_value = 'mock_tns'
        mock_connection = Mock()
        mock_oracledb.connect.return_value = mock_connection
        
        with patch.dict(os.environ, {'PCDS_USR': 'test_user', 'PCDS_21P': 'test_pwd'}):
            db_manager = DatabaseConnectionManager()
            result = db_manager._get_pcds_connection('p_uscb_cnsmrlnd_svc')
            
        mock_oracledb.connect.assert_called_once()
        assert result == mock_connection
    
    @patch('pyathena.connect')
    def test_get_aws_connection(self, mock_connect):
        """Test AWS connection creation."""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        db_manager = DatabaseConnectionManager()
        result = db_manager._get_aws_connection('test_db')
        
        mock_connect.assert_called_once()
        assert result == mock_connection
    
    @patch('ldap3.Server')
    @patch('ldap3.Connection')
    def test_solve_ldap(self, mock_connection_class, mock_server_class):
        """Test LDAP resolution."""
        mock_server = Mock()
        mock_connection = Mock()
        mock_server_class.return_value = mock_server
        mock_connection_class.return_value = mock_connection
        
        # Mock the LDAP entry
        mock_entry = Mock()
        mock_entry.orclNetDescString.value = 'test_tns_string'
        mock_connection.entries = [mock_entry]
        
        ldap_dsn = 'ldap://server:4050/service,cn=OracleContext,dc=domain,dc=com'
        result = DatabaseConnectionManager._solve_ldap(ldap_dsn)
        
        assert result == 'test_tns_string'
        mock_connection.bind.assert_called_once()
        mock_connection.search.assert_called_once()


class TestS3Manager:
    """Test cases for S3Manager class."""
    
    @patch('utils.aws.s3.upload')
    @patch('utils.logger')
    def test_upload_file(self, mock_logger, mock_upload):
        """Test S3 file upload."""
        S3Manager.upload_file('local_path', 's3://bucket/key')
        mock_upload.assert_called_once()
        mock_logger.info.assert_called_once()
    
    @patch('utils.aws.s3.list_objects')
    def test_object_exists_true(self, mock_list):
        """Test S3 object existence check - exists."""
        mock_list.return_value = ['s3://bucket/key']
        result = S3Manager.object_exists('s3://bucket/key')
        assert result is True
    
    @patch('utils.aws.s3.list_objects')
    def test_object_exists_false(self, mock_list):
        """Test S3 object existence check - doesn't exist."""
        mock_list.return_value = []
        result = S3Manager.object_exists('s3://bucket/key')
        assert result is False
    
    @patch('utils.aws.s3.download')
    @patch('pandas.read_parquet')
    def test_load_dataframe(self, mock_read_parquet, mock_download):
        """Test DataFrame loading from S3."""
        mock_df = pd.DataFrame({'A': [1, 2, 3]})
        mock_read_parquet.return_value = mock_df
        
        result = S3Manager.load_dataframe('s3://bucket/key')
        
        mock_download.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('utils.aws.s3.read_json')
    def test_load_json(self, mock_read_json):
        """Test JSON loading from S3."""
        mock_data = pd.DataFrame({'key': ['value']})
        mock_read_json.return_value = mock_data
        
        result = S3Manager.load_json('s3://bucket/key.json')
        
        assert result == {'key': ['value']}


class TestFileProcessor:
    """Test cases for FileProcessor class."""
    
    def test_read_meta_json(self):
        """Test meta JSON file reading."""
        test_data = {
            'table1': {
                'pcds_cols': 'col1; col2',
                'pcds_types': 'VARCHAR2(10); NUMBER',
                'pcds_nrows': 100,
                'pcds_id': 'DATE_COL',
                'aws_cols': 'col1; col2',
                'aws_types': 'varchar(10); decimal',
                'aws_nrows': 100,
                'aws_id': 'date_col',
                'pcds_tbl': 'service.table',
                'aws_tbl': 'db.table',
                'time_excludes': '',
                'last_modified': '2023-01-01'
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(test_data))):
            result = FileProcessor.read_meta_json('test.json')
        
        assert 'table1' in result
        assert hasattr(result['table1'], 'pcds')
        assert hasattr(result['table1'], 'aws')
    
    @patch('pandas.read_excel')
    def test_read_input_excel(self, mock_read_excel):
        """Test Excel input file reading."""
        # Mock Excel data
        mock_df = pd.DataFrame({
            'group': ['group1', 'group2'],
            'pcds_tbl': ['table1', 'table2'],
            'pcds_backup': [None, None],
            'pcds_svc': ['service1', 'service2'],
            'pcds_id': ['col1', 'col2'],
            'aws_id': ['COL1', 'COL2']
        })
        mock_read_excel.return_value = mock_df
        
        # Mock MetaTable
        meta_table = Mock()
        meta_table.file = 'test.xlsx'
        meta_table.sheet = 'Sheet1'
        meta_table.skip_rows = 0
        meta_table.select_cols = {'group': 'group', 'pcds_tbl': 'pcds_tbl'}
        meta_table.select_rows = []
        
        result = FileProcessor.read_input_excel(meta_table)
        
        assert len(result) == 2
        assert 'pcds_tbl' in result.columns
        assert result['pcds_id'].iloc[0] == 'COL1'  # Should be uppercase
    
    @patch('pandas.read_excel')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_process_column_mapping(self, mock_exists, mock_file, mock_read_excel):
        """Test column mapping processing."""
        mock_exists.return_value = False  # Force processing
        
        # Mock Excel data
        mock_sheets = {
            'table1': pd.DataFrame({
                'pcds_column': ['COL1', 'COL2'],
                'aws_column': ['col1', 'col2'],
                'comment': ['', '']
            })
        }
        mock_read_excel.return_value = mock_sheets
        
        # Mock ColumnMap
        column_map = Mock()
        column_map.file = 'mapping.xlsx'
        column_map.to_json = 'mapping.json'
        column_map.overwrite = True
        column_map.excludes = []
        column_map.na_str = ''
        column_map.pcds_col = ['pcds_column']
        column_map.aws_col = ['aws_column']
        
        result = FileProcessor.process_column_mapping(column_map)
        
        assert 'table1' in result
        mock_file.assert_called()


class TestSQLEngine:
    """Test cases for SQLEngine class."""
    
    @patch.object(DatabaseConnectionManager, 'get_connection')
    @patch.object(QueryProcessor, 'execute_query')
    def test_execute_pcds_query(self, mock_execute, mock_get_conn):
        """Test PCDS query execution."""
        mock_conn = Mock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_df = pd.DataFrame({'result': [1, 2, 3]})
        mock_execute.return_value = mock_df
        
        engine = SQLEngine('PCDS')
        result = engine._execute_pcds_query('SELECT * FROM table', service_name='test_service')
        
        pd.testing.assert_frame_equal(result, mock_df)
    
    @patch('utils.aws.athena.start_query_execution')
    @patch('utils.aws.athena.get_query_results')
    def test_execute_aws_query(self, mock_get_results, mock_start_query):
        """Test AWS query execution."""
        mock_start_query.return_value = 'query_id'
        mock_df = pd.DataFrame({'result': [1, 2, 3]})
        mock_get_results.return_value = mock_df
        
        engine = SQLEngine('AWS')
        result = engine._execute_aws_query('SELECT * FROM database.table')
        
        pd.testing.assert_frame_equal(result, mock_df)
        mock_start_query.assert_called_once()
        mock_get_results.assert_called_once_with('query_id', boto3_session=None)
    
    def test_get_query_range_none(self):
        """Test query range with no partitioning."""
        engine = SQLEngine('AWS')
        mock_result = pd.DataFrame({'min': ['2023-01-01'], 'max': ['2023-01-31']})
        
        with patch.object(engine, 'execute', return_value=mock_result):
            result = engine.get_query_range('db.table', 'date_col', 'none')
        
        assert len(result) == 1
        assert result[0].start_date == '2023-01-01'
        assert result[0].end_date == '2023-01-31'


class TestURLUtils:
    """Test cases for URLUtils class."""
    
    def test_get_s3_info_basic(self):
        """Test basic S3 info extraction."""
        s3_url = 's3://bucket/path/to/file.txt'
        result = URLUtils.get_s3_info(s3_url)
        
        assert result.bucket == 'bucket'
        assert result.prefix == 'path/to/file.txt'
        assert result.scheme == 's3'
        assert result.ext == '.txt'
    
    def test_get_s3_info_with_basename(self):
        """Test S3 info with basename addition."""
        s3_url = 's3://bucket/path/'
        result = URLUtils.get_s3_info(s3_url, basename='newfile.csv')
        
        assert 'newfile.csv' in result.url
    
    @patch('utils.TODAY', '20230101')
    def test_get_s3_info_with_today(self):
        """Test S3 info with today's date addition."""
        s3_url = 's3://bucket/path/file.txt'
        result = URLUtils.get_s3_info(s3_url, add_today=True)
        
        assert '20230101' in result.url


class TestQueryProcessor:
    """Test cases for QueryProcessor class."""
    
    @patch('utils.psql.read_sql_query')
    def test_execute_query(self, mock_read_sql):
        """Test query execution with caching."""
        mock_df = pd.DataFrame({'result': [1, 2, 3]})
        mock_read_sql.return_value = mock_df
        mock_conn = Mock()
        
        processor = QueryProcessor()
        result = processor.execute_query('SELECT * FROM table', mock_conn)
        
        pd.testing.assert_frame_equal(result, mock_df)
        mock_read_sql.assert_called_once()
    
    @patch.object(S3Manager, 'object_exists')
    @patch.object(S3Manager, 'load_dataframe')
    @patch.object(QueryProcessor, '_unload_to_s3')
    def test_execute_athena_query_cached(self, mock_unload, mock_load_df, mock_exists):
        """Test Athena query execution with existing cache."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'result': [1, 2, 3]})
        mock_load_df.return_value = mock_df
        
        processor = QueryProcessor()
        result = processor.execute_athena_query('SELECT * FROM table', 's3://bucket/')
        
        pd.testing.assert_frame_equal(result, mock_df)
        mock_unload.assert_not_called()  # Should not unload if cached
    
    @patch.object(S3Manager, 'object_exists')
    @patch.object(S3Manager, 'load_dataframe')
    @patch.object(QueryProcessor, '_unload_to_s3')
    def test_execute_athena_query_no_cache(self, mock_unload, mock_load_df, mock_exists):
        """Test Athena query execution without cache."""
        mock_exists.return_value = False
        mock_df = pd.DataFrame({'result': [1, 2, 3]})
        mock_load_df.return_value = mock_df
        
        processor = QueryProcessor()
        result = processor.execute_athena_query('SELECT * FROM database.table', 's3://bucket/')
        
        pd.testing.assert_frame_equal(result, mock_df)
        mock_unload.assert_called_once()  # Should unload if not cached
    
    @patch('utils.aws.athena.unload')
    def test_unload_to_s3_success(self, mock_unload):
        """Test successful S3 unload operation."""
        mock_response = Mock()
        mock_response.raw_payload = {'Status': {'State': 'SUCCEEDED'}}
        mock_unload.return_value = mock_response
        
        processor = QueryProcessor()
        # Should not raise exception
        processor._unload_to_s3('SELECT * FROM database.table', 's3://bucket/path/')
        
        mock_unload.assert_called_once()
    
    @patch('utils.aws.athena.unload')
    def test_unload_to_s3_failure(self, mock_unload):
        """Test failed S3 unload operation."""
        mock_response = Mock()
        mock_response.raw_payload = {'Status': {'State': 'FAILED'}}
        mock_unload.return_value = mock_response
        
        processor = QueryProcessor()
        with pytest.raises(Exception, match="Athena query failed"):
            processor._unload_to_s3('SELECT * FROM database.table', 's3://bucket/path/')


class TestApplicationManager:
    """Test cases for ApplicationManager class."""
    
    def test_determine_app_type_excel(self):
        """Test Excel application type determination."""
        app_manager = ApplicationManager('test.xlsx')
        assert app_manager.app_type == 'excel'
    
    def test_determine_app_type_powerpoint(self):
        """Test PowerPoint application type determination."""
        app_manager = ApplicationManager('test.pptx')
        assert app_manager.app_type == 'ppt'
    
    def test_determine_app_type_unsupported(self):
        """Test unsupported file type."""
        with pytest.raises(NotImplementedError):
            ApplicationManager('test.txt')
    
    @patch('builtins.open', side_effect=IOError)
    def test_is_open_true(self, mock_open):
        """Test file is open detection."""
        app_manager = ApplicationManager('test.xlsx')
        assert app_manager.is_open() is True
    
    @patch('builtins.open', mock_open())
    def test_is_open_false(self, mock_file):
        """Test file is not open detection."""
        app_manager = ApplicationManager('test.xlsx')
        assert app_manager.is_open() is False


class TestAWSManager:
    """Test cases for AWSManager class."""
    
    @patch('requests.post')
    @patch('requests.get')
    @patch('boto3.Session')
    @patch('utils.logger')
    def test_renew_credentials_success(self, mock_logger, mock_session, mock_get, mock_post):
        """Test successful AWS credentials renewal."""
        # Mock token response
        mock_post.return_value.json.return_value = {'token': 'test_token'}
        
        # Mock credentials response
        mock_get.return_value.json.return_value = {
            'Credentials': {
                'AccessKeyId': 'test_access_key',
                'SecretAccessKey': 'test_secret_key',
                'SessionToken': 'test_session_token'
            }
        }
        
        with patch.dict(os.environ, {'EDP_USR': 'test_user', 'EDP_PWD': 'test_pwd'}):
            AWSManager.renew_credentials()
        
        # Verify environment variables are set
        assert os.environ['AWS_ACCESS_KEY_ID'] == 'test_access_key'
        assert os.environ['AWS_SECRET_ACCESS_KEY'] == 'test_secret_key'
        assert os.environ['AWS_SESSION_TOKEN'] == 'test_session_token'
        
        mock_logger.info.assert_called()
        mock_session.assert_called_once_with(region_name='us-east-1')


# Integration tests
class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_legacy_function_compatibility(self):
        """Test that legacy functions still work."""
        # Import legacy functions
        from utils import get_abbr, get_mem, start_run
        
        # Test legacy function calls
        assert get_abbr(1500) == '1.5K'
        
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = get_mem(df)
        assert 'B' in result
        
        # This should not raise an exception
        with patch('utils.logger'):
            start_run()
    
    @patch.object(S3Manager, 'object_exists')
    @patch.object(DatabaseConnectionManager, 'get_connection')
    def test_sql_engine_with_s3_caching(self, mock_get_conn, mock_s3_exists):
        """Test SQL engine integration with S3 caching."""
        mock_s3_exists.return_value = False
        mock_conn = Mock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        
        engine = SQLEngine('PCDS')
        processor = QueryProcessor()
        
        # This tests the integration between components
        with patch.object(processor, 'execute_query') as mock_execute:
            mock_execute.return_value = pd.DataFrame({'test': [1]})
            result = engine._execute_pcds_query('SELECT * FROM table', service_name='test')
            assert len(result) == 1
    
    def test_file_processor_with_udict(self):
        """Test FileProcessor integration with UDict."""
        test_data = {
            'Table1': {'COL1': 'col1', 'COL2': 'col2'},
            'table2': {'COL3': 'col3', 'COL4': 'col4'}
        }
        
        udict_data = UDict(**test_data)
        
        # Test case-insensitive access
        assert udict_data['table1']['COL1'] == 'col1'
        assert udict_data['TABLE2']['COL3'] == 'col3'


# Performance tests
class TestPerformance:
    """Performance-related tests."""
    
    def test_timer_accuracy(self):
        """Test Timer accuracy for performance measurement."""
        import time
        
        with Timer() as timer:
            time.sleep(0.1)  # Sleep for 100ms
            elapsed = timer.elapsed
        
        # Allow for some variance in timing
        assert 0.09 <= elapsed <= 0.15
    
    def test_udict_performance(self):
        """Test UDict performance with large datasets."""
        # Create a large dictionary
        large_dict = {f'Key{i}': f'Value{i}' for i in range(1000)}
        udict = UDict(**large_dict)
        
        # Time case-insensitive lookups
        with Timer() as timer:
            for i in range(100):
                _ = udict[f'key{i}']
            lookup_time = timer.elapsed
        
        # Should complete quickly
        assert lookup_time < 0.1


# Error handling tests
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_database_connection_error(self):
        """Test database connection error handling."""
        with patch.dict(os.environ, {}, clear=True):
            db_manager = DatabaseConnectionManager()
            with pytest.raises(KeyError):
                db_manager._get_pcds_connection('invalid_service')
    
    def test_s3_file_not_found(self):
        """Test S3 file not found error handling."""
        with patch('utils.aws.s3.download', side_effect=botocore.exceptions.ClientError(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )):
            with pytest.raises(FileNotFoundError):
                S3Manager.load_dataframe('s3://bucket/nonexistent.parquet')
    
    def test_sql_engine_invalid_query(self):
        """Test SQL engine with invalid query."""
        engine = SQLEngine('AWS')
        with pytest.raises(ValueError, match="Cannot extract database from query"):
            engine._execute_aws_query('INVALID SQL')
    
    def test_url_utils_malformed_url(self):
        """Test URLUtils with malformed URLs."""
        # Should handle gracefully without raising exceptions
        result = URLUtils.get_s3_info('not-a-valid-url')
        assert result.scheme == ''
        assert result.bucket == ''


# Mock data fixtures
@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'value': range(10),
        'category': ['A', 'B'] * 5
    })


@pytest.fixture
def sample_meta_json():
    """Fixture providing sample meta JSON data."""
    return {
        'table1': {
            'pcds_cols': 'col1; col2; col3',
            'pcds_types': 'VARCHAR2(10); NUMBER; DATE',
            'pcds_nrows': 1000,
            'pcds_id': 'DATE_COL',
            'aws_cols': 'col1; col2; col3',
            'aws_types': 'varchar(10); decimal; date',
            'aws_nrows': 1000,
            'aws_id': 'date_col',
            'pcds_tbl': 'service.table1',
            'aws_tbl': 'database.table1',
            'time_excludes': '2023-01-01; 2023-01-02',
            'last_modified': '2023-01-01'
        }
    }


# Parametrized tests
@pytest.mark.parametrize("input_value,expected", [
    (1000, '1.0K'),
    (1500000, '1.5M'),
    (2500000000, '2.5B'),
    (0.25, '0.25'),
    (np.nan, ''),
])
def test_format_abbreviation_parametrized(input_value, expected):
    """Parametrized test for format_abbreviation function."""
    result = DataUtils.format_abbreviation(input_value)
    assert result == expected


@pytest.mark.parametrize("platform,expected_class", [
    ('PCDS', SQLEngine),
    ('AWS', SQLEngine),
])
def test_sql_engine_platform_initialization(platform, expected_class):
    """Parametrized test for SQLEngine initialization."""
    engine = SQLEngine(platform)
    assert isinstance(engine, expected_class)
    assert engine.platform == platform


@pytest.mark.parametrize("file_extension,app_type", [
    ('.xlsx', 'excel'),
    ('.pptx', 'ppt'),
])
def test_application_manager_file_types(file_extension, app_type):
    """Parametrized test for ApplicationManager file type detection."""
    app_manager = ApplicationManager(f'test{file_extension}')
    assert app_manager.app_type == app_type


if __name__ == '__main__':
    # Run specific test classes or all tests
    pytest.main([
        __file__, 
        '-v',  # verbose output
        '--tb=short',  # shorter traceback format
        '--cov=utils',  # coverage report
        '--cov-report=html'  # HTML coverage report
    ])