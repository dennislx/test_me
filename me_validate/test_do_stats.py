"""
Comprehensive Tests for Restructured Statistics Module

This module contains unit tests, integration tests, and fixtures for testing
the statistics comparison functionality between PCDS and AWS data sources.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import json

# Import the modules to be tested
import utils_type as ut
from do_stats import (
    DateUtils, ValidationUtils, AWSQueryHandler, StatsDataframe,
    StatisticsComparator, FileManager, StatisticsProcessor,
    ComparisonResult, PartitionStatistics
)


# ===============================
# Test Fixtures
# ===============================

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return ut.StatConfig(
        input=ut.StatInputConfig(
            name="test_analysis",
            step="step_stat",
            env_file=Path("test.env"),
            csv_file=Path("test_meta.csv"),
            json_config={"meta": Path("test_meta.json")},
            select_rows=["table1", "table2"],
            processing_range=ut.ProcessingRange(start=1, end=10)
        ),
        output=ut.StatOutputConfig(
            folder=Path("test_output"),
            pickle_file=Path("test_results.pkl"),
            csv_config=ut.CSVConfig(
                file=Path("test_results.csv"),
                columns=["table", "status", "details"]
            ),
            log_config=ut.LogConfig(level="info"),
            s3_config=ut.S3Config(
                run=Path("s3://test-bucket/run"),
                data=Path("s3://test-bucket/data")
            ),
            reuse_aws_data=True,
            drop_na_values=False
        )
    )


@pytest.fixture
def sample_dataframe_continuous():
    """Create a sample DataFrame with continuous data."""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'amount': np.random.normal(1000, 100, 100),
        'score': np.random.uniform(0, 100, 100),
        'count': np.random.poisson(5, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_categorical():
    """Create a sample DataFrame with categorical data."""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'status': np.random.choice(['active', 'inactive', 'pending'], 100),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_mixed():
    """Create a sample DataFrame with mixed data types."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = {
        'date_col': dates,
        'amount': np.random.normal(1000, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'active': np.random.choice([True, False], 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_meta_info():
    """Create sample metadata information."""
    return ut.MetaJSON(
        pcds_cols="ID,AMOUNT,CATEGORY",
        pcds_types="NUMBER,NUMBER,VARCHAR",
        pcds_nrows=1000,
        pcds_id="ID",
        aws_cols="id,amount,category",
        aws_types="bigint,decimal,varchar",
        aws_nrows=950,
        aws_id="id",
        pcds_table="schema.pcds_table",
        aws_table="schema.aws_table"
    )


@pytest.fixture
def mock_s3_operations():
    """Mock S3 operations."""
    with patch('utils.s3_exist') as mock_exist, \
         patch('utils.s3_load_df') as mock_load, \
         patch('utils.s3_save_df') as mock_save, \
         patch('utils.s3_upload') as mock_upload:
        
        mock_exist.return_value = False
        mock_load.return_value = pd.DataFrame({'test': [1, 2, 3]})
        mock_save.return_value = None
        mock_upload.return_value = None
        
        yield {
            'exist': mock_exist,
            'load': mock_load,
            'save': mock_save,
            'upload': mock_upload
        }


# ===============================
# Unit Tests for Utility Classes
# ===============================

class TestDateUtils:
    """Test cases for DateUtils class."""
    
    def test_extend_excludes(self):
        """Test extending exclude list with date range."""
        exclude_list = ['2024-01-01', '2024-01-05']
        until_date = '2024-01-03'
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 4)
            
            result = DateUtils.extend_excludes(exclude_list, until_date)
            
            assert len(result) >= 2  # At least the original excludes
            assert pd.Timestamp('2024-01-01') in result
            assert pd.Timestamp('2024-01-05') in result
    
    def test_get_aws_until_condition_date_type(self):
        """Test AWS until condition for date types."""
        result = DateUtils.get_aws_until_condition(
            'date_col', 'date', '2024-01-15'
        )
        
        expected = ["DATE_FORMAT(date_col, '%Y-%m-%d') <= '2024-01-15'"]
        assert result == expected
    
    def test_get_aws_until_condition_other_type(self):
        """Test AWS until condition for non-date types."""
        result = DateUtils.get_aws_until_condition(
            'date_col', 'varchar', '2024-01-15'
        )
        
        expected = ["date_col <= '2024-01-15'"]
        assert result == expected
    
    def test_get_aws_where_condition_full_partition(self):
        """Test AWS where condition for full partition."""
        result = DateUtils.get_aws_where_condition(
            'date_col', 'date', 'full'
        )
        
        assert result == ''
    
    def test_get_aws_where_condition_year_partition(self):
        """Test AWS where condition for year partition."""
        result = DateUtils.get_aws_where_condition(
            'date_col', 'date', 'date_col=2024'
        )
        
        expected = "DATE_FORMAT(date_col, '%Y') = '2024'"
        assert result == expected
    
    def test_get_aws_where_condition_month_partition(self):
        """Test AWS where condition for year-month partition."""
        result = DateUtils.get_aws_where_condition(
            'date_col', 'date', 'date_col=2024-03'
        )
        
        expected = "DATE_FORMAT(date_col, '%Y-%m') = '2024-03'"
        assert result == expected


class TestValidationUtils:
    """Test cases for ValidationUtils class."""
    
    def test_get_duplicates(self):
        """Test getting duplicate items from list."""
        items = ['a', 'b', 'c', 'a', 'd', 'b']
        result = ValidationUtils.get_duplicates(items)
        
        assert set(result) == {'a', 'b'}
    
    def test_get_duplicates_no_duplicates(self):
        """Test with no duplicates."""
        items = ['a', 'b', 'c', 'd']
        result = ValidationUtils.get_duplicates(items)
        
        assert result == []
    
    def test_validate_column_mappings_valid(self, sample_meta_info):
        """Test validation with valid column mappings."""
        result = ValidationUtils.validate_column_mappings(sample_meta_info)
        assert result is True
    
    def test_validate_column_mappings_invalid(self):
        """Test validation with invalid column mappings."""
        meta_info = ut.MetaJSON()
        meta_info.aws.column_mapping = {'': ''}
        
        result = ValidationUtils.validate_column_mappings(meta_info)
        assert result is False


# ===============================
# Unit Tests for AWSQueryHandler
# ===============================

class TestAWSQueryHandler:
    """Test cases for AWSQueryHandler class."""
    
    def test_init(self):
        """Test initialization of AWSQueryHandler."""
        handler = AWSQueryHandler(use_cache=True)
        assert handler.use_cache is True
        
        handler = AWSQueryHandler(use_cache=False)
        assert handler.use_cache is False
    
    @patch('utils.s3_exist')
    @patch('utils.s3_load_df')
    def test_query_athena_with_cache(self, mock_load, mock_exist):
        """Test query execution with cache hit."""
        mock_exist.return_value = True
        mock_load.return_value = pd.DataFrame({'test': [1, 2, 3]})
        
        handler = AWSQueryHandler(use_cache=True)
        result = handler.query_athena(
            columns='*',
            database='test_db',
            table='test_table',
            where_conditions=['date > "2024-01-01"'],
            s3_cache='s3://test/cache.parquet'
        )
        
        mock_exist.assert_called_once_with('s3://test/cache.parquet')
        mock_load.assert_called_once_with('s3://test/cache.parquet')
        assert len(result) == 3
    
    @patch('utils.SQLengine')
    @patch('utils.s3_save_df')
    @patch('utils.s3_exist')
    def test_query_athena_without_cache(self, mock_exist, mock_save, mock_engine):
        """Test query execution without cache."""
        mock_exist.return_value = False
        mock_result = pd.DataFrame({'test': [1, 2, 3]})
        mock_engine.return_value.execute.return_value = mock_result
        
        handler = AWSQueryHandler(use_cache=True)
        result = handler.query_athena(
            columns='*',
            database='test_db',
            table='test_table',
            where_conditions=['date > "2024-01-01"'],
            s3_cache='s3://test/cache.parquet'
        )
        
        mock_exist.assert_called_once_with('s3://test/cache.parquet')
        mock_engine.assert_called_once_with('AWS')
        mock_save.assert_called_once()
        assert len(result) == 3


# ===============================
# Unit Tests for StatsDataframe
# ===============================

class TestStatsDataframe:
    """Test cases for StatsDataframe class."""
    
    def test_init_and_categorization(self):
        """Test initialization and column categorization."""
        col2type = {
            'id': 'NUMBER',
            'amount': 'NUMBER',
            'name': 'VARCHAR',
            'date_col': 'DATE',
            'unknown_col': 'UNKNOWN_TYPE'
        }
        
        stats_df = StatsDataframe('PCDS', col2type, 'id')
        
        assert stats_df.platform == 'PCDS'
        assert stats_df.index_col == 'id'
        assert 'amount' in stats_df.continuous_cols
        assert 'name' in stats_df.categorical_cols
        assert 'date_col' in stats_df.date_cols
        assert 'UNKNOWN_TYPE' in stats_df.unknown_types
    
    def test_describe_continuous(self, sample_dataframe_continuous):
        """Test continuous variable description."""
        numeric_df = sample_dataframe_continuous[['amount', 'score', 'count']]
        result = StatsDataframe._describe_continuous(numeric_df)
        
        assert 'N_Total' in result.columns
        assert 'N_Unique' in result.columns
        assert 'N_Missing' in result.columns
        assert 'Min' in result.columns
        assert 'Max' in result.columns
        assert 'Mean' in result.columns
        assert 'Std' in result.columns
        assert 'P_50' in result.columns  # Median
        
        assert len(result) == 3  # Three columns
        assert result.loc['amount', 'N_Total'] == 100
    
    def test_describe_categorical(self, sample_dataframe_categorical):
        """Test categorical variable description."""
        cat_df = sample_dataframe_categorical[['category', 'status']]
        result = StatsDataframe._describe_categorical(cat_df)
        
        assert 'N_Total' in result.columns
        assert 'N_Unique' in result.columns
        assert 'N_Missing' in result.columns
        assert 'Freq' in result.columns
        
        assert len(result) == 2  # Two columns
        assert result.loc['category', 'N_Total'] == 100
        assert isinstance(result.loc['category', 'Freq'], dict)
    
    def test_describe_quantiles(self, sample_dataframe_continuous):
        """Test quantile calculation."""
        numeric_df = sample_dataframe_continuous[['amount']]
        result = StatsDataframe._describe_quantiles(numeric_df)
        
        expected_columns = ['P_01', 'P_10', 'P_25', 'P_50', 'P_75', 'P_90', 'P_99']
        assert all(col in result.columns for col in expected_columns)
        
        # Check that quantiles are in ascending order
        quantiles = result.loc['amount', expected_columns].values
        assert all(quantiles[i] <= quantiles[i+1] for i in range(len(quantiles)-1))
    
    def test_transform_with_mapping(self, sample_dataframe_mixed):
        """Test dataframe transformation with column mapping."""
        col2type = {
            'date_col': 'DATE',
            'amount': 'NUMBER',
            'category': 'VARCHAR',
            'active': 'VARCHAR'
        }
        column_map = {
            'date_col': 'mapped_date',
            'amount': 'mapped_amount',
            'category': 'mapped_category',
            'active': 'mapped_active'
        }
        
        stats_df = StatsDataframe('PCDS', col2type, 'date_col')
        result = stats_df.transform(sample_dataframe_mixed, column_map)
        
        assert 'Type' in result.columns
        assert 'mapped_amount' in result.index
        assert 'mapped_category' in result.index
        assert len(result) == 4  # All columns should be processed


# ===============================
# Unit Tests for StatisticsComparator
# ===============================

class TestStatisticsComparator:
    """Test cases for StatisticsComparator class."""
    
    def test_compare_statistics_no_differences(self):
        """Test comparison with identical statistics."""
        data = {
            'N_Total': [100, 100],
            'N_Unique': [10, 5],
            'Mean': [50.0, 25.0],
            'Type': ['NUMBER', 'VARCHAR']
        }
        df1 = pd.DataFrame(data, index=['col1', 'col2'])
        df2 = df1.copy()
        
        result = StatisticsComparator.compare_statistics(df1, df2)
        
        assert result.has_mismatch is False
        assert len(result.mismatched_columns) == 0
    
    def test_compare_statistics_with_differences(self):
        """Test comparison with statistical differences."""
        data1 = {
            'N_Total': [100, 100],
            'N_Unique': [10, 5],
            'Mean': [50.0, 25.0],
            'Type': ['NUMBER', 'VARCHAR']
        }
        data2 = {
            'N_Total': [95, 100],  # Difference in N_Total
            'N_Unique': [10, 6],   # Difference in N_Unique
            'Mean': [50.0, 25.0],
            'Type': ['NUMBER', 'VARCHAR']
        }
        
        df1 = pd.DataFrame(data1, index=['col1', 'col2'])
        df2 = pd.DataFrame(data2, index=['col1', 'col2'])
        
        result = StatisticsComparator.compare_statistics(df1, df2)
        
        assert result.has_mismatch is True
        assert 'col1' in result.mismatched_columns
        assert 'col2' in result.mismatched_columns
    
    def test_compare_statistics_with_nan_values(self):
        """Test comparison handling NaN values."""
        data1 = {
            'N_Total': [100, np.nan],
            'Mean': [50.0, np.nan],
            'Type': ['NUMBER', 'VARCHAR']
        }
        data2 = {
            'N_Total': [100, np.nan],
            'Mean': [50.0, np.nan],
            'Type': ['NUMBER', 'VARCHAR']
        }
        
        df1 = pd.DataFrame(data1, index=['col1', 'col2'])
        df2 = pd.DataFrame(data2, index=['col1', 'col2'])
        
        result = StatisticsComparator.compare_statistics(df1, df2)
        
        assert result.has_mismatch is False
    
    def test_compare_statistics_frequency_differences(self):
        """Test comparison of frequency columns."""
        data1 = {
            'N_Total': [100, 100],
            'Freq': [{'A': 50, 'B': 50}, {'X': 30, 'Y': 70}],
            'Type': ['VARCHAR', 'VARCHAR']
        }
        data2 = {
            'N_Total': [100, 100],
            'Freq': [{'A': 45, 'B': 55}, {'X': 30, 'Y': 70}],  # Different frequencies
            'Type': ['VARCHAR', 'VARCHAR']
        }
        
        df1 = pd.DataFrame(data1, index=['col1', 'col2'])
        df2 = pd.DataFrame(data2, index=['col1', 'col2'])
        
        result = StatisticsComparator.compare_statistics(df1, df2)
        
        assert result.has_mismatch is True
        assert 'col1' in result.mismatched_columns
        assert 'col2' not in result.mismatched_columns


# ===============================
# Unit Tests for FileManager
# ===============================

class TestFileManager:
    """Test cases for FileManager class."""
    
    def test_init(self, sample_config):
        """Test FileManager initialization."""
        file_manager = FileManager(sample_config)
        assert file_manager.config == sample_config
    
    def test_get_s3_key_full_partition(self, sample_config):
        """Test S3 key generation for full partition."""
        file_manager = FileManager(sample_config)
        s3_url = "s3://test-bucket/data/table_name.parquet"
        
        result = file_manager.get_s3_key(s3_url, 'full')
        expected = "table_name_None"
        
        assert result == expected
    
    def test_get_s3_key_with_partition(self, sample_config):
        """Test S3 key generation with partition."""
        file_manager = FileManager(sample_config)
        s3_url = "s3://test-bucket/data/year=2024/table_name.parquet"
        
        result = file_manager.get_s3_key(s3_url, 'year=2024')
        expected = "year=2024_table_name"
        
        assert result == expected
    
    @patch('utils.s3_upload')
    @patch('os.walk')
    def test_upload_results_to_s3(self, mock_walk, mock_upload, sample_config):
        """Test uploading results to S3."""
        mock_walk.return_value = [
            ('/test/output', [], ['step_stat_results.csv', 'other_file.txt'])
        ]
        
        file_manager = FileManager(sample_config)
        file_manager.upload_results_to_s3()
        
        # Should only upload files that start with the step name
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args[0]
        assert 'step_stat_results.csv' in call_args[0]
    
    @patch('utils.download_froms3')
    @patch('os.path.exists')
    def test_download_meta_files(self, mock_exists, mock_download, sample_config):
        """Test downloading meta files."""
        mock_exists.return_value = False
        
        file_manager = FileManager(sample_config)
        file_manager.download_meta_files()
        
        mock_download.assert_called_once()


# ===============================
# Unit Tests for ComparisonResult
# ===============================

class TestComparisonResult:
    """Test cases for ComparisonResult class."""
    
    def test_init_default(self):
        """Test default initialization."""
        result = ComparisonResult()
        
        assert result.has_mismatch is False
        assert len(result.mismatched_columns) == 0
        assert result.details == ""
    
    def test_init_with_values(self):
        """Test initialization with values."""
        columns = {'col1', 'col2'}
        result = ComparisonResult(
            has_mismatch=True,
            mismatched_columns=columns,
            details="Test details"
        )
        
        assert result.has_mismatch is True
        assert result.mismatched_columns == columns
        assert result.details == "Test details"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ComparisonResult(
            has_mismatch=True,
            mismatched_columns={'col1', 'col2'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['Column Stats UnMatch'] is True
        assert 'col1' in result_dict['Stats UnMatch Details']
        assert 'col2' in result_dict['Stats UnMatch Details']
    
    def test_to_dict_with_details(self):
        """Test conversion to dictionary with details."""
        result = ComparisonResult(
            has_mismatch=True,
            mismatched_columns={'col1', 'col2'},
            details="Custom details"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['Column Stats UnMatch'] is True
        assert result_dict['Stats UnMatch Details'] == "Custom details"


# ===============================
# Integration Tests
# ===============================

class TestStatisticsProcessorIntegration:
    """Integration tests for StatisticsProcessor."""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test CSV
            csv_file = Path(temp_dir) / "test_meta.csv"
            csv_data = pd.DataFrame({
                'PCDS Table Details with DB Name': ['test_table'],
                'Consumer Loans Data Product': ['test_product']
            })
            csv_data.to_csv(csv_file, index=False)
            
            # Create test JSON
            json_file = Path(temp_dir) / "test_meta.json"
            meta_data = {
                'test_table': {
                    'pcds_cols': 'ID,AMOUNT',
                    'pcds_types': 'NUMBER,NUMBER',
                    'pcds_nrows': 100,
                    'pcds_id': 'ID',
                    'aws_cols': 'id,amount',
                    'aws_types': 'bigint,decimal',
                    'aws_nrows': 100,
                    'aws_id': 'id',
                    'pcds_table': 'schema.test_table',
                    'aws_table': 'schema.test_table'
                }
            }
            with open(json_file, 'w') as f:
                json.dump(meta_data, f)
            
            yield {
                'temp_dir': temp_dir,
                'csv_file': csv_file,
                'json_file': json_file
            }
    
    @patch('utils.s3_walk')
    @patch('utils.s3_load_json')
    @patch('pandas.read_parquet')
    def test_process_single_table(self, mock_parquet, mock_s3_json, 
                                 mock_s3_walk, sample_config, temp_files):
        """Test processing a single table."""
        # Setup mocks
        mock_s3_walk.return_value = [('full', 's3://test/pcds_data.parquet')]
        mock_s3_json.return_value = {'test_key': {'ctime': '2024-01-01T00:00:00'}}
        mock_parquet.return_value = pd.DataFrame({
            'ID': [1, 2, 3],
            'AMOUNT': [100, 200, 300]
        })
        
        # Update config with temp files
        sample_config.input.csv_file = temp_files['csv_file']
        sample_config.input.json_config['meta'] = temp_files['json_file']
        sample_config.output.folder = Path(temp_files['temp_dir'])
        sample_config.output.csv_config.file = Path(temp_files['temp_dir']) / "output.csv"
        
        with patch.object(AWSQueryHandler, 'query_athena') as mock_query:
            mock_query.return_value = pd.DataFrame({
                'id': [1, 2, 3],
                'amount': [100, 200, 300]
            })
            
            processor = StatisticsProcessor(sample_config)
            
            # Mock the file manager methods
            with patch.object(processor.file_manager, 'download_meta_files'):
                stats_data = processor.process_all_tables()
            
            assert 'test_table' in stats_data
            assert len(stats_data['test_table']) > 0


# ===============================
# Performance Tests
# ===============================

class TestPerformance:
    """Performance tests for critical functions."""
    
    def test_stats_calculation_performance(self):
        """Test performance of statistics calculation on large dataset."""
        import time
        
        # Create large dataset
        np.random.seed(42)
        large_df = pd.DataFrame({
            'id': range(10000),
            'amount': np.random.normal(1000, 100, 10000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
            'score': np.random.uniform(0, 100, 10000)
        })
        
        col2type = {
            'id': 'NUMBER',
            'amount': 'NUMBER',
            'category': 'VARCHAR',
            'score': 'NUMBER'
        }
        
        stats_df = StatsDataframe('PCDS', col2type, 'id')
        
        start_time = time.time()
        result = stats_df.transform(large_df)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        assert len(result) == 4
        assert 'Type' in result.columns
    
    def test_comparison_performance(self):
        """Test performance of statistics comparison."""
        import time
        
        # Create large statistics DataFrames
        np.random.seed(42)
        n_cols = 1000
        
        data = {
            'N_Total': np.random.randint(100, 10000, n_cols),
            'N_Unique': np.random.randint(1, 100, n_cols),
            'Mean': np.random.normal(50, 10, n_cols),
            'Std': np.random.uniform(1, 20, n_cols),
            'Type': ['NUMBER'] * n_cols
        }
        
        df1 = pd.DataFrame(data, index=[f'col_{i}' for i in range(n_cols)])
        df2 = df1.copy()
        
        # Introduce some differences
        df2.iloc[::10, 0] += 1  # Change every 10th N_Total value
        
        start_time = time.time()
        result = StatisticsComparator.compare_statistics(df1, df2)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 2.0
        assert result.has_mismatch is True
        assert len(result.mismatched_columns) == n_cols // 10  # Every 10th column


# ===============================
# Error Handling Tests
# ===============================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_statsdf_with_empty_dataframe(self):
        """Test StatsDataframe with empty DataFrame."""
        empty_df = pd.DataFrame()
        col2type = {'col1': 'NUMBER'}
        
        stats_df = StatsDataframe('PCDS', col2type, 'col1')
        
        # Should handle empty DataFrame gracefully
        with pytest.raises((KeyError, IndexError)):
            stats_df.transform(empty_df)
    
    def test_statsdf_with_missing_columns(self):
        """Test StatsDataframe with missing expected columns."""
        df = pd.DataFrame({'existing_col': [1, 2, 3]})
        col2type = {'missing_col': 'NUMBER'}
        
        stats_df = StatsDataframe('PCDS', col2type, 'missing_col')
        
        # Should handle missing columns gracefully
        with pytest.raises(KeyError):
            stats_df.transform(df)
    
    def test_comparison_with_mismatched_shapes(self):
        """Test comparison with DataFrames of different shapes."""
        df1 = pd.DataFrame({
            'N_Total': [100, 200],
            'Type': ['NUMBER', 'VARCHAR']
        }, index=['col1', 'col2'])
        
        df2 = pd.DataFrame({
            'N_Total': [100],
            'Type': ['NUMBER']
        }, index=['col1'])
        
        # Should handle shape mismatch
        with pytest.raises((ValueError, IndexError)):
            StatisticsComparator.compare_statistics(df1, df2)
    
    def test_file_manager_with_invalid_paths(self, sample_config):
        """Test FileManager with invalid file paths."""
        sample_config.output.folder = Path("/nonexistent/path")
        
        file_manager = FileManager(sample_config)
        
        # Should handle invalid paths gracefully
        with patch('os.walk', side_effect=OSError("Path not found")):
            with pytest.raises(OSError):
                file_manager.upload_results_to_s3()
    
    def test_aws_query_handler_connection_error(self):
        """Test AWSQueryHandler with connection errors."""
        handler = AWSQueryHandler(use_cache=False)
        
        with patch('utils.SQLengine') as mock_engine:
            mock_engine.return_value.execute.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                handler.query_athena(
                    columns='*',
                    database='test_db',
                    table='test_table',
                    where_conditions=[],
                    s3_cache='s3://test/cache.parquet'
                )


# ===============================
# Edge Case Tests
# ===============================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_statsdf_with_all_null_values(self):
        """Test statistics calculation with all null values."""
        df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        col2type = {'col1': 'VARCHAR', 'col2': 'NUMBER'}
        stats_df = StatsDataframe('PCDS', col2type, 'col1')
        
        result = stats_df.transform(df)
        
        assert result.loc['col1', 'N_Missing'] == 3
        assert result.loc['col2', 'N_Missing'] == 3
        assert result.loc['col1', 'N_Unique'] == 0
    
    def test_statsdf_with_single_row(self):
        """Test statistics calculation with single row."""
        df = pd.DataFrame({
            'id': [1],
            'amount': [100.0],
            'category': ['A']
        })
        
        col2type = {'id': 'NUMBER', 'amount': 'NUMBER', 'category': 'VARCHAR'}
        stats_df = StatsDataframe('PCDS', col2type, 'id')
        
        result = stats_df.transform(df)
        
        assert result.loc['amount', 'N_Total'] == 1
        assert result.loc['amount', 'N_Unique'] == 1
        assert result.loc['amount', 'Min'] == 100.0
        assert result.loc['amount', 'Max'] == 100.0
        assert result.loc['category', 'N_Unique'] == 1
    
    def test_comparison_with_identical_dataframes(self):
        """Test comparison with completely identical DataFrames."""
        data = {
            'N_Total': [100, 200, 300],
            'N_Unique': [10, 20, 30],
            'Mean': [50.0, 75.0, 25.0],
            'Type': ['NUMBER', 'NUMBER', 'VARCHAR']
        }
        
        df1 = pd.DataFrame(data, index=['col1', 'col2', 'col3'])
        df2 = df1.copy()
        
        result = StatisticsComparator.compare_statistics(df1, df2)
        
        assert result.has_mismatch is False
        assert len(result.mismatched_columns) == 0
    
    def test_date_utils_with_invalid_dates(self):
        """Test DateUtils with invalid date formats."""
        exclude_list = ['invalid-date', '2024-13-45']  # Invalid dates
        until_date = '2024-01-01'
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 2)
            
            # Should handle invalid dates gracefully
            result = DateUtils.extend_excludes(exclude_list, until_date)
            
            # Should still return valid dates
            assert len(result) > 0
    
    def test_validation_utils_empty_inputs(self):
        """Test ValidationUtils with empty inputs."""
        # Test with empty list
        assert ValidationUtils.get_duplicates([]) == []
        
        # Test with single item
        assert ValidationUtils.get_duplicates(['single']) == []
        
        # Test with None values
        assert ValidationUtils.get_duplicates([None, None, 'a']) == [None]


# ===============================
# Mock Data Generators
# ===============================

class MockDataGenerator:
    """Utility class for generating mock data for tests."""
    
    @staticmethod
    def create_pcds_dataframe(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
        """Create a mock PCDS DataFrame."""
        np.random.seed(seed)
        
        return pd.DataFrame({
            'LOAN_ID': range(1, n_rows + 1),
            'AMOUNT': np.random.normal(10000, 2000, n_rows),
            'RATE': np.random.uniform(0.03, 0.08, n_rows),
            'TERM': np.random.choice([12, 24, 36, 48, 60], n_rows),
            'STATUS': np.random.choice(['ACTIVE', 'CLOSED', 'DEFAULT'], n_rows),
            'ORIG_DATE': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
            'CUSTOMER_TYPE': np.random.choice(['RETAIL', 'COMMERCIAL'], n_rows)
        })
    
    @staticmethod
    def create_aws_dataframe(n_rows: int = 100, seed: int = 42, 
                           introduce_differences: bool = False) -> pd.DataFrame:
        """Create a mock AWS DataFrame."""
        np.random.seed(seed)
        
        df = pd.DataFrame({
            'loan_id': range(1, n_rows + 1),
            'amount': np.random.normal(10000, 2000, n_rows),
            'rate': np.random.uniform(0.03, 0.08, n_rows),
            'term': np.random.choice([12, 24, 36, 48, 60], n_rows),
            'status': np.random.choice(['active', 'closed', 'default'], n_rows),
            'orig_date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
            'customer_type': np.random.choice(['retail', 'commercial'], n_rows)
        })
        
        if introduce_differences:
            # Introduce some data differences
            df.loc[0:9, 'amount'] *= 1.01  # 1% difference in first 10 rows
            df.loc[10:19, 'status'] = df.loc[10:19, 'status'].str.upper()  # Case differences
            df.loc[20:29, 'rate'] = np.round(df.loc[20:29, 'rate'], 2)  # Precision differences
        
        return df
    
    @staticmethod
    def create_meta_info(table_name: str = "test_table") -> ut.MetaJSON:
        """Create mock metadata information."""
        return ut.MetaJSON(
            pcds_cols="LOAN_ID,AMOUNT,RATE,TERM,STATUS,ORIG_DATE,CUSTOMER_TYPE",
            pcds_types="NUMBER,NUMBER,NUMBER,NUMBER,VARCHAR,DATE,VARCHAR",
            pcds_nrows=1000,
            pcds_id="LOAN_ID",
            aws_cols="loan_id,amount,rate,term,status,orig_date,customer_type",
            aws_types="bigint,decimal,decimal,int,varchar,date,varchar",
            aws_nrows=950,
            aws_id="loan_id",
            pcds_table=f"pcds_schema.{table_name}",
            aws_table=f"aws_schema.{table_name}"
        )


# ===============================
# End-to-End Tests
# ===============================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def complete_test_setup(self):
        """Create complete test setup with all required files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test data
            pcds_df = MockDataGenerator.create_pcds_dataframe(100)
            aws_df = MockDataGenerator.create_aws_dataframe(100, introduce_differences=True)
            meta_info = MockDataGenerator.create_meta_info()
            
            # Create files
            csv_file = temp_path / "meta.csv"
            json_file = temp_path / "meta.json"
            output_csv = temp_path / "results.csv"
            pickle_file = temp_path / "results.pkl"
            
            # Write CSV
            meta_csv_data = pd.DataFrame({
                'PCDS Table Details with DB Name': ['test_table'],
                'Consumer Loans Data Product': ['test_product']
            })
            meta_csv_data.to_csv(csv_file, index=False)
            
            # Write JSON
            meta_json_data = {'test_table': meta_info.__dict__}
            with open(json_file, 'w') as f:
                json.dump(meta_json_data, f, default=str)
            
            # Create configuration
            config = ut.StatConfig(
                input=ut.StatInputConfig(
                    name="test_analysis",
                    step="step_stat",
                    env_file=temp_path / "test.env",
                    csv_file=csv_file,
                    json_config={"meta": json_file},
                    processing_range=ut.ProcessingRange(start=1, end=1)
                ),
                output=ut.StatOutputConfig(
                    folder=temp_path,
                    pickle_file=pickle_file,
                    csv_config=ut.CSVConfig(
                        file=output_csv,
                        columns=["table", "Column Stats UnMatch", "Stats UnMatch Details"]
                    ),
                    s3_config=ut.S3Config(
                        data=temp_path / "s3_data"
                    ),
                    reuse_aws_data=False
                )
            )
            
            yield {
                'config': config,
                'pcds_df': pcds_df,
                'aws_df': aws_df,
                'temp_path': temp_path,
                'files': {
                    'csv': csv_file,
                    'json': json_file,
                    'output_csv': output_csv,
                    'pickle': pickle_file
                }
            }
    
    def test_full_statistics_workflow(self, complete_test_setup):
        """Test the complete statistics comparison workflow."""
        setup = complete_test_setup
        config = setup['config']
        
        # Mock external dependencies
        with patch('utils.s3_walk') as mock_s3_walk, \
             patch('utils.s3_load_json') as mock_s3_json, \
             patch('pandas.read_parquet') as mock_parquet, \
             patch.object(AWSQueryHandler, 'query_athena') as mock_query, \
             patch.object(FileManager, 'download_meta_files'), \
             patch.object(FileManager, 'upload_results_to_s3'):
            
            # Setup mocks
            mock_s3_walk.return_value = [('full', 's3://test/pcds_data.parquet')]
            mock_s3_json.return_value = {'test_key': {'ctime': '2024-01-01T00:00:00'}}
            mock_parquet.return_value = setup['pcds_df']
            mock_query.return_value = setup['aws_df']
            
            # Run the processor
            processor = StatisticsProcessor(config)
            stats_data = processor.process_all_tables()
            
            # Verify results
            assert 'test_table' in stats_data
            assert len(stats_data['test_table']) > 0
            
            # Check that output files would be created
            partition_stats = stats_data['test_table']['full']
            assert isinstance(partition_stats, dict)
            assert 'PCDS_STAT' in partition_stats
            assert 'AWS_STAT' in partition_stats


# ===============================
# Benchmark Tests
# ===============================

class TestBenchmarks:
    """Benchmark tests for performance monitoring."""
    
    def test_large_dataset_processing(self):
        """Benchmark processing of large datasets."""
        import time
        
        # Create large datasets
        large_pcds = MockDataGenerator.create_pcds_dataframe(50000)
        large_aws = MockDataGenerator.create_aws_dataframe(50000)
        
        col2type_pcds = {
            'LOAN_ID': 'NUMBER',
            'AMOUNT': 'NUMBER',
            'RATE': 'NUMBER',
            'TERM': 'NUMBER',
            'STATUS': 'VARCHAR',
            'ORIG_DATE': 'DATE',
            'CUSTOMER_TYPE': 'VARCHAR'
        }
        
        col2type_aws = {
            'loan_id': 'bigint',
            'amount': 'decimal',
            'rate': 'decimal',
            'term': 'int',
            'status': 'varchar',
            'orig_date': 'date',
            'customer_type': 'varchar'
        }
        
        # Benchmark PCDS processing
        start_time = time.time()
        stats_pcds = StatsDataframe('PCDS', col2type_pcds, 'LOAN_ID')
        pcds_result = stats_pcds.transform(large_pcds)
        pcds_time = time.time() - start_time
        
        # Benchmark AWS processing
        start_time = time.time()
        stats_aws = StatsDataframe('AWS', col2type_aws, 'loan_id')
        aws_result = stats_aws.transform(large_aws)
        aws_time = time.time() - start_time
        
        # Benchmark comparison
        start_time = time.time()
        comparison_result = StatisticsComparator.compare_statistics(pcds_result, aws_result)
        comparison_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert pcds_time < 30.0  # Should process 50K rows in under 30 seconds
        assert aws_time < 30.0
        assert comparison_time < 10.0  # Comparison should be faster
        
        # Correctness assertions
        assert len(pcds_result) == 7  # All columns processed
        assert len(aws_result) == 7
        assert isinstance(comparison_result, ComparisonResult)
        
        print(f"Performance Results:")
        print(f"  PCDS Processing: {pcds_time:.2f}s")
        print(f"  AWS Processing: {aws_time:.2f}s")
        print(f"  Comparison: {comparison_time:.2f}s")
        print(f"  Total: {pcds_time + aws_time + comparison_time:.2f}s")


# ===============================
# Test Configuration
# ===============================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--cov=do_stats",  # Coverage for main module
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing"  # Terminal coverage with missing lines
    ])