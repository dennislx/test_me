
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from pathlib import Path
from typing import Dict, List, Any
import xlwings as xw

# Import the modules to be tested
import utils_type as ut
from do_xlsx import (
    ExcelPosition, CellRange, ColumnStatistics, IssueTracker,
    PositionTracker, ExcelFormatter, ExcelRowWriter, ExcelStylist,
    IssueAnalyzer, ExcelReportGenerator, DEFAULT_COLUMNS, COLOR_SCHEME,
    ISSUE_DESCRIPTIONS, MAX_TEXT_LENGTH
)


@pytest.fixture
def sample_column_statistics():
    """Create sample ColumnStatistics object."""
    pcds_stats = pd.DataFrame({
        'N_Total': [100, 200],
        'N_Unique': [10, 50],
        'Mean': [50.0, 105.5],
        'Type': ['NUMBER', 'NUMBER']
    }, index=['col1', 'col2'])
    
    aws_stats = pd.DataFrame({
        'N_Total': [95, 200],
        'N_Unique': [10, 48],
        'Mean': [50.0, 105.0],
        'Type': ['NUMBER', 'NUMBER']
    }, index=['mapped_col1', 'mapped_col2'])
    
    return ColumnStatistics(
        pcds_stats=pcds_stats,
        pcds_name='test_pcds_table',
        pcds_mismatches=['col1'],
        aws_stats=aws_stats,
        aws_name='test_aws_table',
        aws_mismatches=['mapped_col1'],
        pcds_to_aws_mapping={'col1': 'mapped_col1', 'col2': 'mapped_col2'}
    )


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return ut.StatConfig(
        input=ut.StatInputConfig(
            name="test_excel",
            step="step_xlsx",
            env_file=Path("test.env"),
            json_config={"meta": Path("test_meta.json")}
        ),
        output=ut.StatOutputConfig(
            folder=Path("test_output"),
            pickle_file=Path("test_stats.pkl"),
            s3_config=ut.S3Config(run=Path("s3://test-bucket/run"))
        )
    )


@pytest.fixture
def sample_meta_info():
    """Create sample meta information."""
    return ut.MetaJSON(
        pcds_cols="COL1,COL2,COL3",
        pcds_types="NUMBER,NUMBER,VARCHAR",
        pcds_nrows=1000,
        pcds_id="COL1",
        aws_cols="col1,col2,col3",
        aws_types="bigint,decimal,varchar",
        aws_nrows=950,
        aws_id="col1",
        pcds_table="schema.pcds_table",
        aws_table="schema.aws_table"
    )


@pytest.fixture
def mock_xlwings_workbook():
    """Mock xlwings workbook."""
    mock_workbook = Mock(spec=xw.Book)
    mock_sheet = Mock(spec=xw.Sheet)
    mock_range = Mock(spec=xw.Range)
    
    # Setup sheet mocking
    mock_workbook.sheets = [mock_sheet]
    mock_workbook.sheets.__getitem__ = Mock(return_value=mock_sheet)
    mock_workbook.sheets.add = Mock(return_value=mock_sheet)
    
    # Setup range mocking
    mock_sheet.__getitem__ = Mock(return_value=mock_range)
    mock_sheet.used_range = Mock()
    mock_sheet.used_range.last_cell = Mock()
    mock_sheet.used_range.last_cell.row = 10
    mock_sheet.used_range.api = Mock()
    mock_sheet.used_range.api.Columns = Mock()
    mock_sheet.used_range.api.Columns.AutoFit = Mock()
    
    # Setup range properties
    mock_range.value = None
    mock_range.font = Mock()
    mock_range.font.color = None
    mock_range.number_format = None
    mock_range.merge = Mock()
    mock_range.characters = Mock()
    
    return mock_workbook


# ===============================
# Unit Tests for Data Classes
# ===============================

class TestExcelPosition:
    """Test cases for ExcelPosition class."""
    
    def test_valid_position(self):
        """Test creating valid position."""
        pos = ExcelPosition(5, 3)
        assert pos.row == 5
        assert pos.col == 3
    
    def test_zero_position(self):
        """Test position at origin."""
        pos = ExcelPosition(0, 0)
        assert pos.row == 0
        assert pos.col == 0
    
    def test_invalid_negative_row(self):
        """Test invalid negative row."""
        with pytest.raises(ValueError, match="Row and column must be non-negative"):
            ExcelPosition(-1, 0)
    
    def test_invalid_negative_col(self):
        """Test invalid negative column."""
        with pytest.raises(ValueError, match="Row and column must be non-negative"):
            ExcelPosition(0, -1)


class TestCellRange:
    """Test cases for CellRange class."""
    
    def test_valid_range(self):
        """Test creating valid range."""
        cell_range = CellRange(1, 2, 5, 8)
        assert cell_range.start_row == 1
        assert cell_range.start_col == 2
        assert cell_range.end_row == 5
        assert cell_range.end_col == 8
        assert cell_range.width == 7  # 8 - 2 + 1
        assert cell_range.height == 5  # 5 - 1 + 1
    
    def test_single_cell_range(self):
        """Test single cell range."""
        cell_range = CellRange(3, 3, 3, 3)
        assert cell_range.width == 1
        assert cell_range.height == 1
    
    def test_invalid_row_order(self):
        """Test invalid row order."""
        with pytest.raises(ValueError, match="Start position must be before end position"):
            CellRange(5, 2, 3, 8)
    
    def test_invalid_col_order(self):
        """Test invalid column order."""
        with pytest.raises(ValueError, match="Start position must be before end position"):
            CellRange(1, 8, 5, 2)


class TestColumnStatistics:
    """Test cases for ColumnStatistics class."""
    
    def test_initialization(self, sample_stats_dataframe):
        """Test ColumnStatistics initialization."""
        col_stats = ColumnStatistics(
            pcds_stats=sample_stats_dataframe.copy(),
            pcds_name="test_pcds",
            pcds_mismatches=["col1"],
            aws_stats=sample_stats_dataframe.copy(),
            aws_name="test_aws",
            aws_mismatches=["col1"],
            pcds_to_aws_mapping={"col1": "col1", "col2": "col2", "col3": "col3"}
        )
        
        assert col_stats.pcds_name == "test_pcds"
        assert col_stats.aws_name == "test_aws"
        assert col_stats.pcds_mismatches == ["col1"]
        assert col_stats.aws_mismatches == ["col1"]
    
    def test_transpose_dataframe(self, sample_stats_dataframe):
        """Test dataframe transposition."""
        result = ColumnStatistics._transpose_dataframe(
            sample_stats_dataframe,
            ["col1"],
            ["col1", "col2", "col3"]
        )
        
        # Should be transposed
        assert result.shape[0] == sample_stats_dataframe.shape[1]
        assert result.shape[1] == sample_stats_dataframe.shape[0]
        
        # Type should be first row
        assert result.index[0] == 'Type'
    
    def test_transpose_empty_dataframe(self):
        """Test transposition with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = ColumnStatistics._transpose_dataframe(empty_df, [], [])
        assert result.empty


class TestIssueTracker:
    """Test cases for IssueTracker class."""
    
    def test_initialization(self):
        """Test IssueTracker initialization."""
        tracker = IssueTracker()
        assert len(tracker.issues) == 4
        assert all(isinstance(issue_set, set) for issue_set in tracker.issues)
    
    def test_add_issue(self):
        """Test adding issues."""
        tracker = IssueTracker()
        tracker.add_issue(0, "col1")
        tracker.add_issue(1, "col2")
        tracker.add_issue(0, "col3")  # Same type, different column
        
        assert "col1" in tracker.issues[0]
        assert "col3" in tracker.issues[0]
        assert "col2" in tracker.issues[1]
        assert len(tracker.issues[0]) == 2
        assert len(tracker.issues[1]) == 1
    
    def test_add_invalid_issue_type(self):
        """Test adding invalid issue type."""
        tracker = IssueTracker()
        tracker.add_issue(5, "col1")  # Invalid type
        tracker.add_issue(-1, "col2")  # Invalid type
        
        # Should not add anything
        assert all(len(issue_set) == 0 for issue_set in tracker.issues)
    
    def test_get_issue_counts(self):
        """Test getting issue counts."""
        tracker = IssueTracker()
        tracker.add_issue(0, "col1")
        tracker.add_issue(0, "col2")
        tracker.add_issue(1, "col3")
        
        counts = tracker.get_issue_counts()
        assert counts == [2, 1, 0, 0]
    
    def test_get_issues_for_column(self):
        """Test getting issues for specific column."""
        tracker = IssueTracker()
        tracker.add_issue(0, "col1")
        tracker.add_issue(2, "col1")
        tracker.add_issue(1, "col2")
        
        col1_issues = tracker.get_issues_for_column("col1")
        col2_issues = tracker.get_issues_for_column("col2")
        col3_issues = tracker.get_issues_for_column("col3")
        
        assert set(col1_issues) == {0, 2}
        assert col2_issues == [1]
        assert col3_issues == []
    
    def test_clear(self):
        """Test clearing all issues."""
        tracker = IssueTracker()
        tracker.add_issue(0, "col1")
        tracker.add_issue(1, "col2")
        
        tracker.clear()
        
        assert all(len(issue_set) == 0 for issue_set in tracker.issues)


# ===============================
# Unit Tests for Utility Classes
# ===============================

class TestPositionTracker:
    """Test cases for PositionTracker class."""
    
    def test_initialization(self):
        """Test PositionTracker initialization."""
        tracker = PositionTracker()
        assert tracker.worksheet_count == 0
        assert tracker.current_row == 0
        assert tracker.region_start_col == 0
        assert tracker.region_start_row == 0
    
    def test_increment_operations(self):
        """Test increment operations."""
        tracker = PositionTracker()
        
        tracker.increment_worksheets(2)
        assert tracker.worksheet_count == 2
        
        tracker.increment_row(5)
        assert tracker.current_row == 5
        
        tracker.increment_row()  # Default increment of 1
        assert tracker.current_row == 6
    
    def test_set_operations(self):
        """Test set operations."""
        tracker = PositionTracker()
        
        tracker.set_row(10)
        assert tracker.current_row == 10
        
        tracker.set_region_start(5, 3)
        assert tracker.region_start_row == 5
        assert tracker.region_start_col == 3
    
    def test_get_position(self):
        """Test getting current position."""
        tracker = PositionTracker()
        tracker.set_row(7)
        
        pos = tracker.get_position()
        assert isinstance(pos, ExcelPosition)
        assert pos.row == 7
        assert pos.col == 0
    
    def test_get_region_start(self):
        """Test getting region start position."""
        tracker = PositionTracker()
        tracker.set_region_start(3, 5)
        
        pos = tracker.get_region_start()
        assert isinstance(pos, ExcelPosition)
        assert pos.row == 3
        assert pos.col == 5


class TestExcelFormatter:
    """Test cases for ExcelFormatter class."""
    
    def test_get_rgb_color_valid(self):
        """Test getting RGB for valid color."""
        rgb = ExcelFormatter.get_rgb_color('red')
        assert rgb == (255, 0, 0)
        
        rgb = ExcelFormatter.get_rgb_color('blue')
        assert rgb == (0, 0, 255)
    
    def test_get_rgb_color_invalid(self):
        """Test getting RGB for invalid color."""
        rgb = ExcelFormatter.get_rgb_color('invalid_color')
        assert rgb == (0, 0, 0)  # Should default to black
    
    def test_format_cell_values(self):
        """Test formatting cell values."""
        df = pd.DataFrame({
            'short': ['abc', 'def'],
            'long': ['a' * 100, 'b' * 150],
            'numeric': [123, 456]
        })
        
        formatted = ExcelFormatter.format_cell_values(df, max_length=50)
        
        assert formatted.loc[0, 'short'] == 'abc'
        assert len(formatted.loc[0, 'long']) == 50
        assert formatted.loc[0, 'numeric'] == 123
    
    @patch('xlwings.Range')
    def test_apply_superscript(self, mock_range):
        """Test applying superscript formatting."""
        mock_cell = Mock()
        mock_cell.value = "Test*"
        mock_cell.characters = Mock()
        mock_range.__iter__ = Mock(return_value=iter([mock_cell]))
        
        ExcelFormatter.apply_superscript(mock_range)
        
        mock_cell.characters.__getitem__.assert_called_with(slice(4, 5))


# ===============================
# Unit Tests for Excel Writers
# ===============================

class TestExcelRowWriter:
    """Test cases for ExcelRowWriter class."""
    
    def test_initialization(self, mock_xlwings_workbook):
        """Test ExcelRowWriter initialization."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        tracker = PositionTracker()
        
        writer = ExcelRowWriter(mock_sheet, tracker)
        
        assert writer.worksheet == mock_sheet
        assert writer.tracker == tracker
        assert isinstance(writer.formatter, ExcelFormatter)
    
    def test_write_info_row_simple(self, mock_xlwings_workbook):
        """Test writing simple info row."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        tracker = PositionTracker()
        writer = ExcelRowWriter(mock_sheet, tracker)
        
        info_dict = {0: "Test", 1: "Value"}
        end_col = writer.write_info_row(info_dict)
        
        assert end_col == 2
        # Verify calls were made to set cell values
        assert mock_sheet.__getitem__.call_count >= 2
    
    def test_write_info_row_with_merge(self, mock_xlwings_workbook):
        """Test writing info row with merged cells."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        tracker = PositionTracker()
        writer = ExcelRowWriter(mock_sheet, tracker)
        
        info_dict = {"1:3": "Merged Value", 4: "Single"}
        end_col = writer.write_info_row(info_dict)
        
        assert end_col == 5
    
    def test_write_dataframe(self, mock_xlwings_workbook, sample_stats_dataframe):
        """Test writing dataframe."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        tracker = PositionTracker()
        writer = ExcelRowWriter(mock_sheet, tracker)
        
        rows_written = writer.write_dataframe(sample_stats_dataframe)
        
        assert rows_written == len(sample_stats_dataframe)
    
    def test_write_empty_dataframe(self, mock_xlwings_workbook):
        """Test writing empty dataframe."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        tracker = PositionTracker()
        writer = ExcelRowWriter(mock_sheet, tracker)
        
        empty_df = pd.DataFrame()
        rows_written = writer.write_dataframe(empty_df)
        
        assert rows_written == 0


class TestExcelStylist:
    """Test cases for ExcelStylist class."""
    
    def test_initialization(self, mock_xlwings_workbook):
        """Test ExcelStylist initialization."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        stylist = ExcelStylist(mock_sheet)
        
        assert stylist.worksheet == mock_sheet
        assert isinstance(stylist.formatter, ExcelFormatter)
    
    def test_values_equal_numeric(self):
        """Test numeric value equality."""
        assert ExcelStylist._values_equal(1.0, 1.0)
        assert ExcelStylist._values_equal(1.0, 1.0000000001)  # Within tolerance
        assert not ExcelStylist._values_equal(1.0, 2.0)
    
    def test_values_equal_none(self):
        """Test None value equality."""
        assert ExcelStylist._values_equal(None, None)
        assert not ExcelStylist._values_equal(None, 1.0)
        assert not ExcelStylist._values_equal(1.0, None)
    
    def test_values_equal_string(self):
        """Test string value equality."""
        assert ExcelStylist._values_equal("abc", "abc")
        assert not ExcelStylist._values_equal("abc", "def")
    
    def test_style_comparison_region_invalid_dimensions(self, mock_xlwings_workbook):
        """Test styling with invalid dimensions."""
        mock_sheet = mock_xlwings_workbook.sheets[0]
        stylist = ExcelStylist(mock_sheet)
        
        pcds_range = CellRange(1, 1, 3, 3)  # 3x3
        aws_range = CellRange(5, 1, 6, 3)   # 2x3
        
        with pytest.raises(ValueError, match="must have the same dimensions"):
            stylist.style_comparison_region(pcds_range, aws_range)


# ===============================
# Unit Tests for Issue Analysis
# ===============================

class TestIssueAnalyzer:
    """Test cases for IssueAnalyzer class."""
    
    def test_analyze_empty_dataframes(self):
        """Test analysis with empty dataframes."""
        empty_df = pd.DataFrame()
        tracker = IssueAnalyzer.analyze_column_issues(empty_df, empty_df)
        
        assert isinstance(tracker, IssueTracker)
        assert all(len(issue_set) == 0 for issue_set in tracker.issues)
    
    def test_count_differences(self):
        """Test counting differences between series."""
        series1 = pd.Series([1, 2, 3, 4])
        series2 = pd.Series([1, 2, 4, 4])  # One difference
        
        diff_count = IssueAnalyzer._count_differences(series1, series2)
        assert diff_count == 1
    
    def test_count_differences_no_diff(self):
        """Test counting with no differences."""
        series1 = pd.Series([1, 2, 3])
        series2 = pd.Series([1, 2, 3])
        
        diff_count = IssueAnalyzer._count_differences(series1, series2)
        assert diff_count == 0
    
    def test_categorize_issue_type_1(self):
        """Test categorizing issue type 1 (zero as missing)."""
        pcds_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': {'A': 10, 'B': 15}
        })
        aws_series = pd.Series({
            'N_Unique': 4,  # One less unique
            'N_Missing': 5,  # Has missing values
            'Freq': {'A': 10, 'B': 15}
        })
        
        issue_type = IssueAnalyzer._categorize_issue(pcds_series, aws_series, 2)
        assert issue_type == 0  # Issue type 1 (zero-indexed)
    
    def test_categorize_issue_type_2(self):
        """Test categorizing issue type 2 (precision truncation)."""
        pcds_series = pd.Series({
            'N_Unique': 10,
            'N_Missing': 0,
            'Freq': None
        })
        aws_series = pd.Series({
            'N_Unique': 8,  # Less unique values
            'N_Missing': 0,
            'Freq': None
        })
        
        issue_type = IssueAnalyzer._categorize_issue(pcds_series, aws_series, 3)
        assert issue_type == 1  # Issue type 2 (zero-indexed)
    
    def test_categorize_issue_type_3(self):
        """Test categorizing issue type 3 (categorical differences)."""
        pcds_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': {'A': 10, 'B': 15}
        })
        aws_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': {'A': 12, 'B': 13}  # Different frequency
        })
        
        issue_type = IssueAnalyzer._categorize_issue(pcds_series, aws_series, 2)
        assert issue_type == 2  # Issue type 3 (zero-indexed)
    
    def test_categorize_issue_type_4(self):
        """Test categorizing issue type 4 (other differences)."""
        pcds_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': None
        })
        aws_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': None
        })
        
        issue_type = IssueAnalyzer._categorize_issue(pcds_series, aws_series, 10)  # Many differences
        assert issue_type == 3  # Issue type 4 (zero-indexed)
    
    def test_categorize_no_issue(self):
        """Test when no issue is found."""
        pcds_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': None
        })
        aws_series = pd.Series({
            'N_Unique': 5,
            'N_Missing': 0,
            'Freq': None
        })
        
        issue_type = IssueAnalyzer._categorize_issue(pcds_series, aws_series, 0)  # No differences
        assert issue_type is None


# ===============================
# Integration Tests
# ===============================

class TestExcelReportGeneratorIntegration:
    """Integration tests for ExcelReportGenerator."""
    
    @patch('xlwings.Book')
    def test_initialization(self, mock_book_class, sample_config):
        """Test ExcelReportGenerator initialization."""
        generator = ExcelReportGenerator(sample_config)
        
        assert generator.config == sample_config
        assert generator.summary_columns == DEFAULT_COLUMNS
        assert generator.workbook is None
        assert isinstance(generator.position_tracker, PositionTracker)
    
    @patch('xlwings.Book')
    def test_initialization_custom_columns(self, mock_book_class, sample_config):
        """Test initialization with custom columns."""
        custom_columns = ['Col1', 'Col2', 'Col3']
        generator = ExcelReportGenerator(sample_config, custom_columns)
        
        assert generator.summary_columns == custom_columns
    
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('xlwings.Book')
    def test_generate_report_setup(self, mock_book_class, mock_remove, 
                                  mock_exists, sample_config, mock_xlwings_workbook):
        """Test report generation setup."""
        mock_exists.return_value = True
        mock_book_class.return_value = mock_xlwings_workbook
        
        generator = ExcelReportGenerator(sample_config)
        
        # Mock the required methods
        with patch.object(generator, '_create_summary_sheet'), \
             patch.object(generator, '_add_issue_descriptions'), \
             patch.object(generator, '_finalize_workbook'):
            
            result_path = generator.generate_report({}, {})
            
            # Verify file operations
            mock_remove.assert_called_once()
            mock_book_class.assert_called_once()
            
            expected_path = os.path.join(sample_config.output.folder, f'{sample_config.input.step}.xlsx')
            assert result_path == expected_path


# ===============================
# Error Handling Tests
# ===============================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_excel_position_invalid_values(self):
        """Test ExcelPosition with invalid values."""
        with pytest.raises(ValueError):
            ExcelPosition(-1, 5)
        
        with pytest.raises(ValueError):
            ExcelPosition(5, -1)
    
    def test_cell_range_invalid_order(self):
        """Test CellRange with invalid order."""
        with pytest.raises(ValueError):
            CellRange(5, 2, 3, 8)  # start_row > end_row
        
        with pytest.raises(ValueError):
            CellRange(1, 8, 5, 2)  # start_col > end_col
    
    def test_issue_tracker_invalid_issue_type(self):
        """Test IssueTracker with invalid issue types."""
        tracker = IssueTracker()
        
        # These should not raise errors but should not add issues
        tracker.add_issue(-1, "col1")
        tracker.add_issue(10, "col2")
        
        assert all(len(issue_set) == 0 for issue_set in tracker.issues)
    
    def test_column_statistics_empty_dataframes(self):
        """Test ColumnStatistics with empty dataframes."""
        empty_df = pd.DataFrame()
        
        col_stats = ColumnStatistics(
            pcds_stats=empty_df,
            pcds_name="test",
            pcds_mismatches=[],
            aws_stats=empty_df,
            aws_name="test",
            aws_mismatches=[],
            pcds_to_aws_mapping={}
        )
        
        assert col_stats.pcds_stats.empty
        assert col_stats.aws_stats.empty


# ===============================
# Performance Tests
# ===============================

class TestPerformance:
    """Performance tests for critical functions."""
    
    def test_large_dataframe_transpose_performance(self):
        """Test performance of dataframe transposition on large data."""
        import time
        
        # Create large dataframe
        n_cols = 1000
        n_rows = 50
        
        large_df = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)]
        )
        large_df['Type'] = 'NUMBER'
        
        mismatches = [f'col_{i}' for i in range(0, n_cols, 10)]  # Every 10th column
        columns = list(large_df.columns[:-1])  # Exclude 'Type'
        
        start_time = time.time()
        result = ColumnStatistics._transpose_dataframe(large_df, mismatches, columns)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 2.0
        assert not result.empty
        assert result.shape[0] == large_df.shape[1]  # Transposed
    
    def test_issue_analysis_performance(self):
        """Test performance of issue analysis on large datasets."""
        import time
        
        # Create large statistics dataframes
        n_cols = 500
        
        pcds_data = {
            'N_Total': np.random.randint(100, 10000, n_cols),
            'N_Unique': np.random.randint(1, 100, n_cols),
            'Mean': np.random.normal(50, 10, n_cols),
            'Type': ['NUMBER'] * n_cols
        }
        
        aws_data = pcds_data.copy()
        # Introduce some differences
        aws_data['N_Total'] = aws_data['N_Total'] + np.random.randint(-10, 10, n_cols)
        
        pcds_df = pd.DataFrame(pcds_data, index=[f'col_{i}' for i in range(n_cols)])
        aws_df = pd.DataFrame(aws_data, index=[f'col_{i}' for i in range(n_cols)])
        
        start_time = time.time()
        tracker = IssueAnalyzer.analyze_column_issues(pcds_df, aws_df)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0
        assert isinstance(tracker, IssueTracker)


# ===============================
# Mock Data Generators
# ===============================

class MockDataGenerator:
    """Generate mock data for testing Excel functionality."""
    
    @staticmethod
    def create_mock_stats_data() -> Dict[str, Dict]:
        """Create mock statistics data structure."""
        return {
            'test_table_1': {
                'full': {
                    'PCDS_STAT': pd.DataFrame({
                        'N_Total': [100, 200],
                        'N_Unique': [10, 50],
                        'Mean': [50.0, 105.5],
                        'Type': ['NUMBER', 'NUMBER']
                    }, index=['col1', 'col2']),
                    'PCDS_MISMATCH': ['col1'],
                    'PCDS_NAME': 'pcds_test_table',
                    'AWS_STAT': pd.DataFrame({
                        'N_Total': [95, 200],
                        'N_Unique': [10, 48],
                        'Mean': [50.0, 105.0],
                        'Type': ['NUMBER', 'NUMBER']
                    }, index=['col1', 'col2']),
                    'AWS_MISMATCH': ['col1'],
                    'AWS_NAME': 'aws_test_table'
                }
            },
            'test_table_2': {
                'year=2024': {
                    'PCDS_STAT': pd.DataFrame({
                        'N_Total': [150, 300, 250],
                        'N_Unique': [15, 75, 50],
                        'Mean': [75.0, 150.0, 125.0],
                        'Type': ['NUMBER', 'NUMBER', 'VARCHAR']
                    }, index=['col1', 'col2', 'col3']),
                    'PCDS_MISMATCH': ['col1', 'col3'],
                    'PCDS_NAME': 'pcds_test_table_2',
                    'AWS_STAT': pd.DataFrame({
                        'N_Total': [148, 300, 245],
                        'N_Unique': [15, 75, 48],
                        'Mean': [75.1, 150.0, 125.0],
                        'Type': ['NUMBER', 'NUMBER', 'VARCHAR']
                    }, index=['col1', 'col2', 'col3']),
                    'AWS_MISMATCH': ['col1', 'col3'],
                    'AWS_NAME': 'aws_test_table_2'
                }
            }
        }
    
    @staticmethod
    def create_mock_meta_data() -> Dict[str, Any]:
        """Create mock metadata structure."""
        return {
            'test_table_1': MockDataGenerator._create_single_meta('test_table_1'),
            'test_table_2': MockDataGenerator._create_single_meta('test_table_2')
        }
    
    @staticmethod
    def _create_single_meta(table_name: str) -> ut.MetaJSON:
        """Create metadata for a single table."""
        return ut.MetaJSON(
            pcds_cols="COL1,COL2,COL3",
            pcds_types="NUMBER,NUMBER,VARCHAR",
            pcds_nrows=1000,
            pcds_id="COL1",
            aws_cols="col1,col2,col3",
            aws_types="bigint,decimal,varchar",
            aws_nrows=950,
            aws_id="col1",
            pcds_table=f"schema.{table_name}",
            aws_table=f"schema.{table_name}"
        )


# ===============================
# End-to-End Integration Tests
# ===============================

class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @patch('xlwings.Book')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('pandas.read_pickle')
    @patch('utils.read_meta_json')
    @patch('utils.download_froms3')
    @patch('utils.aws_creds_renew')
    @patch('load_dotenv')
    def test_main_function_flow(self, mock_load_dotenv, mock_aws_creds, mock_download,
                               mock_read_meta, mock_read_pickle, mock_remove,
                               mock_exists, mock_book_class, sample_config, 
                               mock_xlwings_workbook):
        """Test the main function flow."""
        # Setup mocks
        mock_exists.return_value = True
        mock_book_class.return_value = mock_xlwings_workbook
        mock_read_pickle.return_value = MockDataGenerator.create_mock_stats_data()
        mock_read_meta.return_value = MockDataGenerator.create_mock_meta_data()
        
        # Import and patch the main function
        from do_xlsx import main
        
        with patch('do_xlsx.create_argument_parser') as mock_parser, \
             patch('confection.Config') as mock_config_class:
            
            # Setup argument parser mock
            mock_args = Mock()
            mock_args.config = Path("test_config.cfg")
            mock_args.columns = DEFAULT_COLUMNS
            mock_parser.return_value.parse_args.return_value = mock_args
            
            # Setup config mock
            mock_config = Mock()
            mock_config.from_disk.return_value = sample_config.__dict__
            mock_config_class.return_value = mock_config
            
            # Mock ExcelReportGenerator
            with patch('do_xlsx.ExcelReportGenerator') as mock_generator_class:
                mock_generator = Mock()
                mock_generator.generate_report.return_value = "test_report.xlsx"
                mock_generator_class.return_value = mock_generator
                
                # Run main function
                main()
                
                # Verify key function calls
                mock_load_dotenv.assert_called_once()
                mock_aws_creds.assert_called_once()
                mock_generator_class.assert_called_once()
                mock_generator.generate_report.assert_called_once()


# ===============================
# Edge Cases and Boundary Tests
# ===============================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_stats_data(self, sample_config, mock_xlwings_workbook):
        """Test handling empty statistics data."""
        with patch('xlwings.Book', return_value=mock_xlwings_workbook):
            generator = ExcelReportGenerator(sample_config)
            
            with patch.object(generator, '_create_summary_sheet'), \
                 patch.object(generator, '_add_issue_descriptions'), \
                 patch.object(generator, '_finalize_workbook'):
                
                # Should handle empty data gracefully
                result = generator.generate_report({}, {})
                assert result is not None
    
    def test_malformed_stats_data(self, sample_config, mock_xlwings_workbook):
        """Test handling malformed statistics data."""
        malformed_data = {
            'bad_table': {
                'partition': {
                    # Missing required keys
                    'PCDS_STAT': pd.DataFrame(),
                    'AWS_STAT': pd.DataFrame()
                    # Missing PCDS_NAME, AWS_NAME, etc.
                }
            }
        }
        
        with patch('xlwings.Book', return_value=mock_xlwings_workbook):
            generator = ExcelReportGenerator(sample_config)
            
            with patch.object(generator, '_create_summary_sheet'), \
                 patch.object(generator, '_add_issue_descriptions'), \
                 patch.object(generator, '_finalize_workbook'):
                
                # Should handle malformed data gracefully
                with pytest.raises((KeyError, TypeError)):
                    generator.generate_report(malformed_data, {})
    
    def test_very_long_text_values(self):
        """Test handling very long text values."""
        long_text = 'a' * 1000
        df = pd.DataFrame({'col': [long_text, 'normal']})
        
        formatted = ExcelFormatter.format_cell_values(df, max_length=MAX_TEXT_LENGTH)
        
        assert len(formatted.loc[0, 'col']) == MAX_TEXT_LENGTH
        assert formatted.loc[1, 'col'] == 'normal'
    
    def test_unicode_text_values(self):
        """Test handling unicode text values."""
        unicode_df = pd.DataFrame({
            'unicode_col': ['café', '北京', '🚀', 'émoji']
        })
        
        formatted = ExcelFormatter.format_cell_values(unicode_df)
        
        # Should preserve unicode characters
        assert 'café' in formatted.values
        assert '北京' in formatted.values
        assert '🚀' in formatted.values
    
    def test_nan_and_inf_values(self):
        """Test handling NaN and infinity values."""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, np.inf, -np.inf],
            'col2': ['normal', None, 'text', '']
        })
        
        # Should not raise errors
        formatted = ExcelFormatter.format_cell_values(df)
        assert not formatted.empty
    
    def test_position_tracker_edge_values(self):
        """Test PositionTracker with edge values."""
        tracker = PositionTracker()
        
        # Test with very large values
        tracker.set_row(1000000)
        tracker.set_region_start(999999, 16384)  # Excel max columns
        
        pos = tracker.get_position()
        assert pos.row == 1000000
        
        region_pos = tracker.get_region_start()
        assert region_pos.row == 999999
        assert region_pos.col == 16384


# ===============================
# Compatibility Tests
# ===============================

class TestCompatibility:
    """Test compatibility with different data types and formats."""
    
    def test_different_pandas_dtypes(self):
        """Test with different pandas data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'datetime_col': pd.date_range('2024-01-01', periods=3),
            'category_col': pd.Categorical(['X', 'Y', 'Z'])
        })
        
        # Should handle all data types
        formatted = ExcelFormatter.format_cell_values(df)
        assert len(formatted) == len(df)
        assert len(formatted.columns) == len(df.columns)
    
    def test_different_index_types(self):
        """Test with different index types."""
        # Numeric index
        df1 = pd.DataFrame({'col': [1, 2, 3]}, index=[10, 20, 30])
        
        # String index
        df2 = pd.DataFrame({'col': [1, 2, 3]}, index=['a', 'b', 'c'])
        
        # Datetime index
        df3 = pd.DataFrame({'col': [1, 2, 3]}, 
                          index=pd.date_range('2024-01-01', periods=3))
        
        # All should be handled correctly
        for df in [df1, df2, df3]:
            formatted = ExcelFormatter.format_cell_values(df)
            assert not formatted.empty
    
    def test_multi_level_columns(self):
        """Test with multi-level column names."""
        arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        
        df = pd.DataFrame(np.random.randn(3, 4), columns=index)
        
        # Should handle multi-level columns
        formatted = ExcelFormatter.format_cell_values(df)
        assert not formatted.empty


# ===============================
# Configuration Tests
# ===============================

class TestConfiguration:
    """Test different configuration scenarios."""
    
    def test_default_configuration(self):
        """Test with default configuration values."""
        config = ut.StatConfig(
            input=ut.StatInputConfig(
                name="test",
                step="test_step",
                env_file=Path("test.env")
            ),
            output=ut.StatOutputConfig(
                folder=Path("output"),
                pickle_file=Path("test.pkl")
            )
        )
        
        generator = ExcelReportGenerator(config)
        assert generator.summary_columns == DEFAULT_COLUMNS
    
    def test_custom_columns_configuration(self):
        """Test with custom column configuration."""
        custom_columns = ['Custom1', 'Custom2', 'Custom3']
        config = ut.StatConfig(
            input=ut.StatInputConfig(
                name="test",
                step="test_step", 
                env_file=Path("test.env")
            ),
            output=ut.StatOutputConfig(
                folder=Path("output"),
                pickle_file=Path("test.pkl")
            )
        )
        
        generator = ExcelReportGenerator(config, custom_columns)
        assert generator.summary_columns == custom_columns
    
    def test_color_scheme_access(self):
        """Test color scheme configuration access."""
        # All color scheme keys should be valid colors
        for color_name in COLOR_SCHEME.values():
            rgb = ExcelFormatter.get_rgb_color(color_name)
            assert isinstance(rgb, tuple)
            assert len(rgb) == 3
            assert all(0 <= val <= 255 for val in rgb)
    
    def test_issue_descriptions_completeness(self):
        """Test that all issue descriptions are defined."""
        # Should have descriptions for issues 1-4
        expected_issues = {1, 2, 3, 4}
        actual_issues = set(ISSUE_DESCRIPTIONS.keys())
        assert actual_issues == expected_issues
        
        # All descriptions should be non-empty strings
        for description in ISSUE_DESCRIPTIONS.values():
            assert isinstance(description, str)
            assert len(description) > 0


# ===============================
# Cleanup and Utility Tests
# ===============================

class TestCleanupUtilities:
    """Test cleanup and utility functions."""
    
    def test_temp_file_cleanup(self):
        """Test temporary file cleanup."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            temp_path = tmp.name
        
        # File should exist
        assert os.path.exists(temp_path)
        
        # Clean up
        os.remove(temp_path)
        assert not os.path.exists(temp_path)
    
    def test_mock_cleanup(self, mock_xlwings_workbook):
        """Test that mocks are properly reset between tests."""
        # This test ensures mocks don't leak between tests
        mock_sheet = mock_xlwings_workbook.sheets[0]
        mock_sheet.reset_mock()
        
        # Should have no call history
        assert mock_sheet.call_count == 0

