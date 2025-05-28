"""
Restructured Excel Generation Module

This module creates Excel workbooks with statistical comparison results between PCDS and AWS data.
Refactored for better modularity, testability, and maintainability.
"""

import os
import argparse
import pickle
import xlwings as xw
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from PIL import ImageColor
from dotenv import load_dotenv

import utils
import utils_type as ut
from confection import Config


# ===============================
# Constants and Configuration
# ===============================

MAX_TEXT_LENGTH = 80
DEFAULT_COLUMNS = [
    'PCDS Table',
    'AWS Table', 
    'Columns Compared',
    'Columns Differed',
    'ISSUE 1*',
    'ISSUE 2*',
    'ISSUE 3*',
    'ISSUE 4*'
]

ISSUE_DESCRIPTIONS = {
    1: '0 is treated as missing value on AWS',
    2: 'Truncated precision on AWS, e.g., 12.2 on PCDS becomes 12 on AWS',
    3: 'Categorical value difference, e.g., PEAR on PCDS becomes PENULLAR on AWS',
    4: 'Other reasons that make column statistics different, e.g., dw_lod_tmp are recorded differently'
}

COLOR_SCHEME = {
    'header': 'black',
    'section_header': 'darkblue',
    'table_header': 'blue',
    'match': 'green',
    'mismatch': 'red',
    'issue_primary': 'darkgreen',
    'issue_secondary': 'green'
}


# ===============================
# Data Classes
# ===============================

@dataclass
class ExcelPosition:
    """Represents a position in an Excel worksheet."""
    row: int
    col: int
    
    def __post_init__(self):
        if self.row < 0 or self.col < 0:
            raise ValueError("Row and column must be non-negative")


@dataclass
class CellRange:
    """Represents a range of cells in Excel."""
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    
    def __post_init__(self):
        if self.start_row > self.end_row or self.start_col > self.end_col:
            raise ValueError("Start position must be before end position")
    
    @property
    def width(self) -> int:
        return self.end_col - self.start_col + 1
    
    @property
    def height(self) -> int:
        return self.end_row - self.start_row + 1


@dataclass
class ColumnStatistics:
    """Container for column statistics from both platforms."""
    pcds_stats: pd.DataFrame
    pcds_name: str
    pcds_mismatches: List[str]
    aws_stats: pd.DataFrame
    aws_name: str
    aws_mismatches: List[str]
    pcds_to_aws_mapping: Dict[str, str]
    
    def __post_init__(self):
        """Process statistics after initialization."""
        self.pcds_stats = self._transpose_dataframe(
            self.pcds_stats, 
            self.pcds_mismatches, 
            list(self.pcds_to_aws_mapping.keys())
        )
        self.aws_stats = self._transpose_dataframe(
            self.aws_stats,
            self.aws_mismatches,
            list(self.pcds_to_aws_mapping.values())
        )
    
    @staticmethod
    def _transpose_dataframe(df: pd.DataFrame, mismatches: List[str], 
                           columns: List[str]) -> pd.DataFrame:
        """Transpose and reorder dataframe columns."""
        if df.empty:
            return df
        
        # Ensure 'Type' column is first
        stat_cols = ['Type'] + [c for c in df.columns if c != 'Type']
        df = df[stat_cols] if all(c in df.columns for c in stat_cols) else df
        
        # Transpose
        df = df.T
        
        # Reorder columns - mismatches first, then others
        mismatches = sorted(mismatches, key=lambda x: columns.index(x) if x in columns else float('inf'))
        ordered_columns = mismatches + [c for c in columns if c in df.columns and c not in mismatches]
        
        # Only select columns that exist
        available_columns = [c for c in ordered_columns if c in df.columns]
        if available_columns:
            df = df[available_columns]
        
        return df


@dataclass
class IssueTracker:
    """Tracks different types of issues found in data comparison."""
    issues: List[set] = field(default_factory=lambda: [set() for _ in range(4)])
    
    def add_issue(self, issue_type: int, column_name: str):
        """Add an issue for a specific column."""
        if 0 <= issue_type < 4:
            self.issues[issue_type].add(column_name)
    
    def get_issue_counts(self) -> List[int]:
        """Get count of issues for each type."""
        return [len(issue_set) for issue_set in self.issues]
    
    def get_issues_for_column(self, column_name: str) -> List[int]:
        """Get issue types for a specific column."""
        return [i for i, issue_set in enumerate(self.issues) if column_name in issue_set]
    
    def clear(self):
        """Clear all issues."""
        self.issues = [set() for _ in range(4)]


# ===============================
# Position Tracker
# ===============================

class PositionTracker:
    """Tracks current position and manages cursor movement in Excel."""
    
    def __init__(self):
        self.worksheet_count = 0
        self.current_row = 0
        self.region_start_col = 0
        self.region_start_row = 0
    
    def increment_worksheets(self, count: int = 1):
        """Increment worksheet counter."""
        self.worksheet_count += count
    
    def increment_row(self, count: int = 1):
        """Move cursor down by specified rows."""
        self.current_row += count
    
    def set_row(self, row: int):
        """Set absolute row position."""
        self.current_row = row
    
    def set_region_start(self, row: int, col: int):
        """Set the start position for region operations."""
        self.region_start_row = row
        self.region_start_col = col
    
    def get_position(self) -> ExcelPosition:
        """Get current position."""
        return ExcelPosition(self.current_row, 0)
    
    def get_region_start(self) -> ExcelPosition:
        """Get region start position."""
        return ExcelPosition(self.region_start_row, self.region_start_col)


# ===============================
# Excel Formatting Utilities
# ===============================

class ExcelFormatter:
    """Handles Excel formatting operations."""
    
    @staticmethod
    def get_rgb_color(color_name: str) -> Tuple[int, int, int]:
        """Get RGB values for a color name."""
        try:
            return ImageColor.getrgb(color_name)
        except ValueError:
            # Default to black if color not found
            return ImageColor.getrgb('black')
    
    @staticmethod
    def apply_superscript(cell_range: xw.Range):
        """Apply superscript formatting to asterisk characters."""
        for cell in cell_range:
            if cell.value and isinstance(cell.value, str):
                asterisk_pos = str(cell.value).find('*')
                if asterisk_pos >= 0:
                    cell.characters[asterisk_pos:asterisk_pos+1].api.Font.Superscript = True
    
    @staticmethod
    def format_cell_values(df: pd.DataFrame, max_length: int = MAX_TEXT_LENGTH) -> pd.DataFrame:
        """Format cell values to prevent overflow."""
        return df.map(lambda x: x[:max_length] if isinstance(x, str) else x)
    
    @staticmethod
    def apply_number_formatting(worksheet: xw.Sheet, cell_range: CellRange):
        """Apply number formatting to a range of cells."""
        for row in range(cell_range.start_row, cell_range.end_row + 1):
            for col in range(cell_range.start_col, cell_range.end_col + 1):
                cell = worksheet[row, col]
                if cell.value is not None:
                    cell.number_format = '0.00'


# ===============================
# Excel Writer Classes
# ===============================

class ExcelRowWriter:
    """Handles writing rows to Excel worksheets."""
    
    def __init__(self, worksheet: xw.Sheet, position_tracker: PositionTracker):
        self.worksheet = worksheet
        self.tracker = position_tracker
        self.formatter = ExcelFormatter()
    
    def write_info_row(self, info_dict: Dict[Union[str, int], Any], 
                      color: Optional[str] = None, row: Optional[int] = None) -> int:
        """Write an information row with optional formatting."""
        target_row = row if row is not None else self.tracker.current_row
        end_col = 0
        
        for col_spec, value in info_dict.items():
            if ':' in str(col_spec):
                # Handle merged cells (e.g., "1:4" means merge columns 1-4)
                start_col, end_col = map(int, str(col_spec).split(':'))
                self.worksheet[target_row, start_col:end_col+1].merge()
                self.worksheet[target_row, start_col].value = value
                end_col += 1
            else:
                col_index = int(col_spec)
                self.worksheet[target_row, col_index].value = value
                end_col = col_index + 1
        
        # Apply color formatting if specified
        if color:
            rgb_color = self.formatter.get_rgb_color(color)
            self.worksheet[target_row, :end_col].font.color = rgb_color
        
        return end_col
    
    def write_dataframe(self, df: pd.DataFrame) -> int:
        """Write a dataframe to the worksheet."""
        if df.empty:
            return 0
        
        start_row = self.tracker.current_row
        formatted_df = self.formatter.format_cell_values(df)
        
        # Write the dataframe
        self.worksheet[start_row, 1].value = formatted_df
        
        return len(df)


class ExcelStylist:
    """Handles styling and formatting of Excel cells."""
    
    def __init__(self, worksheet: xw.Sheet):
        self.worksheet = worksheet
        self.formatter = ExcelFormatter()
    
    def style_comparison_region(self, pcds_range: CellRange, aws_range: CellRange):
        """Apply styling to comparison regions based on value matches."""
        # Ensure ranges have the same dimensions
        if pcds_range.width != aws_range.width or pcds_range.height != aws_range.height:
            raise ValueError("PCDS and AWS ranges must have the same dimensions")
        
        for row_offset in range(pcds_range.height):
            for col_offset in range(pcds_range.width):
                pcds_row = pcds_range.start_row + row_offset
                pcds_col = pcds_range.start_col + col_offset
                aws_row = aws_range.start_row + row_offset
                aws_col = aws_range.start_col + col_offset
                
                pcds_cell = self.worksheet[pcds_row, pcds_col]
                aws_cell = self.worksheet[aws_row, aws_col]
                
                # Apply number formatting
                pcds_cell.number_format = '0.00'
                aws_cell.number_format = '0.00'
                
                # Compare values and apply colors
                if self._values_equal(pcds_cell.value, aws_cell.value):
                    color = self.formatter.get_rgb_color(COLOR_SCHEME['match'])
                else:
                    color = self.formatter.get_rgb_color(COLOR_SCHEME['mismatch'])
                
                pcds_cell.font.color = color
                aws_cell.font.color = color
    
    @staticmethod
    def _values_equal(val1: Any, val2: Any, tolerance: float = 1e-10) -> bool:
        """Check if two values are equal with tolerance for floating point."""
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False
        
        try:
            # Try numeric comparison with tolerance
            return abs(float(val1) - float(val2)) < tolerance
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(val1) == str(val2)


# ===============================
# Issue Analysis
# ===============================

class IssueAnalyzer:
    """Analyzes differences between PCDS and AWS statistics to categorize issues."""
    
    @staticmethod
    def analyze_column_issues(pcds_stats: pd.DataFrame, aws_stats: pd.DataFrame) -> IssueTracker:
        """Analyze statistics to identify and categorize issues."""
        tracker = IssueTracker()
        
        if pcds_stats.empty or aws_stats.empty:
            return tracker
        
        # Ensure both dataframes have the same columns
        common_columns = pcds_stats.columns.intersection(aws_stats.columns)
        
        for column in common_columns:
            if column == 'Type':
                continue
            
            try:
                pcds_row = pcds_stats[column]
                aws_row = aws_stats[column]
                
                # Count differences
                differences = IssueAnalyzer._count_differences(pcds_row, aws_row)
                
                # Categorize issue based on patterns
                issue_type = IssueAnalyzer._categorize_issue(pcds_row, aws_row, differences)
                
                if issue_type is not None:
                    tracker.add_issue(issue_type, column)
                    
            except (KeyError, IndexError) as e:
                # Log error but continue processing
                print(f"Warning: Error processing column {column}: {e}")
                continue
        
        return tracker
    
    @staticmethod
    def _count_differences(pcds_series: pd.Series, aws_series: pd.Series) -> int:
        """Count the number of differences between two series."""
        try:
            return (pcds_series != aws_series).sum()
        except (ValueError, TypeError):
            # Handle comparison of different data types
            return sum(1 for p, a in zip(pcds_series, aws_series) if p != a)
    
    @staticmethod
    def _categorize_issue(pcds_series: pd.Series, aws_series: pd.Series, diff_count: int) -> Optional[int]:
        """Categorize the type of issue based on statistical patterns."""
        try:
            # Get key statistics
            pcds_n_unique = pcds_series.get('N_Unique', 0)
            aws_n_unique = aws_series.get('N_Unique', 0)
            pcds_n_missing = pcds_series.get('N_Missing', 0)
            aws_n_missing = aws_series.get('N_Missing', 0)
            pcds_freq = pcds_series.get('Freq')
            aws_freq = aws_series.get('Freq')
            
            # Issue 1: Zero treated as missing on AWS
            if (pcds_n_unique == aws_n_unique + 1 and 
                pcds_n_missing == 0 and aws_n_missing > 0):
                return 0
            
            # Issue 2: Precision truncation
            elif (pcds_n_unique > aws_n_unique and diff_count <= 5):
                return 1
            
            # Issue 3: Categorical differences
            elif ((pcds_freq is not None and not pd.isna(pcds_freq)) or 
                  (aws_freq is not None and not pd.isna(aws_freq))) and diff_count <= 3:
                return 2
            
            # Issue 4: Other differences
            elif diff_count > 0:
                return 3
            
            return None
            
        except (KeyError, AttributeError, TypeError):
            # If we can't categorize, default to "other"
            return 3 if diff_count > 0 else None


# ===============================
# Main Excel Generator
# ===============================

class ExcelReportGenerator:
    """Main class for generating Excel reports from statistics data."""
    
    def __init__(self, config: ut.StatConfig, summary_columns: List[str] = None):
        self.config = config
        self.summary_columns = summary_columns or DEFAULT_COLUMNS
        self.workbook = None
        self.position_tracker = PositionTracker()
        
    def generate_report(self, stats_data: Dict[str, Dict], meta_json: Dict[str, ut.MetaJSON]) -> str:
        """Generate the complete Excel report."""
        # Setup paths and files
        excel_path = os.path.join(self.config.output.folder, f'{self.config.input.step}.xlsx')
        
        # Remove existing file
        if os.path.exists(excel_path):
            os.remove(excel_path)
        
        # Create workbook
        self.workbook = xw.Book()
        self.workbook.save(excel_path)
        
        try:
            # Create summary sheet
            self._create_summary_sheet()
            
            # Process each table
            for table_name, table_stats in stats_data.items():
                self._create_table_sheet(table_name, table_stats, meta_json.get(table_name))
            
            # Add issue descriptions
            self._add_issue_descriptions()
            
            # Auto-fit columns and save
            self._finalize_workbook()
            self.workbook.save(excel_path)
            
            return excel_path
            
        except Exception as e:
            print(f"Error generating Excel report: {e}")
            raise
        finally:
            if self.workbook:
                self.workbook.close()
    
    def _create_summary_sheet(self):
        """Create and setup the summary sheet."""
        summary_sheet = self.workbook.sheets[0]
        summary_sheet.name = 'SUMMARY'
        
        # Write headers
        summary_sheet[0, :len(self.summary_columns)].value = self.summary_columns
        
        # Apply header formatting
        formatter = ExcelFormatter()
        formatter.apply_superscript(summary_sheet[0, :len(self.summary_columns)])
    
    def _create_table_sheet(self, table_name: str, table_stats: Dict, meta_info: ut.MetaJSON):
        """Create a sheet for a specific table's statistics."""
        try:
            # Create or get worksheet
            try:
                worksheet = self.workbook.sheets.add(table_name.upper(), 
                                                   after=self.workbook.sheets[self.position_tracker.worksheet_count])
            except ValueError:
                worksheet = self.workbook.sheets[table_name.upper()]
            
            worksheet.clear()
            self.position_tracker.increment_worksheets()
            self.position_tracker.set_row(0)
            
            # Setup writers and stylist
            row_writer = ExcelRowWriter(worksheet, self.position_tracker)
            stylist = ExcelStylist(worksheet)
            
            # Track overall issues for this table
            table_issue_tracker = IssueTracker()
            
            # Process each vintage/partition
            for partition_name, partition_data in table_stats.items():
                column_stats = ColumnStatistics(**partition_data, 
                                              pcds_to_aws_mapping=meta_info.pcds.column_mapping)
                
                # Write partition header
                row_writer.write_info_row(
                    {'0': 'Vintage: ', '1:4': partition_name}, 
                    COLOR_SCHEME['section_header']
                )
                self.position_tracker.increment_row(2)
                
                # Write PCDS section
                self._write_platform_section(row_writer, stylist, 'PCDS', column_stats, table_issue_tracker)
                
                # Write AWS section  
                self._write_platform_section(row_writer, stylist, 'AWS', column_stats, table_issue_tracker)
                
                # Style comparison regions
                self._style_comparison_regions(stylist, column_stats)
                
                # Write issue summary for this partition
                self._write_partition_issues(row_writer, column_stats, table_issue_tracker)
            
            # Update summary sheet with table info
            self._update_summary_sheet(table_name, column_stats, table_issue_tracker)
            
        except Exception as e:
            print(f"Error creating sheet for table {table_name}: {e}")
            raise
    
    def _write_platform_section(self, row_writer: ExcelRowWriter, stylist: ExcelStylist,
                               platform: str, column_stats: ColumnStatistics, 
                               issue_tracker: IssueTracker):
        """Write a platform section (PCDS or AWS) to the worksheet."""
        if platform == 'PCDS':
            stats_df = column_stats.pcds_stats
            platform_name = column_stats.pcds_name
        else:
            stats_df = column_stats.aws_stats
            platform_name = column_stats.aws_name
        
        # Write platform header
        row_writer.write_info_row(
            {'0': f'{platform}: ', '1:4': platform_name},
            COLOR_SCHEME['table_header']
        )
        self.position_tracker.increment_row()
        
        # Set region start for styling
        if platform == 'PCDS':
            self.position_tracker.set_region_start(self.position_tracker.current_row + 1, 2)
        
        # Write statistics dataframe
        if not stats_df.empty:
            rows_written = row_writer.write_dataframe(stats_df)
            self.position_tracker.increment_row(rows_written + 2)
        else:
            self.position_tracker.increment_row(2)
    
    def _style_comparison_regions(self, stylist: ExcelStylist, column_stats: ColumnStatistics):
        """Apply styling to comparison regions."""
        if column_stats.pcds_stats.empty or column_stats.aws_stats.empty:
            return
        
        # Calculate region dimensions
        region_start = self.position_tracker.get_region_start()
        n_rows = len(column_stats.pcds_stats) - 1  # Exclude header row
        n_cols = len(column_stats.aws_mismatches)
        
        if n_rows > 0 and n_cols > 0:
            pcds_range = CellRange(
                region_start.row, region_start.col,
                region_start.row + n_rows - 1, region_start.col + n_cols - 1
            )
            
            aws_range = CellRange(
                region_start.row + n_rows + 4, region_start.col,
                region_start.row + 2 * n_rows + 3, region_start.col + n_cols - 1
            )
            
            stylist.style_comparison_region(pcds_range, aws_range)
    
    def _write_partition_issues(self, row_writer: ExcelRowWriter, column_stats: ColumnStatistics,
                               issue_tracker: IssueTracker):
        """Write issue summary for a partition."""
        # Analyze issues
        partition_issues = IssueAnalyzer.analyze_column_issues(
            column_stats.pcds_stats[column_stats.pcds_mismatches] if column_stats.pcds_mismatches else pd.DataFrame(),
            column_stats.aws_stats[column_stats.aws_mismatches] if column_stats.aws_mismatches else pd.DataFrame()
        )
        
        # Build issue row data
        issue_data = {'0:2': 'Issue: '}
        
        for i, column in enumerate(column_stats.aws_mismatches):
            issue_types = partition_issues.get_issues_for_column(column)
            if issue_types:
                issue_data[str(i + 3)] = f"issue_{issue_types[0] + 1}"
        
        row_writer.write_info_row(issue_data)
        self.position_tracker.increment_row()
        
        # Merge issues into table tracker
        for i in range(4):
            issue_tracker.issues[i].update(partition_issues.issues[i])
    
    def _update_summary_sheet(self, table_name: str, column_stats: ColumnStatistics, 
                            issue_tracker: IssueTracker):
        """Update the summary sheet with table information."""
        summary_sheet = self.workbook.sheets['SUMMARY']
        row_index = self.position_tracker.worksheet_count  # Use worksheet count as row index
        
        # Calculate summary statistics
        n_columns = len(column_stats.pcds_stats.columns) if not column_stats.pcds_stats.empty else 0
        n_issues = len(column_stats.aws_mismatches)
        issue_counts = issue_tracker.get_issue_counts()
        
        # Write summary row
        summary_data = [
            column_stats.pcds_name,
            column_stats.aws_name,
            n_columns,
            n_issues
        ] + issue_counts
        
        summary_sheet[row_index, :len(summary_data)].value = summary_data
    
    def _add_issue_descriptions(self):
        """Add issue descriptions to the summary sheet."""
        summary_sheet = self.workbook.sheets['SUMMARY']
        row_writer = ExcelRowWriter(summary_sheet, PositionTracker())
        
        # Find the last used row
        last_row = summary_sheet.used_range.last_cell.row + 4
        
        # Write issue descriptions
        for issue_num, description in ISSUE_DESCRIPTIONS.items():
            color = COLOR_SCHEME['issue_primary'] if issue_num == 1 else COLOR_SCHEME['issue_secondary']
            row_writer.write_info_row(
                {'0': f'Issue {issue_num}: ', '1:5': description},
                color,
                last_row + issue_num - 1
            )
    
    def _finalize_workbook(self):
        """Finalize the workbook with formatting."""
        summary_sheet = self.workbook.sheets['SUMMARY']
        summary_sheet.used_range.api.Columns.AutoFit()


# ===============================
# CLI and Main Function
# ===============================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate Excel reports from statistics comparison data'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Config file for making Excel report on local Windows PC',
        default=r'files\inputs\config_stats_0505.cfg'
    )
    parser.add_argument(
        '--columns',
        nargs='+',
        type=str,
        help='Excel summary sheet columns',
        default=DEFAULT_COLUMNS
    )
    return parser


def main():
    """Main function to generate Excel reports."""
    # Parse arguments
    args = create_argument_parser().parse_args()
    
    # Load configuration
    config = Config().from_disk(args.config)
    config = ut.StatConfig(**config)
    
    # Setup environment
    load_dotenv(config.input.env_file)
    utils.aws_creds_renew()
    
    try:
        # Download data if needed
        step_file = os.path.join(config.output.folder, 'step_stat.pkl')
        if not os.path.exists(step_file):
            utils.download_froms3(
                utils.urljoin(f'{config.output.s3_config.run}/', config.input.name),
                config.output.folder,
                config.input.step
            )
        
        # Load data
        stats_data = pd.read_pickle(config.output.pickle_file)
        meta_json = utils.read_meta_json(config.input.json_config['meta'])
        
        # Generate report
        generator = ExcelReportGenerator(config, args.columns)
        excel_path = generator.generate_report(stats_data, meta_json)
        
        print(f"Excel report generated successfully: {excel_path}")
        
    except Exception as e:
        print(f"Error generating Excel report: {e}")
        raise


if __name__ == "__main__":
    main()