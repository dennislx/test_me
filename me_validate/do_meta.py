"""
Refactored Meta Analysis Module

This module conducts meta information analysis by comparing PCDS and AWS database schemas,
including column types, row counts, and data consistency checks.
"""

import re
import os
import pickle
import json
import argparse
import csv
from datetime import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import pandas as pd
import pyathena as pa
from loguru import logger
from confection import Config
from tqdm import tqdm
from dotenv import load_dotenv

import utils
import utils_type as ut


# Constants
SEP = '; '
PCDS_SQL_META = """
    SELECT
        column_name,
        data_type || 
        CASE
            WHEN data_type = 'NUMBER' THEN 
                CASE WHEN data_precision IS NULL AND data_scale IS NULL
                    THEN NULL
                ELSE
                    '(' || TO_CHAR(data_precision) || ',' || TO_CHAR(data_scale) || ')'
                END
            WHEN data_type LIKE '%CHAR%' THEN
                '(' || TO_CHAR(data_length) || ')'
            ELSE NULL
        END AS data_type
    FROM all_tab_cols
    WHERE table_name = UPPER('{table}')
    ORDER BY column_id
""".strip()

PCDS_SQL_NROW = "SELECT COUNT(*) AS nrow FROM {table}"
PCDS_SQL_DATE = "SELECT {date}, count(*) AS nrows FROM {table} GROUP BY {date}"

AWS_SQL_META = """
    SELECT column_name, data_type 
    FROM information_schema.columns
    WHERE table_schema = LOWER('{db}') AND table_name = LOWER('{table}')
""".strip()

AWS_SQL_NROW = "SELECT COUNT(*) AS nrow FROM {db}.{table}"
AWS_SQL_DATE = "SELECT {date}, count(*) AS nrows FROM {db}.{table} GROUP BY {date}"


class PullStatus(Enum):
    """Enumeration of possible data pull statuses."""
    NONEXIST_PCDS = 'Nonexisting PCDS Table'
    NONEXIST_AWS = 'Nonexisting AWS Table'
    NONDATE_PCDS = 'Nonexisting Date Variable in PCDS'
    NONDATE_AWS = 'Nonexisting Date Variable in AWS'
    EMPTY_PCDS = 'Empty PCDS Table'
    EMPTY_AWS = 'Empty AWS Table'
    NO_MAPPING = 'Column Mapping Not Provided'
    SUCCESS = 'Successful Data Access'


@dataclass
class MetaResult:
    """Container for metadata comparison results."""
    row_unmatch: bool = False
    row_unmatch_details: str = ''
    time_span_unmatch: bool = False
    time_span_variable: str = ''
    time_unmatch_details: str = ''
    column_type_unmatch: bool = False
    type_unmatch_details: str = ''
    pcds_extra_columns: bool = False
    pcds_unique_columns: str = ''
    aws_extra_columns: bool = False
    aws_unique_columns: str = ''
    uncaptured_column_mappings: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV output."""
        return {
            'Row UnMatch': self.row_unmatch,
            'Row UnMatch Details': self.row_unmatch_details,
            'Time Span UnMatch': self.time_span_unmatch,
            'Time Span Variable': self.time_span_variable,
            'Time UnMatch Details': self.time_unmatch_details,
            'Column Type UnMatch': self.column_type_unmatch,
            'Type UnMatch Details': self.type_unmatch_details,
            'PCDS Extra Columns': self.pcds_extra_columns,
            'PCDS Unique Columns': self.pcds_unique_columns,
            'AWS Extra Columns': self.aws_extra_columns,
            'AWS Unique Columns': self.aws_unique_columns,
            'Uncaptured Column Mappings': self.uncaptured_column_mappings,
        }


class DatabaseConnector:
    """Handles database connections for PCDS and AWS."""
    
    @staticmethod
    def pcds_connect(service_name: str, ldap_service: str = 'ldap://oid.barcapint.com:4050'):
        """Create PCDS database connection."""
        svc2server = {
            'p_uscb_cnsmrlnd_svc': '21P',
            'p_uscb_rft_svc': '30P',
            'pcds_svc': '00P'
        }
        
        import oracledb
        if service_name not in svc2server:
            raise pd.errors.DatabaseError("Service Name Is Not Provided")
        
        dns_tns = utils.solve_ldap(f'{ldap_service}/{service_name},cn=OracleContext,dc=barcapint,dc=com')
        pcds_pwd = f'PCDS_{svc2server.get(service_name)}'
        usr, pwd = os.environ['PCDS_USR'], os.environ[pcds_pwd]
        return oracledb.connect(user=usr, password=pwd, dsn=dns_tns)

    @staticmethod
    def aws_connect():
        """Create AWS Athena connection."""
        return pa.connect(
            s3_staging_dir="s3://355538383407-us-east-1-athena-output/uscb-analytics/",
            region_name="us-east-1",
        )


class DataTypeMapper:
    """Handles mapping between PCDS and AWS data types."""
    
    @staticmethod
    def map_pcds_to_aws(pcds_dtype: str, aws_dtype: str) -> bool:
        """
        Map PCDS data types to AWS data types and check compatibility.
        
        Mapping rules:
        NUMBER => decimal(38,0) or double
        NUMBER(d,0) => decimal(d+{1,2},0)
        NUMBER(d,n) => decimal(d+{1,2},n)
        VARCHAR2(n) => varchar(n)
        TIMESTAMP(n) => timestamp(n-{3})
        DATE => date or timestamp
        CHAR(n) => problematic if n != 1
        """
        match pcds_dtype:
            case 'NUMBER':
                return aws_dtype == 'double'
            case _ if pcds_dtype.startswith('NUMBER'):
                match = re.match(r'NUMBER\(\d*,(\d+)\)', pcds_dtype)
                if match:
                    scale = match.group(1)
                    aws_match = re.match(r'decimal\(\d*,(\d+)\)', aws_dtype)
                    return bool(aws_match and aws_match.group(1) == scale)
                return False
            case _ if pcds_dtype.startswith('VARCHAR2'):
                return pcds_dtype.replace('VARCHAR2', 'varchar') == aws_dtype
            case _ if pcds_dtype.startswith('CHAR'):
                match = re.match(r'CHAR\((\d+)\)', pcds_dtype)
                if match:
                    n = match.group(1)
                    return not (aws_dtype.startswith('VARCHAR') and n != '1')
                return False
            case 'DATE':
                return aws_dtype == 'date' or aws_dtype.startswith('timestamp')
            case _ if pcds_dtype.startswith('TIMESTAMP'):
                return aws_dtype.startswith('timestamp')
            case _:
                logger.info(f">>> Mismatched type: PCDS ({pcds_dtype}) ==> AWS ({aws_dtype})")
                return False


class MetaAnalyzer:
    """Main class for conducting meta analysis."""
    
    def __init__(self, config: ut.MetaConfig):
        self.config = config
        self.col_maps = utils.process_mapping(config.column_maps)
        self.mismatch_data = {}
        self.next_data = {}
        
    def process_pcds_meta(self, info_str: str, col_map: Optional[str]) -> Tuple[Dict, bool]:
        """Process PCDS metadata including column types and row counts."""
        service, table = info_str.split('.', maxsplit=1)
        logger.info(f"\tStart processing {info_str}")
        
        try:
            with DatabaseConnector.pcds_connect(service) as conn:
                df_type = utils.costly_query(PCDS_SQL_META.format(table=table), conn)
                df_nrow = utils.costly_query(PCDS_SQL_NROW.format(table=table), conn)
        except pd.errors.DatabaseError:
            logger.warning(f"Couldn't find {table.upper()} in {service.upper()}")
            raise ut.NONEXIST_TABLE("PCDS View Not Existing")
        
        # Apply column mapping if available
        rename_columns = {}
        df_type.columns = [x.lower() for x in df_type.columns]
        
        if col_map and col_map in self.col_maps:
            rename_columns = self.col_maps[col_map]
        
        df_type['aws_colname'] = df_type['column_name'].map(rename_columns)
        return {'column': df_type, 'row': df_nrow}, len(rename_columns) > 0

    def process_pcds_date(self, info_str: str, date_var: str) -> pd.DataFrame:
        """Process PCDS date variable information."""
        service, table = info_str.split('.', maxsplit=1)
        
        try:
            with DatabaseConnector.pcds_connect(service) as conn:
                df_meta = utils.costly_query(
                    PCDS_SQL_DATE.format(table=table, date=date_var), conn
                )
            logger.info(f"\tFinish Processing {info_str}")
        except pd.errors.DatabaseError:
            logger.warning(f"Column {date_var.upper()} not found in {table.upper()}")
            raise ut.NONEXIST_DATEVAR("Date-like Variable Not In PCDS")
        
        return df_meta

    def process_aws_meta(self, info_str: str) -> Dict:
        """Process AWS metadata including column types and row counts."""
        database, table = info_str.split('.', maxsplit=1)
        conn = DatabaseConnector.aws_connect()
        logger.info(f"\tStart processing {info_str}")
        
        try:
            df_type = utils.costly_query(AWS_SQL_META.format(table=table, db=database), conn)
            df_nrow = utils.costly_query(AWS_SQL_NROW.format(table=table, db=database), conn)
        except pd.errors.DatabaseError:
            logger.warning(f"Couldn't find {table.lower()} in {database.lower()}")
            raise ut.NONEXIST_TABLE("AWS View Not Existing")
        
        return {'column': df_type, 'row': df_nrow}

    def process_aws_date(self, info_str: str, date_var: str) -> pd.DataFrame:
        """Process AWS date variable information."""
        database, table = info_str.split('.', maxsplit=1)
        conn = DatabaseConnector.aws_connect()
        
        try:
            df_meta = utils.costly_query(
                AWS_SQL_DATE.format(table=table, db=database, date=date_var), conn
            )
            logger.info(f"\tFinish Processing {info_str}")
        except pd.errors.DatabaseError:
            logger.warning(f"Column {date_var.upper()} not found in {table.upper()}")
            raise ut.NONEXIST_DATEVAR("Date-like Variable Not In AWS")
        
        return df_meta

    def compare_metadata(self, pcds_meta: Dict, aws_meta: Dict) -> MetaResult:
        """Compare PCDS and AWS metadata and return results."""
        pcds_c, aws_c = pcds_meta['column'], aws_meta['column']
        
        # Handle missing column mappings
        if pcds_c['aws_colname'].isna().all():
            pcds_c['aws_colname'] = pcds_c['column_name'].str.lower()
            uncaptured = "Column Mapping Not Provided"
        else:
            uncaptured = ""
        
        # Process merge and comparisons
        profile = self._process_merge(pcds_c, aws_c)
        logger.info(">>> Finish Merging Type Data")
        
        # Update next data dictionary
        d = (
            profile.col_mapping
            .drop(columns='type_match')
            .apply(lambda x: SEP.join(x.tolist()), axis=0)
            .to_dict()
        )
        
        self.next_data.update(
            pcds_cols=d['column_name_pcds'],
            pcds_types=d['data_type_pcds'],
            pcds_nrows=int(pcds_meta['row'].iloc[0].item()),
            aws_cols=d['column_name_aws'],
            aws_types=d['data_type_aws'],
            aws_nrows=int(aws_meta['row'].iloc[0].item()),
        )
        
        return MetaResult(
            row_unmatch=self.next_data['pcds_nrows'] != self.next_data['aws_nrows'],
            row_unmatch_details=f"PCDS({self.next_data['pcds_nrows']}) : AWS({self.next_data['aws_nrows']})",
            type_unmatch_details=profile.mismatches,
            column_type_unmatch=len(profile.mismatches) > 0,
            pcds_extra_columns=len(profile.unique_pcds) > 0,
            pcds_unique_columns=SEP.join(profile.unique_pcds),
            aws_extra_columns=len(profile.unique_aws) > 0,
            aws_unique_columns=SEP.join(profile.unique_aws),
            uncaptured_column_mappings=uncaptured or profile.uncaptured,
        )

    def _process_merge(self, pcds: pd.DataFrame, aws: pd.DataFrame) -> ut.MetaMergeResult:
        """Process the merge between PCDS and AWS column data."""
        # Find unmapped columns
        unmapped_pcds = (
            pcds.query('aws_colname != aws_colname')
            ['column_name'].str.lower().to_list()
        )
        unmapped_aws = (
            aws.query('~column_name.isin(@pcds.aws_colname)')
            ['column_name'].to_list()
        )
        
        # Find common mappings
        map_uncaptured = self._find_common_mappings(unmapped_pcds, unmapped_aws)
        map_uncaptured = {k.upper(): v for k, v in map_uncaptured.items()}
        uncaptured = SEP.join(f'{k}->{v}' for k, v in map_uncaptured.items())
        
        # Update column mappings
        pcds['aws_colname'] = (
            pcds['aws_colname']
            .combine_first(pcds['column_name'].map(map_uncaptured))
        )
        
        # Merge dataframes
        df_match = pd.merge(
            pcds, aws,
            left_on='aws_colname', right_on='column_name',
            suffixes=['_pcds', '_aws'],
            how='outer', indicator=True
        )
        
        # Extract unique columns
        pcds_cols = ['column_name_pcds', 'data_type_pcds']
        pcds_unique = df_match.query('_merge == "left_only"')[pcds_cols]
        aws_cols = ['column_name_aws', 'data_type_aws']
        aws_unique = df_match.query('_merge == "right_only"')[aws_cols]
        
        # Process type matching
        merged = (
            df_match.query('_merge == "both"')
            .drop(columns=['aws_colname', '_merge'])
        )
        merged['type_match'] = merged.apply(self._apply_type_mapping, axis=1)
        self.mismatch_data = merged.query('~type_match')
        
        mismatch_d = self.mismatch_data[['data_type_pcds', 'data_type_aws']].drop_duplicates()
        mismatched = SEP.join(f'{row.data_type_pcds}->{row.data_type_aws}' 
                             for row in mismatch_d.itertuples())
        
        return ut.MetaMergeResult(
            unique_pcds=pcds_unique['column_name_pcds'].str.upper().to_list(),
            unique_aws=aws_unique['column_name_aws'].str.lower().to_list(),
            col_mapping=merged,
            mismatches=mismatched,
            uncaptured=uncaptured
        )

    def _apply_type_mapping(self, row) -> bool:
        """Apply data type mapping logic."""
        return DataTypeMapper.map_pcds_to_aws(
            row.data_type_pcds, 
            row.data_type_aws
        )

    def _find_common_mappings(self, list_a, list_b):
        """Find common mappings between two lists using prefix matching."""
        from collections import defaultdict
        import itertools as it
        
        def prefix_cmp(a, b):
            return a.startswith(b) or b.startswith(a)
        
        result, visited = {}, set()
        prefix_d = defaultdict(list)
        
        for x, y in it.product(list_a, list_b):
            if prefix_cmp(x, y):
                prefix_d[x].append(y)
        
        # Exact match prioritized
        for x in list_a:
            if x in list_b and x not in visited:
                result[x] = x
                visited.add(x)
        
        # Prefix match for the rest
        for x in list_a:
            if x in result:
                continue
            for y in prefix_d[x]:
                if y not in visited:
                    result[x] = y
                    visited.add(y)
                    break
        
        return result


class MetaAnalysisRunner:
    """Main runner for the meta analysis process."""
    
    def __init__(self, config_path: Path):
        self.config = Config().from_disk(config_path)
        self.config = ut.MetaConfig(**self.config)
        self.analyzer = MetaAnalyzer(self.config)
        
    def run(self):
        """Execute the complete meta analysis process."""
        self._setup()
        
        tbl_list = utils.read_input_excel(self.config.input.table)
        tbl_list = tbl_list.groupby('aws_tbl').first().reset_index()
        
        start_row, end_row = self.config.input.range
        total = len(tbl_list)
        has_header = False
        
        try:
            utils.start_run()
            utils.aws_creds_renew(15 * 60)
            
            for i, row in enumerate(tqdm(
                tbl_list.itertuples(), desc='Processing ...', total=total
            ), start=1):
                
                if i < start_row or i > end_row:
                    has_header = False
                    continue
                
                result = self._process_single_table(row)
                self._write_results(row, result, has_header)
                has_header = True
                utils.separator()
                
        except Exception as e:
            logger.error("Error in processing ... Stopping")
            logger.exception(e)
            raise
        finally:
            self._cleanup()

    def _setup(self):
        """Setup logging, environment, and output directories."""
        start_row, end_row = self.config.input.range
        try:
            assert start_row <= 1
            os.remove(self.config.output.csv.file)
        except (TypeError, AssertionError, FileNotFoundError):
            pass
        
        os.makedirs(self.config.output.folder, exist_ok=True)
        logger.add(**self.config.output.log.todict())
        logger.info('Configuration:\n' + self.config.tolog())
        load_dotenv(self.config.input.env)

    def _process_single_table(self, row) -> Tuple[PullStatus, MetaResult]:
        """Process a single table comparison."""
        name = row.pcds_tbl.split('.')[1].lower()
        logger.info(f">>> Start {name}")
        
        # Initialize data structures
        self.analyzer.mismatch_data = {}
        self.analyzer.next_data = self.config.output.next.fields.copy()
        
        self.analyzer.next_data.update(
            pcds_tbl=row.pcds_tbl,
            aws_tbl=row.aws_tbl,
            pcds_id=row.pcds_id,
            aws_id=row.aws_id,
            last_modified=dt.now().strftime('%Y-%m-%d'),
        )
        
        pull_status = PullStatus.SUCCESS
        result = MetaResult()
        
        try:
            # Process PCDS metadata
            pcds_meta, exist_mapping = self.analyzer.process_pcds_meta(
                row.pcds_tbl, row.col_map
            )
            
            if not exist_mapping:
                pull_status = PullStatus.NO_MAPPING
            
            # Process AWS metadata
            aws_meta = self.analyzer.process_aws_meta(row.aws_tbl)
            
            # Compare metadata if successful
            if pull_status == PullStatus.SUCCESS:
                result = self.analyzer.compare_metadata(pcds_meta, aws_meta)
            
            # Process date comparisons if needed
            if result.row_unmatch:
                try:
                    pcds_date = self.analyzer.process_pcds_date(row.pcds_tbl, row.pcds_id)
                    aws_date = self.analyzer.process_aws_date(row.aws_tbl, row.aws_id)
                    date_result = self._process_date_comparison(
                        pcds_date, row.pcds_id, aws_date, row.aws_id
                    )
                    result.time_span_unmatch = date_result['Time Span UnMatch']
                    result.time_span_variable = date_result['Time Span Variable']
                    result.time_unmatch_details = date_result['Time UnMatch Details']
                except (ut.NONEXIST_DATEVAR, TypeError):
                    pass
            
        except ut.NONEXIST_TABLE:
            if 'pcds' in str(row.pcds_tbl).lower():
                pull_status = PullStatus.NONEXIST_PCDS
            else:
                pull_status = PullStatus.NONEXIST_AWS
        except ut.NONEXIST_DATEVAR:
            pull_status = PullStatus.NONDATE_PCDS
        
        return pull_status, result

    def _process_date_comparison(self, cnt_pcds, pcds_id, cnt_aws, aws_id):
        """Process date variable comparisons between PCDS and AWS."""
        # Convert types if there are mismatches
        try:
            mismatch = self.analyzer.mismatch_data.query(
                f'column_name_aws == "{aws_id}"'
            ).squeeze()
            if len(mismatch) > 0:
                self._convert_type(cnt_pcds, pcds_id, mismatch.data_type_pcds)
                self._convert_type(cnt_aws, aws_id, mismatch.data_type_aws)
        except AttributeError:
            pass

        cnt_pcds[pcds_id] = cnt_pcds[pcds_id].astype(str)
        cnt_aws[aws_id] = cnt_aws[aws_id].astype(str)
        
        df_all = pd.merge(
            cnt_pcds, cnt_aws,
            left_on=pcds_id, right_on=aws_id,
            suffixes=['_pcds', '_aws'], how='outer'
        )
        
        time_mismatch = df_all.query('NROWS != nrows')
        time_excludes = SEP.join(utils.get_datesort(
            time_mismatch[pcds_id].fillna(time_mismatch[aws_id])
        ))
        
        self.analyzer.next_data.update(time_excludes=time_excludes)
        
        return {
            'Time Span UnMatch': len(time_mismatch) > 0,
            'Time Span Variable': f'{pcds_id} : {aws_id}',
            'Time UnMatch Details': time_excludes
        }

    def _convert_type(self, df, var, dtype):
        """Convert data types for date comparison."""
        match dtype.lower():
            case 'date':
                df[var] = df[var].dt.strftime('%Y-%m-%d')
            case _ if dtype.startswith('timestamp'):
                df[var] = df[var].dt.strftime('%Y-%m-%d')

    def _write_results(self, row, result_data, has_header):
        """Write results to CSV file."""
        pull_status, meta_result = result_data
        
        row_result = {
            'Consumer Loans Data Product': row.group,
            'PCDS Table Details with DB Name': row.pcds_tbl.split('.')[1].lower(),
            'Tables delivered in AWS with DB Name': row.aws_tbl,
            'Status': pull_status.value,
            **meta_result.to_dict()
        }
        
        # Apply column filtering
        row_result['PCDS Unique Columns'] = self._remove_items(
            row_result['PCDS Unique Columns'], 
            self.config.match.drop_cols
        )
        row_result['AWS Unique Columns'] = self._remove_items(
            row_result['AWS Unique Columns'], 
            self.config.match.add_cols
        )
        
        with open(self.config.output.csv.file, 'a+', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=self.config.output.csv.columns)
            if not has_header:
                writer.writeheader()
            writer.writerow(row_result)

    def _remove_items(self, input_str, delete_lst):
        """Remove specific items from a string."""
        pattern = '|'.join(r'\b%s\b;?\s?' % x for x in delete_lst)
        return re.sub(pattern, '', input_str).rstrip('; ')

    def _cleanup(self):
        """Cleanup and upload results."""
        # Save data to pickle
        with open(self.config.output.to_pkl, 'wb') as fp:
            pickle.dump({}, fp)  # Placeholder for data dict
        
        # Upload to S3
        self._upload_to_s3()
        utils.end_run()

    def _upload_to_s3(self):
        """Upload results to S3."""
        s3_root = utils.urljoin(f'{self.config.output.to_s3.run}/', self.config.input.name)
        for root, _, files in os.walk(self.config.output.folder):
            for file in files:
                if file.startswith(self.config.input.step):
                    s3_url = utils.urljoin(f'{s3_root}/', file)
                    local = os.path.join(root, file)
                    utils.s3_upload(local, s3_url)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Conduct Meta Info Analysis')
    parser.add_argument(
        '--config',
        type=Path,
        help='Config file for conducting meta analysis',
        default=r'files\inputs\config_meta_0505.cfg'
    )
    args = parser.parse_args()
    
    runner = MetaAnalysisRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()