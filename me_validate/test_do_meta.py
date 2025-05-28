"""
Unit tests for the restructured meta analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the classes from the restructured module
# (These would be imported from the actual module in practice)
from do_meta import (
    DataTypeMapper, MetaResult, PullStatus, MetaAnalyzer, 
    DatabaseConnector, MetaAnalysisRunner
)
import utils_type as ut


class TestDataTypeMappers:
    """Test cases for DataTypeMapper class."""
    
    def test_number_mapping_success(self):
        """Test successful NUMBER type mapping."""
        assert DataTypeMapper.map_pcds_to_aws('NUMBER', 'double') == True
    
    def test_number_mapping_failure(self):
        """Test failed NUMBER type mapping."""
        assert DataTypeMapper.map_pcds_to_aws('NUMBER', 'varchar') == False
    
    def test_number_with_precision_mapping(self):
        """Test NUMBER with precision mapping."""
        assert DataTypeMapper.map_pcds_to_aws('NUMBER(10,2)', 'decimal(12,2)') == True
        assert DataTypeMapper.map_pcds_to_aws('NUMBER(10,2)', 'decimal(12,3)') == False
    
    def test_varchar2_mapping(self):
        """Test VARCHAR2 type mapping."""
        assert DataTypeMapper.map_pcds_to_aws('VARCHAR2(100)', 'varchar(100)') == True
        assert DataTypeMapper.map_pcds_to_aws('VARCHAR2(100)', 'varchar(200)') == False
    
    def test_char_mapping(self):
        """Test CHAR type mapping."""
        assert DataTypeMapper.map_pcds_to_aws('CHAR(1)', 'char(1)') == True
        assert DataTypeMapper.map_pcds_to_aws('CHAR(10)', 'varchar(10)') == False
    
    def test_date_mapping(self):
        """Test DATE type mapping."""
        assert DataTypeMapper.map_pcds_to_aws('DATE', 'date') == True
        assert DataTypeMapper.map_pcds_to_aws('DATE', 'timestamp') == True