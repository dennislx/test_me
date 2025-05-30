import xlwings as xw
import pandas as pd
import os

def open_or_create_workbook(filename):
    """
    Open existing workbook or create new one if it doesn't exist
    """
    try:
        # Try to open existing workbook
        if os.path.exists(filename):
            wb = xw.Book(filename)
            print(f"Opened existing workbook: {filename}")
        else:
            # Create new workbook if file doesn't exist
            wb = xw.Book()
            wb.save(filename)
            print(f"Created new workbook: {filename}")
        return wb
    except Exception as e:
        print(f"Error with workbook: {e}")
        # Fallback: create new workbook
        wb = xw.Book()
        wb.save(filename)
        return wb

def get_or_create_worksheet(wb, sheet_name):
    """
    Get existing worksheet or create new one if it doesn't exist
    """
    try:
        # Try to get existing worksheet
        ws = wb.sheets[sheet_name]
        print(f"Found existing worksheet: {sheet_name}")
        # Clear existing content
        ws.clear()
        print(f"Cleared existing content in: {sheet_name}")
        return ws
    except:
        # Create new worksheet if it doesn't exist
        ws = wb.sheets.add(sheet_name)
        print(f"Created new worksheet: {sheet_name}")
        return ws

def write_to_excel_smart(df, filename='output.xlsx', sheet_name='Sheet1', clear_existing=True):
    """
    Smart function to write DataFrame to Excel:
    - Opens existing workbook or creates new
    - Uses existing worksheet or creates new
    - Optionally clears existing content
    """
    
    # Open or create workbook
    wb = open_or_create_workbook(filename)
    
    # Get or create worksheet
    ws = get_or_create_worksheet(wb, sheet_name)
    
    # Write headers
    headers = list(df.columns)
    ws.range('A1').value = headers
    
    # Write data
    if not df.empty:
        ws.range('A2').value = df.values
    
    # Format headers
    header_range = ws.range(f'A1:{chr(65 + len(headers) - 1)}1')
    header_range.font.bold = True
    
    # Auto-fit columns
    ws.autofit()
    
    # Save workbook
    wb.save()
    print(f"Data written to {filename}, sheet: {sheet_name}")
    
    return wb, ws

def advanced_workbook_management(filename='analysis.xlsx'):
    """
    Advanced example showing different scenarios
    """
    
    # Scenario 1: Multiple sheets in same workbook
    wb = open_or_create_workbook(filename)
    
    # Create sample data
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
    
    # Write to different sheets
    ws1 = get_or_create_worksheet(wb, 'Data1')
    ws1.range('A1').value = df1
    
    ws2 = get_or_create_worksheet(wb, 'Data2')
    ws2.range('A1').value = df2
    
    # Scenario 2: Check if specific sheet exists
    sheet_names = [sheet.name for sheet in wb.sheets]
    print(f"Existing sheets: {sheet_names}")
    
    if 'Summary' in sheet_names:
        print("Summary sheet exists")
        ws_summary = wb.sheets['Summary']
    else:
        print("Creating Summary sheet")
        ws_summary = wb.sheets.add('Summary')
    
    # Scenario 3: Conditional clearing
    if 'Data1' in sheet_names:
        response = input("Data1 sheet exists. Clear it? (y/n): ")
        if response.lower() == 'y':
            wb.sheets['Data1'].clear()
    
    wb.save()
    return wb

def safe_excel_operation(filename, sheet_name, df, operation='overwrite'):
    """
    Safe operation with error handling
    
    operation options:
    - 'overwrite': Clear existing content and write new
    - 'append': Add to existing content (if any)
    - 'new_only': Only write if sheet doesn't exist
    """
    
    try:
        # Open or create workbook
        wb = open_or_create_workbook(filename)
        
        # Handle different operations
        if operation == 'overwrite':
            ws = get_or_create_worksheet(wb, sheet_name)
            # Content already cleared in get_or_create_worksheet
            
        elif operation == 'append':
            try:
                ws = wb.sheets[sheet_name]
                # Find the last row with data
                last_row = ws.range('A1').end('down').row
                start_row = last_row + 1 if last_row > 1 else 2
                print(f"Appending data starting from row {start_row}")
            except:
                ws = wb.sheets.add(sheet_name)
                start_row = 2
                # Write headers for new sheet
                ws.range('A1').value = list(df.columns)
                ws.range('A1').expand('right').font.bold = True
            
        elif operation == 'new_only':
            sheet_names = [sheet.name for sheet in wb.sheets]
            if sheet_name in sheet_names:
                print(f"Sheet '{sheet_name}' already exists. Skipping.")
                wb.close()
                return None
            else:
                ws = wb.sheets.add(sheet_name)
                start_row = 2
                # Write headers
                ws.range('A1').value = list(df.columns)
                ws.range('A1').expand('right').font.bold = True
        
        # Write data
        if operation == 'overwrite':
            # Write headers and data
            ws.range('A1').value = list(df.columns)
            ws.range('A1').expand('right').font.bold = True
            if not df.empty:
                ws.range('A2').value = df.values
        else:
            # Write only data (headers already handled)
            if not df.empty:
                ws.range(f'A{start_row}').value = df.values
        
        # Auto-fit and save
        ws.autofit()
        wb.save()
        
        print(f"Operation '{operation}' completed successfully")
        return wb, ws
        
    except Exception as e:
        print(f"Error during Excel operation: {e}")
        return None

def batch_excel_operations(data_dict, filename='batch_output.xlsx'):
    """
    Write multiple DataFrames to different sheets in one workbook
    
    data_dict: {'sheet_name': dataframe, ...}
    """
    
    wb = open_or_create_workbook(filename)
    
    for sheet_name, df in data_dict.items():
        print(f"Processing sheet: {sheet_name}")
        ws = get_or_create_worksheet(wb, sheet_name)
        
        # Write data
        ws.range('A1').value = list(df.columns)
        if not df.empty:
            ws.range('A2').value = df.values
        
        # Format
        ws.range('A1').expand('right').font.bold = True
        ws.autofit()
    
    wb.save()
    print(f"Batch operation completed: {filename}")
    return wb
    
    
def write_to_excel_with_merged_cells(df, filename='table_issues_expanded.xlsx'):
    # First create the expanded dataframe normally
    expanded_df = expand_table_issues(df)  # Your original function
    
    wb = xw.Book()
    ws = wb.sheets[0]
    
    # Write headers
    headers = ['Table Name', 'Issue', 'Variable']
    ws.range('A1:C1').value = headers
    ws.range('A1:C1').font.bold = True
    
    # Write data
    current_row = 2
    current_table = None
    merge_start = 2
    
    for _, row in expanded_df.iterrows():
        ws.range(f'A{current_row}').value = row['table_name']
        ws.range(f'B{current_row}').value = row['issue']
        ws.range(f'C{current_row}').value = row['variable']
        
        # Check if we need to merge cells for table name
        if current_table != row['table_name']:
            if current_table is not None and current_row > merge_start:
                # Merge previous table's cells
                ws.range(f'A{merge_start}:A{current_row-1}').merge()
            current_table = row['table_name']
            merge_start = current_row
        
        current_row += 1
    
    # Merge the last table's cells
    if current_row > merge_start:
        ws.range(f'A{merge_start}:A{current_row-1}').merge()
    
    # Format
    ws.autofit()
    ws.range('A:A').api.VerticalAlignment = -4108  # xlCenter
    
    wb.save(filename)
    return wb

# Example usage
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'table_name': ['table1', 'table2'],
        'issues': ['issue1;issue2', 'issue3;issue4'],
        'variables': ['var1;var2', 'varA;varB']
    })
    
    # Example 1: Basic smart writing
    wb, ws = write_to_excel_smart(df, 'my_analysis.xlsx', 'Raw_Data')
    
    # Example 2: Safe operations
    safe_excel_operation('my_analysis.xlsx', 'Processed_Data', df, 'overwrite')
    safe_excel_operation('my_analysis.xlsx', 'Backup_Data', df, 'new_only')
    
    # Example 3: Batch operations
    data_dict = {
        'Sheet1': df,
        'Sheet2': df.head(1),
        'Summary': pd.DataFrame({'Total': [len(df)]})
    }
    batch_excel_operations(data_dict, 'batch_example.xlsx')
    
    # Don't forget to close workbooks
    # wb.close()  # Only if you want to close Excel