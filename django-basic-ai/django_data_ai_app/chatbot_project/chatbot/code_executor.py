import sys
import io
import contextlib
import traceback
import matplotlib.pyplot as plt
import base64
from jupyter_client import KernelManager
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import os
import tempfile
import atexit
import signal
import logging
import threading
import queue
from IPython.core.interactiveshell import InteractiveShell
from io import StringIO, BytesIO
import re

logger = logging.getLogger(__name__)

class REPLExecutor:
    """Interactive REPL for code execution with persistent state"""
    def __init__(self):
        self.shell = InteractiveShell.instance()
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
    def capture_output(self, output_type, value):
        """Capture different types of output"""
        if output_type == 'stream':
            self.output_queue.put(value)
        elif output_type == 'error':
            self.error_queue.put(value)
            
    def execute(self, code):
        """Execute code in the REPL"""
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                # Execute the code
                result = self.shell.run_cell(code)
                
                # Get execution result
                if result.success:
                    if result.result is not None:
                        self.output_queue.put(str(result.result))
                else:
                    if result.error_before_exec:
                        self.error_queue.put(str(result.error_before_exec))
                    if result.error_in_exec:
                        self.error_queue.put(str(result.error_in_exec))
                
            except Exception as e:
                self.error_queue.put(str(e))
        
        # Capture output streams
        stdout_content = stdout.getvalue()
        stderr_content = stderr.getvalue()
        
        if stdout_content:
            self.output_queue.put(stdout_content)
        if stderr_content:
            self.error_queue.put(stderr_content)
            
        # Collect all output
        outputs = []
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get())
        
        errors = []
        while not self.error_queue.empty():
            errors.append(self.error_queue.get())
            
        return {
            'output': '\n'.join(outputs),
            'errors': '\n'.join(errors),
            'success': len(errors) == 0
        }

class CodeExecutor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CodeExecutor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.km = KernelManager()
        self.km.start_kernel()
        self.client = self.km.client()
        self.client.start_channels()
        logger.info("Kernel initialized successfully")
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        
        # Only register signal handlers in the main thread
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGINT, self.signal_handler)
                signal.signal(signal.SIGTERM, self.signal_handler)
            except ValueError:
                logger.warning("Could not register signal handlers (not in main thread)")
    
    def execute_with_data(self, code, data_path=None):
        """Execute code with data file path injected"""
        try:
            modified_code = code
            
            # If there's a data file, inject the path into the code
            if data_path:
                file_extension = os.path.splitext(data_path)[1].lower()
                
                # Determine the correct pandas read function based on file extension
                if file_extension == '.csv':
                    read_command = f"""
df = pd.read_csv('{data_path}',
                 low_memory=False,
                 dtype='object',
                 encoding='utf-8',
                 on_bad_lines='skip')
"""
                elif file_extension in ['.xlsx', '.xls']:
                    read_command = f"""
df = pd.read_excel('{data_path}',
                   dtype='object',
                   engine='openpyxl')
"""
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                # Replace generic dataset references with actual read command
                modified_code = modified_code.replace("pd.read_csv('your_dataset.csv')", read_command.strip())
                modified_code = modified_code.replace('pd.read_csv("your_dataset.csv")', read_command.strip())
                modified_code = modified_code.replace('df = pd.read_csv()', read_command.strip())
                
                # If no read command is present, add the complete setup
                if 'pd.read_' not in modified_code:
                    initial_setup = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data with optimized settings
{read_command}

# Convert numeric columns properly
for column in df.columns:
    try:
        numeric_conversion = pd.to_numeric(df[column], errors='coerce')
        if not numeric_conversion.isna().all():
            df[column] = numeric_conversion
    except:
        continue

# Print available columns for reference
print("Available columns in the dataset:")
for col in df.columns:
    print(f"- {col}")
print("\\n")
"""
                    modified_code = initial_setup + modified_code

                # Replace 'COLUMN_NAME' with the first categorical column if it exists
                if 'COLUMN_NAME' in modified_code:
                    column_setup = """
# Get the first categorical column
categorical_columns = df.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    categorical_column = categorical_columns[0]
    print(f"Using column: {categorical_column}")
else:
    raise ValueError("No categorical columns found in the dataset")
"""
                    modified_code = column_setup + modified_code.replace("'COLUMN_NAME'", "categorical_column").replace('"COLUMN_NAME"', "categorical_column")

            # Capture output
            output_buffer = StringIO()
            figures = []

            # Execute the modified code
            msg_id = self.client.execute(modified_code)
            
            while True:
                try:
                    msg = self.client.get_iopub_msg(timeout=10)
                    msg_type = msg['header']['msg_type']

                    if msg_type == 'stream':
                        output_buffer.write(msg['content']['text'])
                    elif msg_type == 'display_data' or msg_type == 'execute_result':
                        if 'text/plain' in msg['content']['data']:
                            output_buffer.write(msg['content']['data']['text/plain'])
                            output_buffer.write('\n')
                        if 'image/png' in msg['content']['data']:
                            figures.append(msg['content']['data']['image/png'])
                    elif msg_type == 'error':
                        error_msg = '\n'.join(msg['content']['traceback'])
                        logger.error(f"Code execution error: {error_msg}")
                        output_buffer.write(error_msg)
                    
                    # Check if execution is complete
                    if msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                        break

                except Exception as e:
                    error_msg = f"Error during code execution: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    output_buffer.write(error_msg)
                    break

            # Capture any matplotlib figures
            if 'plt' in modified_code:
                try:
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        buf = BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight')
                        plt.close(fig)
                        buf.seek(0)
                        figures.append(base64.b64encode(buf.read()).decode('utf-8'))
                except Exception as e:
                    error_msg = f"Error capturing figures: {str(e)}"
                    logger.error(error_msg)
                    output_buffer.write(f"\n{error_msg}")

            return {
                'output': output_buffer.getvalue(),
                'figures': figures
            }

        except Exception as e:
            error_msg = f"Error in execute_with_data: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                'output': error_msg,
                'figures': []
            }
    
    def cleanup(self):
        """Clean up kernel resources"""
        try:
            self.client.stop_channels()
            self.km.shutdown_kernel()
        except:
            pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.cleanup()
        sys.exit(0)
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup() 