import jupyter_client
import queue
import json
import os

class JupyterKernelManager:
    def __init__(self):
        # Set Jupyter runtime directory
        runtime_dir = os.path.join(os.path.expanduser('~'), '.local', 'share', 'jupyter', 'runtime')
        os.makedirs(runtime_dir, exist_ok=True)
        
        # Initialize kernel with IPython kernel
        self.manager = jupyter_client.KernelManager(kernel_name='python3')
        self.manager.start_kernel()
        
        # Get the client and start channels
        self.client = self.manager.client()
        self.client.start_channels()
        
        # Wait for kernel to be ready
        try:
            self.client.wait_for_ready(timeout=30)
        except RuntimeError:
            self.shutdown_kernel()
            raise Exception("Kernel failed to start")

    def execute_code(self, code):
        msg_id = self.client.execute(code)
        
        # Collect all output
        outputs = []
        timeout_counter = 0
        max_timeouts = 5  # Maximum number of consecutive timeouts
        
        while timeout_counter < max_timeouts:
            try:
                # Get messages from different channels
                msg = self.client.get_iopub_msg(timeout=2)
                
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue
                    
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                if msg_type == 'status' and content['execution_state'] == 'idle':
                    break
                
                if msg_type == 'stream':
                    outputs.append({
                        'type': 'stream',
                        'text': content.get('text', '')
                    })
                elif msg_type == 'display_data':
                    outputs.append({
                        'type': 'display_data',
                        'data': content.get('data', {})
                    })
                elif msg_type == 'execute_result':
                    outputs.append({
                        'type': 'execute_result',
                        'data': content.get('data', {})
                    })
                elif msg_type == 'error':
                    outputs.append({
                        'type': 'error',
                        'ename': content.get('ename', 'Unknown error'),
                        'evalue': content.get('evalue', 'An error occurred'),
                        'traceback': content.get('traceback', [])
                    })
                    break
                
                timeout_counter = 0  # Reset counter on successful message
                    
            except queue.Empty:
                timeout_counter += 1
                continue
        
        try:
            # Get the execution reply
            reply = self.client.get_shell_msg(timeout=10)
            status = reply['content'].get('status', 'error')
        except queue.Empty:
            status = 'error'
            outputs.append({
                'type': 'error',
                'ename': 'Timeout',
                'evalue': 'Kernel execution timed out',
                'traceback': []
            })
        
        return {
            'status': status,
            'outputs': outputs
        }

    def shutdown_kernel(self):
        try:
            self.client.stop_channels()
            self.manager.shutdown_kernel()
        except Exception as e:
            print(f"Error shutting down kernel: {str(e)}")  # Log the error instead of silently passing 