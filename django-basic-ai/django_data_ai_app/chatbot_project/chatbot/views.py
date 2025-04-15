import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.conf import settings
from openai import OpenAI
import os
import pandas as pd
import numpy as np
import seaborn as sns
import PyPDF2
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import logging
from .code_executor import CodeExecutor
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# Initialize code executor
code_executor = CodeExecutor()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return str(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)

def convert_to_serializable(obj):
    """Convert numpy and pandas objects to JSON serializable types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.dtype):
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def generate_insights(df):
    """Generate automatic insights from the dataframe"""
    insights = []
    figures = []
    
    try:
        # Basic statistics
        insights.append({
            'title': 'Dataset Overview',
            'content': f"Total rows: {len(df)}\nTotal columns: {len(df.columns)}\n"
                      f"Columns: {', '.join(df.columns)}"
        })

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Summary statistics
            insights.append({
                'title': 'Numeric Columns Statistics',
                'content': df[numeric_cols].describe().to_string()
            })

            # Correlation heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
                
                # Save plot to bytes
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                figures.append({
                    'title': 'Correlation Heatmap',
                    'image': base64.b64encode(buf.read()).decode('utf-8')
                })

            # Distribution plots for numeric columns (up to 3)
            for col in numeric_cols[:3]:
                plt.figure(figsize=(8, 6))
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                figures.append({
                    'title': f'Distribution of {col}',
                    'image': base64.b64encode(buf.read()).decode('utf-8')
                })

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:3]:  # Analyze up to 3 categorical columns
                value_counts = df[col].value_counts()
                insights.append({
                    'title': f'Category Distribution: {col}',
                    'content': value_counts.to_string()
                })

                # Bar plot for categorical columns
                plt.figure(figsize=(10, 6))
                sns.barplot(x=value_counts.index[:10], y=value_counts.values[:10])
                plt.title(f'Top 10 Categories in {col}')
                plt.xticks(rotation=45)
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                figures.append({
                    'title': f'Top Categories in {col}',
                    'image': base64.b64encode(buf.read()).decode('utf-8')
                })

        # Missing values analysis
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            insights.append({
                'title': 'Missing Values Analysis',
                'content': missing_data[missing_data > 0].to_string()
            })

    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        insights.append({
            'title': 'Error',
            'content': f"Error generating insights: {str(e)}"
        })

    return insights, figures

def process_file(uploaded_file):
    """Process uploaded files with enhanced data analysis"""
    try:
        # Save the file
        filename = default_storage.save(
            os.path.join('uploads', uploaded_file.name),
            ContentFile(uploaded_file.read())
        )
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Process based on file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.csv':
            # Read CSV with optimized settings
            df = pd.read_csv(file_path, 
                           low_memory=False,  # Prevent mixed type inference warning
                           dtype='object',    # Initially read all columns as strings
                           encoding='utf-8',  # Specify encoding
                           on_bad_lines='skip'  # Skip problematic lines
            )
            
            # Convert numeric columns properly
            for column in df.columns:
                try:
                    # Try to convert to numeric, keeping NaN values
                    numeric_conversion = pd.to_numeric(df[column], errors='coerce')
                    # If the conversion resulted in some valid numbers, update the column
                    if not numeric_conversion.isna().all():
                        df[column] = numeric_conversion
                except:
                    continue
            
            insights, figures = generate_insights(df)
            sample_data = df.head(3).to_string()
            
            # Convert dtypes to strings and handle other non-serializable types
            column_types = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            has_nulls = {str(k): bool(v) for k, v in df.isnull().any().to_dict().items()}
            
            result = {
                'summary': f"CSV File Summary:\n{df.describe(include='all').to_string()}\n\nColumns: {', '.join(df.columns)}\n\nSample Data:\n{sample_data}",
                'file_path': file_path,
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns)),
                'column_types': column_types,
                'has_nulls': has_nulls,
                'insights': convert_to_serializable(insights),
                'figures': figures
            }
            return convert_to_serializable(result)
            
        elif file_extension in ['.xlsx', '.xls']:
            # Read Excel with optimized settings
            df = pd.read_excel(file_path, 
                             dtype='object',  # Initially read all columns as strings
                             engine='openpyxl'
            )
            
            # Convert numeric columns properly
            for column in df.columns:
                try:
                    numeric_conversion = pd.to_numeric(df[column], errors='coerce')
                    if not numeric_conversion.isna().all():
                        df[column] = numeric_conversion
                except:
                    continue
            
            insights, figures = generate_insights(df)
            sample_data = df.head(3).to_string()
            
            # Convert dtypes to strings and handle other non-serializable types
            column_types = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            has_nulls = {str(k): bool(v) for k, v in df.isnull().any().to_dict().items()}
            
            result = {
                'summary': f"Excel File Summary:\n{df.describe(include='all').to_string()}\n\nColumns: {', '.join(df.columns)}\n\nSample Data:\n{sample_data}",
                'file_path': file_path,
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns)),
                'column_types': column_types,
                'has_nulls': has_nulls,
                'insights': convert_to_serializable(insights),
                'figures': figures
            }
            return convert_to_serializable(result)
            
        elif file_extension == '.pdf':
            # PDF processing remains unchanged
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Generate a concise summary of the following PDF text:"},
                        {"role": "user", "content": text[:4000]}
                    ],
                    max_tokens=500
                )
                summary = response.choices[0].message.content
                
                result = {
                    'summary': f"PDF Summary:\n{summary}",
                    'file_path': file_path,
                    'total_pages': len(pdf_reader.pages),
                    'insights': [{'title': 'PDF Analysis', 'content': summary}],
                    'figures': []
                }
                return convert_to_serializable(result)
                
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {'error': f"Error processing file: {str(e)}"}

def chatbot_view(request):
    """Main chatbot interaction view with enhanced data analysis"""
    if not os.getenv('OPENAI_API_KEY'):
        return render(request, 'chatbot/chat.html', {'error': 'OpenAI API key not found. Please set it in your .env file.'})

    # Initialize session data if not exists
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []
    if 'current_file' not in request.session:
        request.session['current_file'] = None
    if 'file_summary' not in request.session:
        request.session['file_summary'] = None
    if 'file_metadata' not in request.session:
        request.session['file_metadata'] = None
    if 'insights' not in request.session:
        request.session['insights'] = None
    if 'figures' not in request.session:
        request.session['figures'] = None

    context = {
        'chat_history': request.session['chat_history'],
        'file_content': request.session['file_summary'],
        'file_metadata': request.session['file_metadata'],
        'insights': request.session.get('insights', []),
        'figures': request.session.get('figures', [])
    }
    
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        uploaded_file = request.FILES.get('file_upload')
        
        try:
            # Process file upload if present
            if uploaded_file:
                file_content = process_file(uploaded_file)
                if 'error' in file_content:
                    context['error'] = file_content['error']
                else:
                    request.session['file_summary'] = file_content['summary']
                    request.session['current_file'] = file_content['file_path']
                    # Store insights and figures
                    request.session['insights'] = file_content.get('insights', [])
                    request.session['figures'] = file_content.get('figures', [])
                    # Store additional metadata
                    metadata = {k: v for k, v in file_content.items() 
                              if k not in ['summary', 'file_path', 'insights', 'figures']}
                    request.session['file_metadata'] = metadata
                    
                    context.update({
                        'file_content': file_content['summary'],
                        'file_metadata': metadata,
                        'insights': file_content.get('insights', []),
                        'figures': file_content.get('figures', [])
                    })
            
            if user_input:
                # Prepare system message with enhanced context awareness
                system_message = (
                    "You are a helpful assistant with data analysis capabilities. "
                    "When asked about data analysis, provide Python code examples using pandas, "
                    "matplotlib, or seaborn. Format code blocks with triple backticks. "
                    "Be specific in your analysis and reference the actual data columns and values when possible."
                )
                
                # Add detailed context about available data
                if request.session['current_file']:
                    metadata = request.session.get('file_metadata', {})
                    system_message += (
                        f"\nA dataset has been uploaded and is available as 'df'. "
                        f"The data summary is: {request.session['file_summary']}"
                    )
                    if metadata:
                        system_message += (
                            f"\nAdditional dataset information:"
                            f"\n- Total rows: {metadata.get('total_rows', 'N/A')}"
                            f"\n- Total columns: {metadata.get('total_columns', 'N/A')}"
                            f"\n- Column types: {metadata.get('column_types', {})}"
                            f"\n- Columns with null values: {metadata.get('has_nulls', {})}"
                        )
                
                # Send to OpenAI API with enhanced context
                messages = [
                    {"role": "system", "content": system_message}
                ]
                
                # Add chat history for context
                for msg in request.session['chat_history'][-5:]:  # Last 5 messages for context
                    messages.append({"role": msg['role'], "content": msg['content']})
                
                # Add current user message
                messages.append({"role": "user", "content": user_input})
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                chatbot_response = response.choices[0].message.content
                
                # Process any code blocks in the response
                if request.session.get('current_file'):
                    code_results = process_code_blocks(chatbot_response, request.session['current_file'])
                    context['code_results'] = code_results
                
                # Update chat history
                request.session['chat_history'].append({
                    'role': 'user',
                    'content': user_input
                })
                request.session['chat_history'].append({
                    'role': 'assistant',
                    'content': chatbot_response
                })
                request.session.modified = True
                
                context.update({
                    'user_input': user_input,
                    'chatbot_response': chatbot_response,
                    'chat_history': request.session['chat_history']
                })
            
        except Exception as e:
            logger.error(f"Error in chatbot_view: {str(e)}")
            context['error'] = f"An error occurred: {str(e)}"
    
    return render(request, 'chatbot/chat.html', context)

def extract_code_blocks(text):
    """Extract code blocks from markdown-style text"""
    code_blocks = []
    import re
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        code_blocks.append(match.group(1).strip())
    
    return code_blocks

def process_code_blocks(text, data_path=None):
    """Process and execute code blocks in the text"""
    code_blocks = extract_code_blocks(text)
    results = []
    
    for code in code_blocks:
        try:
            # Pass the actual file path to the code executor
            if data_path:
                result = code_executor.execute_with_data(code, data_path)
            else:
                result = code_executor.execute_with_data(code)
                
            if isinstance(result, dict):
                results.append({
                    'code': code,
                    'output': result.get('output', ''),
                    'figures': result.get('figures', [])
                })
            else:
                results.append({
                    'code': code,
                    'output': str(result),
                    'figures': []
                })
        except Exception as e:
            logger.error(f"Error executing code block: {str(e)}")
            results.append({
                'code': code,
                'output': f"Error: {str(e)}",
                'figures': []
            })
    
    return results

@require_http_methods(["POST"])
def clear_chat(request):
    """Clear chat history and data context from session"""
    try:
        # Clear session data
        request.session['chat_history'] = []
        request.session['current_file'] = None
        request.session['file_summary'] = None
        request.session['file_metadata'] = None
        request.session['insights'] = None
        request.session['figures'] = None
        request.session.modified = True
        
        # Clean up uploaded files
        if request.session.get('current_file'):
            try:
                os.remove(request.session['current_file'])
            except OSError:
                pass
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)})
