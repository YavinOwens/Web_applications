# Django AI Data Analysis Chatbot

A sophisticated Django-based AI chatbot application that combines natural language processing with data analysis capabilities. This application allows users to upload data files, analyze them, and interact with an AI assistant that can provide insights and visualizations based on the uploaded data.

## Features

### 1. AI Chatbot Interface
- Interactive chat interface with real-time responses
- Context-aware conversations
- Support for code execution and data analysis queries
- Markdown formatting support for responses

### 2. Data Analysis Capabilities
- File upload support for various formats (CSV, Excel, PDF)
- Automated data profiling and analysis
- Interactive data visualizations
- Statistical insights generation
- Column-wise data type detection and analysis

### 3. Data Visualization
- Dynamic chart generation
- Support for multiple visualization types:
  - Bar charts
  - Line plots
  - Scatter plots
  - Histograms
  - Correlation matrices
- Responsive and interactive visualization display

### 4. RAG (Retrieval-Augmented Generation)
- Context-aware responses based on uploaded documents
- Intelligent document processing
- Semantic search capabilities
- Source attribution for responses

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/YavinOwens/django_data_ai_app.git
cd django_data_ai_app
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env`
- Add your OpenAI API key and other required variables

5. Run database migrations:
```bash
python manage.py migrate
```

6. Start the development server:
```bash
python manage.py runserver
```

7. Access the application at `http://127.0.0.1:8000`

## Usage

1. **File Upload**
   - Use the file upload form in the left panel
   - Supported formats: CSV, Excel, PDF
   - Files are automatically processed upon upload

2. **Data Analysis**
   - View automatic data profiling in the Data Analysis Dashboard
   - Explore column statistics and data quality metrics
   - Access interactive visualizations

3. **Chat Interface**
   - Ask questions about your data
   - Request specific analyses or visualizations
   - Get AI-powered insights and explanations

4. **Visualizations**
   - View generated visualizations in the right panel
   - Interact with charts for detailed information
   - Request custom visualizations through chat

## Project Structure

```
django_chatbot/
├── chatbot_project/
│   ├── chatbot/
│   │   ├── templates/
│   │   ├── static/
│   │   ├── code_executor.py
│   │   ├── rag_utils.py
│   │   └── views.py
│   ├── media/
│   └── manage.py
└── requirements.txt
```

## Dependencies

- Django 4.2.17
- OpenAI API
- Pandas
- Matplotlib
- NLTK
- Other requirements listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
