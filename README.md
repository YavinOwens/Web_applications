# Web Applications Repository

A collection of web applications built using Django and Streamlit frameworks, featuring AI-powered chatbots, data management tools, and interactive visualizations.

## Repository Structure

| Directory | Framework | Type | Key Features | Data Sources | Use Cases | Python Version | Requirements |
|-----------|-----------|------|-------------|--------------|-----------|----------------|--------------|
| django-basic/ | Django | Basic | Data quality management, CRUD operations | CSV, SQLite | Data validation, Quality control | 3.8+ | [requirements.txt](django-basic/django_data-quality/requirements.txt) |
| django-basic-ai/ | Django | AI | Chatbots, Data analysis | PDFs, Databases | Document analysis, Customer support | 3.8+ | [requirements.txt](django-basic-ai/django_ai_chatbot/requirements.txt) |
| streamlit-basic/ | Streamlit | Basic | CRUD, Mapping | CSV, GeoJSON | Data visualization, Location tracking | 3.8+ | [requirements.txt](streamlit-basic/streamlit_CRUD/requirements.txt) |
| streamlit-basic-ai/ | Streamlit | AI | OpenAI chatbots, Meeting assistant | PDFs, Databases, Websites | Document analysis, Meeting summaries | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) |

## Detailed Breakdown

### Data Processing Capabilities

| Input Format | Processing Features | Output Format | Python Version | Requirements |
|--------------|---------------------|---------------|----------------|--------------|
| CSV | Validation, Cleaning, Transformation | Cleaned CSV, Reports | 3.8+ | [requirements.txt](django-basic/django_data-quality/requirements.txt) |
| PDF | Text extraction, Analysis | Summaries, Insights | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) |
| SQLite/PostgreSQL | Querying, Data manipulation | Processed data, Analytics | 3.8+ | [requirements.txt](django-basic-ai/django_ai_chatbot/requirements.txt) |

### AI Integration Features

| AI Capabilities | Integration Points | Customization | Python Version | Requirements |
|----------------|-------------------|---------------|----------------|--------------|
| OpenAI Chatbots | Web interfaces, APIs | Prompt engineering, Model selection | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) |
| Document Analysis | PDF processing, Text extraction | Analysis parameters, Output format | 3.8+ | [requirements.txt](django-basic-ai/django_ai_chatbot/requirements.txt) |
| Meeting Assistant | Audio processing, Text analysis | Summary length, Key points | 3.8+ | [requirements.txt](streamlit-basic-ai/meeting_assistant/requirements.txt) |

### User Interface Components

| Component Type | Features | Customization | Python Version | Requirements |
|---------------|----------|---------------|----------------|--------------|
| Streamlit Apps | Interactive widgets, Data visualization | Theme, Layout | 3.8+ | [requirements.txt](streamlit-basic/streamlit_CRUD/requirements.txt) |
| Django Admin | CRUD operations, User management | Permissions, Views | 3.8+ | [requirements.txt](django-basic/django_data-quality/requirements.txt) |
| Mapping Tools | Location tracking, Geo-visualization | Map styles, Markers | 3.8+ | [requirements.txt](streamlit-basic/streamlit_CRUD/requirements.txt) |

## Getting Started

Each application directory contains its own setup instructions and requirements. Check the respective README files for detailed information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Feel free to explore the applications and contribute to their development!
