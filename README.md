# Web Applications Repository

This repository contains a collection of web applications built using Django and Streamlit frameworks, showcasing various implementations of AI-powered features and basic web development concepts.

## Repository Structure

| Directory | Framework | Type | Python Version | Requirements | Key Features | Data Sources | Use Cases |
|-----------|-----------|------|----------------|--------------|-------------|--------------|-----------|
| [django-basic/](django-basic/) | Django | Data Quality | 3.8+ | [requirements.txt](django-basic/django_data-quality/requirements.txt) | - Dataset validation<br>- Data governance rules<br>- Data profiling<br>- Rule management | CSV, Excel, SQL | Data quality management, Data governance |
| [django-basic-ai/](django-basic-ai/) | Django | AI Integration | 3.8+ | [requirements.txt](django-basic-ai/django_data_ai_app/requirements.txt) | - Chatbot implementations<br>- Data analysis tools<br>- AI processing | Text, Databases | AI-powered data analysis, Chat interfaces |
| [streamlit-basic/](streamlit-basic/) | Streamlit | Basic Apps | 3.8+ | [requirements.txt](streamlit-basic/CRUD_APP/requirements.txt) | - CRUD operations<br>- Mapping features<br>- Data visualization | CSV, JSON | Simple web apps, Location tracking |
| [streamlit-basic-ai/](streamlit-basic-ai/) | Streamlit | AI Apps | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) | - OpenAI chatbots<br>- Document analysis<br>- Meeting assistant | Websites, PDFs, SQL, CSV | AI assistants, Document processing |

## Detailed Breakdown

### Data Processing Capabilities

| Tool | Python Version | Requirements | Input Formats | Processing Features | Output Formats |
|------|---------------|--------------|--------------|---------------------|----------------|
| Data Quality Manager | 3.8+ | [requirements.txt](django-basic/django_data-quality/requirements.txt) | CSV, Excel, SQL | Validation, Profiling, Rules | Reports, Visualizations |
| AI Chatbots | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) | Text, PDF, SQL | NLP, Analysis, Querying | Text responses, Data insights |
| Mapping Tools | 3.8+ | [reqs.txt](streamlit-basic/maps_app/root_app/reqs.txt) | CSV, GeoJSON | Location tracking, Visualization | Interactive maps |
| Document Processors | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/open_ai_chatbot_talk_with_n_pdfs/reqs.txt) | PDF, Text | Analysis, Extraction | Summaries, Insights |

### AI Integration Features

| Application | Python Version | Requirements | AI Capabilities | Integration Points | Use Cases |
|------------|---------------|--------------|----------------|-------------------|-----------|
| Chatbots | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/open_ai_chatbot_talk_with_website/reqs.txt) | Natural Language Processing | Websites, Databases | Customer support, Data querying |
| Document Analysis | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/open_ai_chatbot_talk_with_n_pdfs/reqs.txt) | Text Analysis, Summarization | PDFs, Documents | Content analysis, Information extraction |
| Meeting Assistant | 3.8+ | [reqs.txt](streamlit-basic-ai/meeting_assistant/reqs.txt) | Speech Processing | Audio files | Meeting transcription, Note taking |
| Data Consultant | 3.8+ | [requirements.txt](streamlit-basic-ai/Gen_ai_data_consultant/requirments/requirements.txt) | Data Analysis, Visualization | Various data sources | Data insights, Business intelligence |

### User Interface Components

| Component | Framework | Python Version | Requirements | Features | Customization |
|-----------|-----------|---------------|--------------|----------|---------------|
| Dashboards | Streamlit | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) | Interactive, Real-time | CSS, Layouts |
| Admin Panels | Django | 3.8+ | [requirements.txt](django-basic/django_data-quality/requirements.txt) | CRUD operations, Management | Templates, Forms |
| Maps | Streamlit/Leaflet | 3.8+ | [reqs.txt](streamlit-basic/maps_app/root_app/reqs.txt) | Interactive, Location-based | Markers, Layers |
| Chat Interfaces | Both | 3.8+ | [reqs.txt](streamlit-basic-ai/Streamlit_openai_chatbots/root_app/reqs.txt) | Real-time, Responsive | Styling, Layouts |

## Getting Started

Each application directory contains its own:
- Requirements file (`requirements.txt` or `reqs.txt`)
- Documentation (README files)
- Example data and configurations

## License

This repository is licensed under the terms specified in the individual LICENSE files within each application directory.

## Contributing

Feel free to explore the different applications and contribute to their development. Each application is self-contained and can be run independently.
