# Audio Transcription and Insight Generation App

This project is a Streamlit web application that allows users to upload audio files, transcribe them using AssemblyAI, generate insights using LangChain, and save the results to a structured Word document.

## Features

- **File Upload**: Upload audio files in formats such as mp3, wav, mp4, and m4a.
- **Transcription**: Automatically transcribe the uploaded audio files using AssemblyAI.
- **Summarization**: Optionally summarize long transcripts.
- **Insight Generation**: Generate insights based on the transcription using LangChain.
- **Word Document**: Save the transcription and insights to a Word document with a meeting-style format.

## Benefits

- **Efficiency**: Save time by automating the transcription and summarization of audio files.
- **Insightful**: Gain valuable insights from transcriptions through automated analysis.
- **Convenience**: Easily save and share structured meeting notes in a Word document format.
- **Accessibility**: Supports multiple audio file formats for versatile usage.

## Pain and Gain

- **Pain**: Manually transcribing and summarizing audio recordings is time-consuming and prone to errors.
- **Gain**: Automate the transcription process, generate actionable insights, and save the results in a well-structured format, enhancing productivity and accuracy.

## Notable Functions

- **Transcription**: Utilizes AssemblyAI for high-quality transcription of audio files.
- **Summarization**: Includes a placeholder function for summarization, which can be customized with actual logic.
- **Insight Generation**: Leverages LangChain to analyze the transcription and provide actionable insights.
- **Document Creation**: Uses `python-docx` to create a formatted Word document with the transcription and insights.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- AssemblyAI API key
- LangChain
- python-docx

## Installation

1. Clone the repository or download the script.

2. Install the required packages:
    ```sh
    pip install streamlit assemblyai langchain python-docx
    ```

3. Set up your AssemblyAI API key and OpenAI API key in the script:
    ```python
    aai.settings.api_key = "your_assemblyai_api_key"
    OPENAI_API_KEY = "your_openai_api_key"
    ```