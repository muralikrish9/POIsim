# Jailbreak Detection System

A comprehensive AI safety system designed to detect and prevent harmful content generation, jailbreak attempts, and unethical AI interactions. This system helps ensure AI responses remain within ethical boundaries by analyzing prompts for potential risks and harmful content.

## What is a Jailbreak?

A "jailbreak" in AI terms refers to attempts to bypass or override an AI system's safety measures and ethical guidelines. This system helps detect such attempts to maintain safe and responsible AI interactions.

## Features

- **Prompt Analysis and Classification**: Analyzes user inputs to identify their nature and potential risks
- **Toxicity Detection**: Measures harmful, offensive, or inappropriate content
- **Sentiment Analysis**: Evaluates the emotional tone and intent of the text
- **Context-Aware Detection**: Considers conversation history to better understand context
- **Conversation History Tracking**: Maintains a record of interactions for better analysis
- **Rich Console Output**: Provides detailed, color-coded analysis results

## Detailed Setup Instructions

### Prerequisites

Before you begin, make sure you have:
- **Python 3.8 or higher**: Download from [python.org](https://www.python.org/downloads/)
- **pip**: Python's package installer (comes with Python)
- **Git** (optional): For version control and easy updates

### Step-by-Step Installation Guide

1. **Get the Code**:
   ```bash
   # If using Git:
   git clone https://github.com/yourusername/jailbreak-detection.git
   cd jailbreak-detection
   
   # Or download the ZIP file and extract it
   ```

2. **Set Up Python Environment**:
   ```bash
   # Create a virtual environment (like a clean room for Python packages)
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   
   # You'll know it's activated when you see (venv) at the start of your command line
   ```

3. **Install Required Packages**:
   ```bash
   # This will install all necessary Python packages
   pip install -r requirements.txt
   ```

4. **Set Up Your API Key**:
   - Create a new file named `.env` in the project folder
   - Add your Google API key (you'll need to get this from Google Cloud Console):
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   - Save the file

5. **Download Language Models**:
   ```bash
   # This downloads the English language model needed for text analysis
   python -m spacy download en_core_web_sm
   ```

### Running the System

1. **Start the Program**:
   ```bash
   python test_jailbreak.py
   ```

2. **Using the System**:
   - Type or paste your text when prompted
   - The system will analyze it and show detailed results
   - Type 'quit' to exit the program

3. **Understanding the Results**:
   - The system will show:
     - Jailbreak probability (how likely it is to be harmful)
     - Content classification (what type of content it is)
     - Toxicity analysis (how harmful the content might be)
     - Risk factors (specific concerns identified)

## Project Structure Explained

- `classifier/`: Contains the core detection algorithms and analysis tools
  - `jailbreak_detector.py`: Main detection logic
  - Other supporting files for analysis
- `test_jailbreak.py`: The main program you run to test prompts
- `requirements.txt`: Lists all Python packages needed
- `.env`: Your configuration file (create this yourself)

## Common Issues and Solutions

1. **Python Not Found**:
   - Make sure Python is installed and added to your PATH
   - Try `python --version` to check installation

2. **Package Installation Errors**:
   - Make sure your virtual environment is activated
   - Try `pip install --upgrade pip` first
   - Check your internet connection

3. **API Key Issues**:
   - Ensure your `.env` file is in the correct location
   - Verify your Google API key is valid and has proper permissions

4. **Model Download Problems**:
   - Check your internet connection
   - Ensure you have enough disk space
   - Try running the download command again

## System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **Internet Connection**: Required for API calls and model downloads
- **GPU** (optional): For faster processing, NVIDIA GPU with CUDA support

## Getting Help

If you encounter any issues:
1. Check the error message carefully
2. Look for similar issues in the GitHub repository
3. Create a new issue on GitHub with:
   - What you were trying to do
   - The exact error message
   - Your system information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google AI for providing the Gemini API
- Hugging Face for transformer models
- The open-source community for various tools and libraries 