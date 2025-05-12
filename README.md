# BNS Integration for VakeelGPT

This integration adds Bharatiya Nyaya Sanhita (BNS) data to the VakeelGPT application, enhancing its ability to answer BNS-related queries.

## Overview

The integration consists of the following components:

1. **BNS Integration Module**: Uses the LLM to generate accurate responses to BNS queries
2. **App Patch**: Integrates the BNS functionality into the VakeelGPT application
3. **Command Interface**: Adds a `/bns` command to the chat interface

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Together API key
- LangChain library

### Installation

1. Ensure all required packages are installed:

```bash
pip install langchain langchain-together
```

2. Set up environment variables:

```bash
export TOGETHER_API_KEY="your_together_api_key"
```

### Running the Integration

#### Step 1: Apply the Patch to app.py

Apply the patch to integrate the BNS data into the VakeelGPT application:

```bash
python app_bns_patch.py
```

#### Step 2: Run the VakeelGPT Application

Run the VakeelGPT application:

```bash
streamlit run app.py
```

## Usage

Once the integration is set up, you can query BNS data in the VakeelGPT application using the `/bns` command:

```
/bns What is the punishment for murder under BNS?
```

## Troubleshooting

### Common Issues

1. **LLM Connection Issues**:
   - Check that your Together API key is correct
   - Ensure you have internet connectivity

2. **Integration Issues**:
   - If the patch fails, a backup of app.py is created as app.py.bak
   - Check the logs for specific errors

### Logs

Log files are created in the current directory.

## How It Works

The BNS integration uses the LLM to generate accurate responses to BNS queries. The LLM is prompted with a specialized template that instructs it to provide detailed information about the Bharatiya Nyaya Sanhita, including specific section numbers, penalties, and other relevant details.

The integration adds a `/bns` command to the VakeelGPT chat interface, which allows users to easily query BNS data. When a user enters a query with the `/bns` prefix, the application extracts the query and sends it to the BNS integration module, which then generates a response using the LLM.

## Example Queries

Here are some example queries you can try:

- `/bns What is the punishment for murder under BNS?`
- `/bns How does BNS define sedition?`
- `/bns What are the key differences between IPC and BNS?`
- `/bns What is the punishment for rape under BNS?`
- `/bns How does BNS handle cyber crimes?`
