# Husky Navigator

Husky Navigator is an AI-powered assistant for Northeastern University's Silicon Valley campus. It provides information about courses, faculty, academic calendars, degree requirements, and class schedules through a conversational interface.

## Overview

This system leverages a Retrieval-Augmented Generation (RAG) approach with the open-source Llama 3 model to provide accurate, university-specific information. It includes:

- Six specialized tools for different query types (course_search, faculty_search, academic_calendar, degree_requirements, course_schedule, northeastern_knowledge_base)
- LLM-based tool selection for optimal query routing
- Document-specific chunking strategies for improved retrieval
- Fallback RAG mechanism for robust error handling
- Streamlit web interface for easy interaction

## Setup Instructions (Google Cloud VM)

### 1. Create a Google Cloud VM

1. Go to Google Cloud Console and create a new VM instance
2. Select the following specifications:
   - Machine type: Choose a GPU-enabled instance (NVIDIA L4)
   - Operating system: Deep Learning on Linux
   - Version: Deep Learning VM with CUDA 12.3 M129
   - Boot disk size: 50 GB
   - Firewall settings:
     - Allow HTTP traffic
     - Allow HTTPS traffic
     - Allow Load Balancer Health Checks

### 2. Connect to your VM

Once your VM is running, connect via SSH

### 3. Install and Run Husky Navigator

Run the following commands:

```bash
# Clone the repository
git clone https://github.com/Ren-97/Husky_Navigator.git

# Change directory
cd Husky_Navigator

# Make the entrypoint script executable
chmod +x entrypoint.sh

# Build the Docker container
sudo bash ./docker-startup build

# Deploy with GPU support
sudo bash ./docker-startup deploy-gpu
```

### 4. Access the Web Interface

Open your browser and navigate to:
```
http://[EXTERNAL_IP]:8501
```
Replace `[EXTERNAL_IP]` with your VM's external IP address.

## Usage

Once you access the web interface, you can:

- Ask questions about courses, faculty, academic calendar, degree requirements, and more
- Use the toggle to control conversation memory and summary mode (Reset it before you want to memorize conversation)
- Reset the conversation when needed 

Example questions:
- "Tell me about DS 5110."
- "Who will teach NLP in fall 2025?"
- "Who is Karl Ni?"
- "What is the first day of summer break?"

Example Conversation with memory:
- "Reset"
- "Who is Karl Ni?"
- "Will he teach NLP in fall 2025?"

## Technical Details

- Frontend: Streamlit
- LLM: Llama 3
- Embedding Model: nomic-embed-text
- Vector Database: ChromaDB
- Framework: LangChain

## Extra Note
If Husky Navigator didn't correctly select the tool, you can add the tool you want after query.
For example, query + 'use [tool]'
