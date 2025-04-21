import os
from pathlib import Path
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Any, Union
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
import streamlit as st

"""### Load and split data"""

def load_document(path):
    filetype = path.suffix.lower()
    if filetype == ".pdf":
        return PyMuPDFLoader(str(path)).load()
    elif filetype == ".csv":
        return CSVLoader(str(path)).load()
    else:
        print(f"Unsupported file type: {filetype}")
        return []

def load_documents_with_custom_metadata(documents):

    processed_docs = []

    # Process each document and customize metadata
    for doc in documents:
        # Extract original filename from source path
        filename = os.path.basename(doc.metadata.get('source', ''))

        # Determine document type based on filename or content
        if "calendar" in filename.lower():
            doc_type = "academic_calendar"
            term = "2025"
        elif "course" in filename.lower():
            doc_type = "course_description"
            term = "current"
        elif "catalog" in filename.lower() or "requirements" in filename.lower():
            doc_type = "degree_requirements"
            term = "current"
        elif "classes" in filename.lower():
            doc_type = "open_classes"
            term = "Summer & Fall 2025"
        elif "faculty" in filename.lower() or "staff" in filename.lower():
            doc_type = "faculty_info"
            term = "current"
        else:
            doc_type = "general"
            term = "current"

        # Create new metadata
        new_metadata = {
            "genre": doc_type,
            "term": term,
            "page": doc.metadata.get('page', 0),
            "source": doc.metadata.get('source', '')
        }

        # Create a new document with the same content but updated metadata
        processed_doc = Document(
            page_content = doc.page_content,
            metadata = new_metadata
        )

        processed_docs.append(processed_doc)

    return processed_docs

def text_splitter_strategy(docs):
    """Apply different chunking strategies based on document type"""
    all_chunks = []
    doc_type = docs.metadata['genre']

    if doc_type == "academic_calendar":
        # Use smaller chunks with less overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents([docs])

    elif doc_type == "course_description":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", "Course", ". ", " ", ""]
        )
        chunks = splitter.split_documents([docs])

    elif doc_type == "faculty_info":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", "Name:", "Faculty:", ". ", " ", ""]
        )
        chunks = splitter.split_documents([docs])

    elif doc_type == "degree_requirements":
        # Degree requirements need larger chunks to maintain program context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", "Program:", "Degree:", "Requirements:", ". ", " ", ""]
        )
        chunks = splitter.split_documents([docs])

    elif doc_type == "open_classes":
        # Class schedules should be chunked by term and course
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "Term:", "Course:", ". ", " ", ""]
        )
        chunks = splitter.split_documents([docs])

    else:
        # Default chunking strategy for unknown document types
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents([docs])

    all_chunks.extend(chunks)
    return all_chunks

def load_and_split_documents(folder_path):
    all_docs = []

    for file in Path(folder_path).glob("*"):
        raw_docs = load_document(file)
        raw_docs_with_new_metadata = load_documents_with_custom_metadata(raw_docs)
        for doc in raw_docs_with_new_metadata:
            all_docs.extend(text_splitter_strategy(doc))
    return all_docs

folder_path = 'docs'
chunks = load_and_split_documents(folder_path)

"""### Vectorstores"""

llm = OllamaLLM(model="llama3", temperature=0)
embedding = OllamaEmbeddings(model = "nomic-embed-text")
persist_directory = 'chroma'

# do not run again, saved vectorbd
vectordb = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory = persist_directory
)

vectordb = Chroma(
    persist_directory = persist_directory,
    embedding_function = embedding
)

class CourseSearchInput(BaseModel):
    query: str = Field(description="The course code, name or topic to search for")

class FacultySearchInput(BaseModel):
    query: str = Field(description="The faculty name or department to search for")

class CalendarInput(BaseModel):
    query: str = Field(description="A calendar event to look up, such as start or end dates, registration periods, holidays, or deadlines. Example queries: 'Fall 2025 registration', 'when does Spring 2024 start?', 'last day to drop a class in Fall 2025.'")

class DegreeRequirementsInput(BaseModel):
    query: str = Field(description="The degree program to look up (e.g., 'MSCS', 'Computer Science masters')")

class CourseScheduleInput(BaseModel):
    query: str = Field(description="A query to search for course offerings and their details in the class schedule. Use this for finding available courses, sections, instructors, meeting times, or course status for specific terms. Example queries: 'available courses in Summer 2025', 'Computer Science courses for Fall 2025', 'evening classes taught by Dr. Smith', 'open sections of CS101', 'CRN 12345 details.'")

class GeneralKnowledgeInput(BaseModel):
    query: str = Field(description="The question or topic to search for")

def search_courses(query: str, vectorstore) -> str:
    docs = vectorstore.similarity_search(f"course {query}", k=3, filter={"genre": "course_description"})
    results = [doc.page_content for doc in docs]
    return "\n\n".join(results) if results else "No courses found matching your query."

def search_faculty(query: str, vectorstore) -> str:
    docs = vectorstore.similarity_search(f"faculty {query}", k=3, filter={"genre": "faculty_info"})
    results = [doc.page_content for doc in docs]
    return "\n\n".join(results) if results else "No faculty members found matching your query."

def search_calendar(query: str, vectorstore) -> str:
    docs = vectorstore.similarity_search(f"{query} academic calendar", k=3, filter={"genre": "academic_calendar"})
    results = [doc.page_content for doc in docs]
    return "\n\n".join(results) if results else f"No calendar information found for {query}."

def search_degree_requirements(query: str, vectorstore) -> str:
    docs = vectorstore.similarity_search(f"{query} degree requirements", k=3, filter={"genre": "degree_requirements"})
    results = [doc.page_content for doc in docs]
    return "\n\n".join(results) if results else f"No degree requirements found for {query}."

def search_course_schedule(query: str, vectorstore) -> str:
    docs = vectorstore.similarity_search(f"{query} schedule offering", k=3, filter={"genre": "open_classes"})
    results = [doc.page_content for doc in docs]
    return "\n\n".join(results) if results else f"No schedule information found for {query}."

# Function to create all tools
def create_university_tools(vectorstore, rag_chain):
    """Create and return all university info tools using current LangChain patterns."""

    # Course search tool
    course_search_tool = Tool(
        name="course_search",
        description="Search for course information by code, name, or topic",
        func=lambda query: search_courses(query, vectorstore),
        args_schema=CourseSearchInput
    )

    # Faculty search tool
    faculty_search_tool = Tool(
        name="faculty_search",
        description="Search for faculty members by name or department",
        func=lambda query: search_faculty(query, vectorstore),
        args_schema=FacultySearchInput
    )

    # Calendar tool
    calendar_tool = Tool(
        name="academic_calendar",
        description="Check important dates in the academic calendar",
        func=lambda query: search_calendar(query, vectorstore),
        args_schema=CalendarInput
    )

    # Degree requirements tool
    degree_requirements_tool = Tool(
        name="degree_requirements",
        description="Check requirements for a specific degree program",
        func=lambda query: search_degree_requirements(query, vectorstore),
        args_schema=DegreeRequirementsInput
    )

    # Course schedule tool
    course_schedule_tool = Tool(
        name="course_schedule",
        description="Check course offerings and schedules for specific terms (Summer, Fall, Spring, Winter) and years. Use this for questions about what classes are offered, course availability, and class schedules. MOST IMPORTANTLY, use this for ALL questions about which professors or instructors are teaching specific courses in any term.",
        func=lambda query: search_course_schedule(query, vectorstore),
        args_schema=CourseScheduleInput
    )

    # General knowledge tool
    general_knowledge_tool = Tool(
        name="northeastern_knowledge_base",
        description="Search for general information about Northeastern University Silicon Valley",
        func=lambda query: rag_chain.invoke({"query": query}),
        args_schema=GeneralKnowledgeInput
    )

    return [
        course_search_tool,
        faculty_search_tool,
        calendar_tool,
        degree_requirements_tool,
        course_schedule_tool,
        general_knowledge_tool
    ]

class HuskyNavigatorLlama3Agent:
    """Husky Navigator agent using Llama 3 with LLM-based tool selection"""

    def __init__(self):
        self.llm = llm
        self.vectorstore = vectordb
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._setup_retrieval()
        self._setup_tools()

        # Initialize chat history for the agent
        self.chat_history = []

    def _setup_retrieval(self):
        """Set up the RAG components for retrieval with enhanced generation"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Step 1: Query reformulation to improve retrieval
        query_reformulation_prompt = PromptTemplate.from_template(
            """You are an expert query reformulation system for Northeastern University Silicon Valley.

            Input: {original_query}

            Rewrite this query to maximize retrieval effectiveness by:
            1. Identifying academic entities (courses, faculty, programs)
            2. Adding relevant university terminology
            3. Focusing on retrievable factual information
            4. Being concise and specific

            OUTPUT ONLY THE REFORMULATED QUERY, NOTHING ELSE.
            """
        )

        # Step 2: Enhanced RAG prompt for better reasoning with context
        rag_reasoning_prompt = PromptTemplate.from_template(
            """You are Husky Navigator, the AI assistant for Northeastern University Silicon Valley.

            CONTEXT: {context}

            QUERY: {question}

            PREVIOUS CONVERSATION: {chat_history}

            TASK:
            Analyze if the context contains sufficient information to answer the query.

            REASONING STEPS:
            1. What specific information does the query request?
            2. What relevant information appears in the context?
            3. Is any critical information missing?
            4. How should I structure my response for maximum clarity?

            Your analysis:
            """
        )

        # Step 3: Final response generation prompt
        final_response_prompt = PromptTemplate.from_template(
            """You are Husky Navigator, the AI assistant for Northeastern University Silicon Valley.

            QUERY: {question}

            YOUR REASONING: {reasoning}

            PREVIOUS CONVERSATION: {chat_history}

            RESPONSE GUIDELINES:
            1. Be conversational yet professional
            2. Be direct and specific
            3. Acknowledge any information gaps
            4. Organize information logically
            5. Avoid unnecessary technical jargon

            Your response:
            """
        )

        # Define the enhanced RAG chain with multi-step reasoning
        self.rag_chain = (
            # Step 1: Query reformulation
            RunnableMap({
                "original_query": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", "")
            })
            | RunnableLambda(lambda x: {
                "original_query": x["original_query"],
                "chat_history": x["chat_history"],
                "reformulated_query": query_reformulation_prompt.format(
                    original_query=x["original_query"]
                ) | self.llm | StrOutputParser()
            })

            # Step 2: Retrieve context based on reformulated query
            | RunnableLambda(lambda x: {
                "original_query": x["original_query"],
                "chat_history": x["chat_history"],
                "reformulated_query": x["reformulated_query"],
                "context": [doc.page_content for doc in self.retriever.invoke(x["reformulated_query"])]
            })
            | RunnableLambda(lambda x: {
                "question": x["original_query"],
                "chat_history": x["chat_history"],
                "context": "\n\n".join(x["context"]) if x["context"] else "No relevant information found."
            })

            # Step 3: Reasoning step
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "chat_history": x["chat_history"],
                "context": x["context"],
                "reasoning": (rag_reasoning_prompt.format(
                    question=x["question"],
                    context=x["context"],
                    chat_history=x["chat_history"]
                ) | self.llm | StrOutputParser())
            })

            # Step 4: Final response generation
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "chat_history": x["chat_history"],
                "reasoning": x["reasoning"],
                "result": (final_response_prompt.format(
                    question=x["question"],
                    chat_history=x["chat_history"],
                    reasoning=x["reasoning"]
                ) | self.llm | StrOutputParser())
            })
            | RunnableLambda(lambda x: x["result"])
        )

    def _setup_tools(self):
        """Create the tools for different query types"""
        # Use the create_university_tools function to get the standard university tools
        university_tools = create_university_tools(self.vectorstore, self.rag_chain)

        # Create a dictionary mapping tool names to tool functions
        self.tools = {}

        # Add each university tool to the tools dictionary
        for tool in university_tools:
            # For each tool, we need to extract the function that will be called
            # The Tool object has a 'func' attribute which is the function we want
            self.tools[tool.name] = tool.func

        # Add the general chat tool which isn't part of the university tools
        self.tools["general_chat"] = lambda query: self.handle_general_chat(query)

    def handle_general_chat(self, query: str) -> str:
        """Handle general conversation that doesn't require university knowledge"""

        general_chat_prompt = """
            You are Husky Navigator, the AI assistant for Northeastern University Silicon Valley.

            QUERY: {query}

            PREVIOUS CONVERSATION: {chat_history}

            Respond conversationally while:
            1. Maintaining your identity as a university assistant
            2. Being friendly but professional
            3. Keeping responses concise
            4. Showing enthusiasm for helping with university matters

            Response:
            """

        prompt = PromptTemplate.from_template(general_chat_prompt)
        chain = prompt | self.llm | StrOutputParser()

        formatted_history = self._format_chat_history()
        return chain.invoke({"query": query, "chat_history": formatted_history})

    def _format_chat_history(self):
        """Format the chat history into a string for prompts"""
        if not self.chat_history:
            return "No previous conversation."

        formatted = []
        for entry in self.chat_history:
            if isinstance(entry, tuple) and len(entry) == 2:
                user_msg, ai_msg = entry
                formatted.append(f"User: {user_msg}")
                formatted.append(f"Husky Navigator: {ai_msg}")

        return "\n".join(formatted)

    def determine_tool_with_llm(self, query: str) -> str:
        """Use the LLM to determine which tool to use for the given query"""

        # Define a prompt for tool selection - keeping the original idea
        tool_selection_prompt = """
            As Husky Navigator, an AI assistant for Northeastern University's Silicon Valley Campus,
            you need to determine which tool is most appropriate for answering this query.

            Query: "{query}"

            Chat history:
            {chat_history}

            Available tools:
            1. course_search: For questions about specific courses, their content, prerequisites, etc.
            2. faculty_search: For questions about professors, instructors, faculty members, or staff.
            3. academic_calendar: For questions about important dates, deadlines, registration periods, etc. ONLY use this for questions about the academic year structure, deadlines, holidays, registration periods, and university-wide dates. Do NOT use this for questions about specific classes being offered.
            4. degree_requirements: For questions about degree programs, graduation requirements, credits needed, etc.
            5. course_schedule: For questions about when specific courses are offered, course availability, and class schedules. MOST IMPORTANTLY, use this for ANY questions about which professors/instructors are teaching specific courses in any term.
            6. northeastern_knowledge_base: For general questions about Northeastern University.
            7. general_chat: For casual conversation, greetings, or questions unrelated to university information.

            IMPORTANT: For ANY questions like "Who is teaching X course?" or "Which professor teaches Y in Fall 2025?" or "Is Professor Z teaching any courses next term?", you MUST use the course_schedule tool, not the faculty_search tool.

            Based on the query and chat history, which ONE tool would be most appropriate to use?
            Respond with ONLY the tool name, nothing else.
            """

        # Create and run the prompt
        prompt = PromptTemplate.from_template(tool_selection_prompt)
        chain = prompt | self.llm | StrOutputParser()

        # Format chat history for context
        formatted_history = self._format_chat_history()

        # Run the chain to get the tool name
        try:
            tool_name = chain.invoke({"query": query, "chat_history": formatted_history}).strip().lower()

            # Handle potential variations in response
            tool_map = {
                "course_search": "course_search",
                "faculty_search": "faculty_search",
                "academic_calendar": "academic_calendar",
                "degree_requirements": "degree_requirements",
                "course_schedule": "course_schedule",
                "northeastern_knowledge_base": "northeastern_knowledge_base",
                "general_chat": "general_chat",
                # Handle potential variations
                "course": "course_search",
                "faculty": "faculty_search",
                "calendar": "academic_calendar",
                "degree": "degree_requirements",
                "schedule": "course_schedule",
                "knowledge": "northeastern_knowledge_base",
                "chat": "general_chat",
                "casual": "general_chat",
                "conversation": "general_chat",
                "greeting": "general_chat",
                "1": "course_search",
                "2": "faculty_search",
                "3": "academic_calendar",
                "4": "degree_requirements",
                "5": "course_schedule",
                "6": "northeastern_knowledge_base",
                "7": "general_chat"
            }

            # Extract just the tool name if the LLM included other text
            for potential_tool in tool_map.keys():
                if potential_tool in tool_name:
                    return tool_map[potential_tool]

            # Default to general chat if no match found
            return "general_chat"

        except Exception as e:
            print(f"Error determining tool with LLM: {str(e)}")
            # Default to general chat if LLM approach fails
            return "general_chat"

    def query(self, question: str, summary_mode: bool = False, use_memory: bool = False) -> Dict[str, Any]:
        """Process a user query by selecting the appropriate tool using LLM and generating enhanced responses"""
        try:
            # If memory is disabled, reset chat history
            if not use_memory:
                print("Memory disabled. Treating as new conversation.")
                temp_history = self.chat_history.copy()  # Save history temporarily
                self.chat_history = []  # Reset history for this query
                
            # Format chat history for context
            formatted_history = self._format_chat_history()

            # Determine which tool to use with LLM
            tool_name = self.determine_tool_with_llm(question)
            print(f"Selected tool: {tool_name}")

            # Execute the tool
            tool_func = self.tools[tool_name]

            # For general chat
            if tool_name == "general_chat":
                result = tool_func(question)
                answer = result
            else:
                # For other tools, enhance the raw result with post-processing
                raw_result = tool_func(question)

                post_processing_prompt = PromptTemplate.from_template(
                    """You are Husky Navigator, the AI assistant for Northeastern University Silicon Valley.

                    QUERY: {question}

                    PREVIOUS CONVERSATION: {chat_history}

                    RAW RETRIEVED INFORMATION: {raw_result}

                    Transform this raw information into a natural, conversational response that:
                    1. Directly addresses the query without mentioning whether there was previous conversation
                    2. Presents information in logical order
                    3. Uses a friendly, professional tone
                    4. Acknowledges any limitations in the available information
                    5. References previous conversation only when directly relevant

                    Response:
                    """
                )

                # Apply post-processing to format the response nicely
                post_processing_chain = post_processing_prompt | self.llm | StrOutputParser()
                answer = post_processing_chain.invoke({
                    "question": question,
                    "chat_history": formatted_history,
                    "raw_result": raw_result
                })

            # If summary mode is enabled, summarize the response
            if summary_mode:
                summarization_prompt = PromptTemplate.from_template(
                    """Condense the following response into a brief summary:

                    ORIGINAL: {answer}

                    Your summary should:
                    1. Include only essential information
                    2. Use at most 3 sentences
                    3. Preserve critical details (names, dates, locations)
                    4. Maintain the same tone and formality level

                    Summary:
                    """
                )
                
                summarization_chain = summarization_prompt | self.llm | StrOutputParser()
                answer = summarization_chain.invoke({"answer": answer})

            # Update chat history
            self.chat_history.append((question, answer))

            # Make sure we don't keep too much history (limit to last 10 exchanges)
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
                
            # If memory was disabled, restore the previous history after processing
            if not use_memory:
                # Store this exchange temporarily
                current_exchange = self.chat_history[-1] if self.chat_history else None
                # Restore previous history
                self.chat_history = temp_history
                # If we had a new exchange, add it to the history for display purposes
                # but it won't affect future queries
                if current_exchange:
                    self.chat_history.append(current_exchange)

            return {
                "answer": answer,
                "tool_used": tool_name,
                "fallback": False
            }

        except Exception as e:
            print(f"Tool execution failed with error: {str(e)}")

            # Enhanced fallback to direct RAG with better response generation
            try:
                # Use the enhanced RAG chain directly with chat history
                response_text = self.rag_chain.invoke({
                    "question": question,
                    "chat_history": formatted_history
                })
                
                # If summary mode is enabled, summarize the fallback response too
                if summary_mode:
                    summarization_prompt = PromptTemplate.from_template(
                        """Condense the following response into a brief summary:

                        ORIGINAL: {answer}

                        Your summary should:
                        1. Include only essential information
                        2. Use at most 3 sentences
                        3. Preserve critical details (names, dates, locations)
                        4. Maintain the same tone and formality level

                        Summary:
                        """
                    )
                    
                    summarization_chain = summarization_prompt | self.llm | StrOutputParser()
                    response_text = summarization_chain.invoke({"answer": response_text})

                # Update chat history
                self.chat_history.append((question, response_text))
                
                # If memory was disabled, restore the previous history after processing
                if not use_memory:
                    # Store this exchange temporarily
                    current_exchange = self.chat_history[-1] if self.chat_history else None
                    # Restore previous history
                    self.chat_history = temp_history
                    # If we had a new exchange, add it to the history for display purposes
                    # but it won't affect future queries
                    if current_exchange:
                        self.chat_history.append(current_exchange)

                return {
                    "answer": response_text,
                    "fallback": True
                }
            except Exception as fallback_error:
                print(f"Fallback RAG also failed: {str(fallback_error)}")

                error_message = "I'm sorry, I encountered an error processing your request. Please try again with a different question."

                # Update chat history with the error response
                self.chat_history.append((question, error_message))
                
                # If memory was disabled, restore the previous history after processing
                if not use_memory:
                    # Store this exchange temporarily
                    current_exchange = self.chat_history[-1] if self.chat_history else None
                    # Restore previous history
                    self.chat_history = temp_history
                    # If we had a new exchange, add it to the history for display purposes
                    # but it won't affect future queries
                    if current_exchange:
                        self.chat_history.append(current_exchange)

                return {
                    "answer": error_message,
                    "fallback": True
                }

    def reset_memory(self):
        """Clear the conversation memory."""
        self.chat_history = []
        if hasattr(self, 'memory'):
            self.memory.clear()
            return True
        return False


husky_agent = HuskyNavigatorLlama3Agent()

"""
print("🐺 Husky Navigator initialized! (Type 'exit' to quit)")
print("---------------------------------------------------")

while True:
    # Get user input
    user_input = input("\nYou: ")

    # Check for exit command
    if user_input.lower() in ['exit', 'quit', 'bye']:
        husky_agent.reset_memory()
        print("Goodbye! Have a great day!")
        break

    # Check for reset command
    if user_input.lower() in ['reset', 'clear memory', 'forget']:
        if husky_agent.reset_memory():
            print("\nHusky Navigator: Memory has been cleared. I've forgotten our previous conversation.")
        continue

    # Process the query
    print("\nHusky Navigator is thinking...")
    try:
        response = husky_agent.query(user_input)

        # Check if fallback was used
        if response.get('fallback', False):
            print("\n(Used fallback RAG approach)")
        else:
            print(f"\n(Used tool: {response.get('tool_used', 'unknown')})")

        # Display the response
        print(f"\nHusky Navigator: {response['answer']}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nHusky Navigator: I'm sorry, I encountered an error processing your request. Could you try rephrasing your question?")

### Evaluation

from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
import random

random.seed(42)
example_gen_chain = QAGenerateChain.from_llm(llm)
examples_gen_llm = example_gen_chain.apply_and_parse([{"doc": t} for t in random.sample(chunks, 5)])

modified_examples = [{'query': 'Introduce Prof Karl',
                      'answer': 'Karl Ni is a Part-Time Lecturer at Northeastern University in Silicon Valley. In this role, he teaches graduate courses in AI / ML courses like Data Mining, Natural Language Processing, Machine Learning, etc.'},
                     {'query': 'Who will teach DS5110 this fall?',
                      'answer': 'Toutiaee, Mohammadhossein.'}]
for ex in examples_gen_llm:
    pair = ex["qa_pairs"]
    modified_examples.append({
        "query": pair["query"],
        "answer": pair["answer"]
    })

examples_by_hand = []
modified_examples += examples_by_hand

predictions = []
for example in modified_examples:
    # Get prediction from your RAG system
    husky_agent.reset_memory()
    response = husky_agent.query(example["query"])
    predictions.append({"result": response["answer"]})

eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(modified_examples, predictions)
graded_outputs

"""