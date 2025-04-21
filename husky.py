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
        description="Check course offerings and schedules for specific terms (Summer, Fall, Spring, Winter) and years. Use this for questions about what classes are offered, course availability, and class schedules for any term.",
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
          """
          You are a query reformulation assistant for a university information system.
          Your task is to reformulate the user's query to make it more effective for retrieving
          relevant information from a vector database.

          Original query: {original_query}

          Chat history:
          {chat_history}

          Create a reformulated search query that:
          1. Identifies the main entities (courses, faculty, programs, dates, etc.)
          2. Adds relevant university-specific terminology
          3. Focuses on retrievable factual information
          4. Is concise and specific
          5. Takes into account the conversation history for context

          Return only the reformulated query, nothing else.
          """
      )

      # Step 2: Enhanced RAG prompt for better reasoning with context
      rag_reasoning_prompt = PromptTemplate.from_template(
          """
          You are Husky Navigator, an AI assistant for Northeastern University's Silicon Valley Campus.

          Review the following retrieved context information carefully:

          {context}

          Original question: {question}

          Chat history:
          {chat_history}

          First, analyze whether the context provides sufficient information to answer the question.
          Then, use your reasoning to formulate a comprehensive answer based on:
          1. The specific information from the context
          2. The exact requirements of the question
          3. The conversation history for context

          If the context doesn't contain the necessary information, acknowledge this limitation clearly.

          Think step-by-step:
          1. What key information does the question ask for?
          2. What relevant information is present in the context?
          3. What's the most accurate and helpful way to present this information?
          4. Is any important information missing from the context?

          Provide your analysis and reasoning:
          """
      )

      # Step 3: Final response generation prompt
      final_response_prompt = PromptTemplate.from_template(
          """
          You are Husky Navigator, an AI assistant for Northeastern University's Silicon Valley Campus.

          Based on the reasoning about the question and context, provide a final response.

          Original question: {question}

          Chat history:
          {chat_history}

          Your reasoning: {reasoning}

          Now provide a clear, concise, and helpful final answer:
          1. Use a natural, conversational tone
          2. Be specific and direct
          3. Acknowledge any limitations in the available information
          4. Cite the document type when relevant
          5. Organize information logically
          6. Avoid unnecessary technical jargon

          Your final response:
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
                  original_query=x["original_query"],
                  chat_history=x["chat_history"]
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
        You are Husky Navigator, a friendly AI assistant for Northeastern University's Silicon Valley Campus.

        The user has asked a general question that doesn't require specific university knowledge.

        Question: {query}

        Chat history:
        {chat_history}

        Provide a friendly, conversational response. Keep it brief but engaging.
        Remember to maintain your identity as Husky Navigator from Northeastern University.
        Reference previous conversation if relevant.
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

        # Define a prompt for tool selection
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
          5. course_schedule: For questions about when specific courses are offered in upcoming terms, what classes are offered during specific terms (like Summer 2025), course availability, and class schedules.
          6. northeastern_knowledge_base: For general questions about Northeastern University.
          7. general_chat: For casual conversation, greetings, or questions unrelated to university information.

          IMPORTANT: If the query is asking which instructor/professor is teaching a specific course in a specific term, ALWAYS use the course_schedule tool.

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

    def query(self, question: str, summary_mode: bool = False) -> Dict[str, Any]:
        """Process a user query by selecting the appropriate tool using LLM and generating enhanced responses"""
        try:
            # Add follow-up question detection
            classification_prompt = PromptTemplate.from_template(
                "Is the following question a follow-up that requires context from earlier conversation? Answer only YES or NO.\n\nQuestion: {query}"
            )
            followup_classifier = LLMChain(llm=self.llm, prompt=classification_prompt)
            
            def is_follow_up_llm(query: str) -> bool:
                result = followup_classifier.run(query)
                return "yes" in result.lower()
            
            # Check if this is a follow-up question
            is_followup = False
            if self.chat_history:  # Only check if there is previous history
                is_followup = is_follow_up_llm(question)
                
            # If not a follow-up, reset chat history
            if not is_followup:
                print("Not a follow-up question. Resetting memory.")
                self.chat_history = []
            else:
                print("Follow-up question detected. Maintaining conversation context.")
                
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
                    """
                    You are Husky Navigator, an AI assistant for Northeastern University's Silicon Valley Campus.

                    The user asked: {question}

                    Chat history:
                    {chat_history}

                    The system retrieved the following information:

                    {raw_result}

                    Transform this raw retrieved information into a natural, conversational response that:
                    1. Directly addresses the user's question
                    2. Organizes the information in a logical flow
                    3. Uses a friendly, helpful tone
                    4. Clarifies any ambiguities
                    5. Acknowledges any limitations in the information
                    6. References the conversation history when relevant

                    Your response should sound like a knowledgeable university advisor, not a search result.
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
                    """
                    Summarize the following response in a concise paragraph:
                    
                    {answer}
                    
                    Provide a summary that:
                    1. Includes only the most important information
                    2. Is no more than 3 sentences long
                    3. Maintains the essential facts and context
                    4. Preserves any critical details (dates, times, locations, etc.)
                    
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

                # Update chat history even with fallback response
                self.chat_history.append((question, response_text))

                # Make sure we don't keep too much history (limit to last 10 exchanges)
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]

                return {
                    "answer": response_text,
                    "fallback": True
                }
            except Exception as fallback_error:
                print(f"Fallback RAG also failed: {str(fallback_error)}")

                error_message = "I'm sorry, I encountered an error processing your request. Please try again with a different question."

                # Still update chat history with the error response
                self.chat_history.append((question, error_message))

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