import os
from crewai import Agent, Task, Crew, Process
# Use the recommended OllamaLLM from langchain_ollama
from langchain_ollama import OllamaLLM
# Import the tool decorator from crewai.tools (based on provided documentation)
from crewai.tools import tool

# --- 1. Configuration ---
# Set Ollama API endpoint if it's not the default (http://localhost:11434)
# os.environ["OLLAMA_HOST"] = "http://your_ollama_server:11434" # Uncomment and modify if needed

# Specify the Ollama model
# Make sure the model (e.g., "cogito:3b") is available in your Ollama instance
# Run `ollama list` in your terminal to see available models
# Run `ollama pull cogito:3b` to download it if needed
ollama_llm = OllamaLLM(model="ollama/cogito:3b", # Added 'ollama/' prefix
                       temperature=0.1,
                       # Use num_predict instead of max_tokens for langchain_ollama
                       num_predict=4096)

# --- 2. Define Custom Tools ---

# Define the tool using the @tool decorator (imported from crewai.tools)
@tool("Simulated Search Tool")
def simulated_search_function(query: str) -> str:
    """Simulates a web search and returns a fixed snippet about AI impact in Spain.
    The input should be the search query string."""
    print(f"--- simulated_search_function: Received query '{query}' ---") # Added print for debugging
    # Simulate finding a relevant snippet regardless of the query
    return ("Según análisis recientes (simulados), la IA está transformando sectores clave en España como el turismo y la banca, "
            "automatizando tareas repetitivas y creando nuevas oportunidades en análisis de datos y desarrollo de IA. "
            "Existe una creciente demanda de perfiles con habilidades digitales avanzadas.")

# --- 3. Define Agents ---
# Agent 1: Project Manager
manager = Agent(
  role='Project Manager',
  goal='Break down the main goal into smaller, manageable tasks for the team. Ensure smooth execution.',
  backstory="""You are an experienced project manager, skilled in decomposing complex objectives
  into actionable steps. You coordinate the team's efforts effectively.""",
  llm=ollama_llm, # Assign the Ollama LLM
  verbose=True,
  allow_delegation=False # This manager assigns tasks, doesn't delegate its own decomposition task
)

# Agent 2: Researcher
researcher = Agent(
  role='Lead Researcher',
  goal='Find and gather relevant, up-to-date information on the impact of AI in the Spanish job market.',
  backstory="""You are an expert researcher with a knack for finding the most pertinent
  information for analysis using available tools. You analyze and organize facts effectively.""",
  # Assign the decorated function directly
  tools=[simulated_search_function], # Use the custom tool function
  llm=ollama_llm,
  verbose=True,
  allow_delegation=False
)

# Agent 3: Writer
writer = Agent(
  role='Content Writer',
  goal='Synthesize the research findings into a clear and concise summary of 500 words.',
  backstory="""You are a skilled writer, capable of transforming complex information
  into easily understandable text. You focus on clarity, conciseness, and accuracy.""",
  llm=ollama_llm, # Assign the Ollama LLM
  verbose=True,
  allow_delegation=False
)

# Agent 4: Reviewer
reviewer = Agent(
  role='Quality Editor',
  goal='Review the drafted summary for accuracy, clarity, coherence, and adherence to the 500-word limit.',
  backstory="""You are a meticulous editor with a keen eye for detail. You ensure the final
  output meets the highest standards of quality and fulfills all requirements.""",
  llm=ollama_llm, # Assign the Ollama LLM
  verbose=True,
  allow_delegation=False
)

# --- 4. Define Tasks ---
# Task 1: Decompose the main goal (Managed by the Manager)
task_decompose = Task(
  description="""Break down the main goal: '{main_goal}' into specific sub-tasks for the researcher and writer.
  Focus on identifying key areas to research and the structure of the final summary.""",
  expected_output="""A list of clear, actionable sub-tasks. For example:
  1. Research current AI adoption rates in Spain by sector.
  2. Identify jobs most likely to be affected (created/displaced).
  3. Find statistics on skills gaps related to AI.
  4. Outline the structure for a 500-word summary.""",
  agent=manager
)

# Task 2: Research (Managed by the Researcher, depends on Task 1)
task_research = Task(
  description="""Based on the manager's plan, conduct thorough research on the impact of AI
  in the Spanish job market using the provided simulated search tool.""", # Updated description
  expected_output="""A detailed report summarizing key findings, statistics, and relevant sources
  regarding AI's impact on employment in Spain, based on the simulated search results.""",
  agent=researcher,
  context=[task_decompose] # Depends on the output of task_decompose
)

# Task 3: Write Summary (Managed by the Writer, depends on Tasks 1 & 2)
task_write = Task(
  description="""Using the manager's outline and the researcher's findings, write a concise
  summary of 500 words about the impact of AI on the Spanish job market.""",
  expected_output="""A well-structured draft summary of approximately 500 words, synthesizing
  the research findings accurately and clearly.""",
  agent=writer,
  context=[task_decompose, task_research] # Depends on outputs of task_decompose and task_research
)

# Task 4: Review Summary (Managed by the Reviewer, depends on Tasks 2 & 3)
task_review = Task(
  description="""Review the draft summary. Check for accuracy based on the research report,
  ensure clarity, coherence, and that the word count is close to 500 words.
  Provide the final, polished version.""",
  expected_output="""The final, reviewed, and polished 500-word summary on the impact of AI
  in the Spanish job market, ready for publication.""",
  agent=reviewer,
  context=[task_research, task_write] # Depends on outputs of task_research and task_write
)

# --- 5. Create the Crew ---
crew = Crew(
  agents=[manager, researcher, writer, reviewer],
  tasks=[task_decompose, task_research, task_write, task_review],
  process=Process.sequential, # Tasks will run in the order they are listed
  verbose=True # Changed from 2 to True for boolean validation
)

# --- 6. Execute the Crew ---
# Define the main goal input for the workflow
main_goal_input = "Investigar el impacto de la IA en el mercado laboral español y redactar un resumen de 500 palabras."

print("Starting the Crew execution...")
print(f"Main Goal: {main_goal_input}")
print("-" * 50)

# Kick off the crew's work
result = crew.kickoff(inputs={'main_goal': main_goal_input})

print("-" * 50)
print("Crew execution finished.")
print("Final Result:")
print(result)
