"""
Unreal Engine Crew AI

Un sistema basado en CrewAI para interactuar con Unreal Engine a través del MCP Server.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from crewai import Agent, Task, Crew, Process
from langchain_ollama import OllamaLLM
from crewai.tools import tool

# Configuramos logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('unreal_crew.log'), logging.StreamHandler()]
)
logger = logging.getLogger("UnrealCrew")

# Aseguramos que unreal_mcp_server esté en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos el servidor MCP
from unreal_mcp_server import get_unreal_connection

# --- 1. Configuración del modelo LLM ---
# Activamos el modo "deep thinking" de Cogito como indica la documentación
# https://ollama.com/library/cogito
ollama_llm = OllamaLLM(
    model="ollama/cogito:3b", 
    temperature=0.2,
    num_predict=4096,
    system="Enable deep thinking subroutine."  # Activación del modo de pensamiento profundo
)

# --- 2. Herramientas personalizadas para MCP ---

@tool
def execute_mcp_command(command: str, params_json = "") -> str:
    """Ejecuta un comando específico en el servidor MCP de Unreal Engine.
    El input debe ser el nombre del comando y opcionalmente parámetros como string JSON o diccionario.
    
    Args:
        command: El nombre del comando MCP a ejecutar
        params_json: Los parámetros del comando. Puede ser un string JSON o un diccionario (default: "")
    
    Returns:
        str: La respuesta del servidor como string JSON
    
    Ejemplos:
      execute_mcp_command("get_actors_in_level")
      execute_mcp_command("spawn_actor", '{"name": "MyCube", "type": "CUBE", "location": [0, 0, 100]}')
      execute_mcp_command("spawn_actor", {"name": "MyCube", "type": "CUBE", "location": [0, 0, 100]}')
      execute_mcp_command("find_actors_by_name", '{"pattern": "Player*"}')
      execute_mcp_command("delete_actor", '{"name": "MyCube"}')
      execute_mcp_command("set_actor_transform", '{"name": "Floor", "location": [0, 0, 0], "rotation": [0, 90, 0]}')
      execute_mcp_command("create_blueprint", '{"name": "BP_Character", "parent_class": "Character"}')
    """
    # Parseamos los parámetros si existen
    params = {}
    if params_json:
        try:
            if isinstance(params_json, str):
                if params_json.strip():
                    params = json.loads(params_json)
                else:
                    params = {}
            elif isinstance(params_json, dict):
                params = params_json
            else:
                error_msg = f"Formato de parámetros inválido: se esperaba string JSON o dict, se recibió {type(params_json)}"
                logger.error(f"{error_msg} - Input: {params_json}")
                return json.dumps({"status": "error", "error": error_msg})
        except json.JSONDecodeError as e:
            error_msg = f"Error al parsear parámetros JSON string: {e} - Input: {params_json}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "error": error_msg})
    else:
        params = {}

    unreal = get_unreal_connection()
    if not unreal:
        logger.error("No se pudo conectar con Unreal Engine")
        return json.dumps({"status": "error", "error": "No se pudo conectar con Unreal Engine"})
    logger.info(f"Ejecutando comando MCP: {command} con parámetros: {params}")
    try:
        response = unreal.send_command(command, params)
        if response is None:
            response = {"status": "error", "error": "No se recibió respuesta del servidor MCP"}
            logger.warning(f"Comando MCP {command} no devolvió respuesta.")
        elif not isinstance(response, dict):
            logger.warning(f"Respuesta MCP no fue un diccionario: {response}. Envolviendo.")
            response = {"status": "unknown", "raw_response": response}
        logger.info(f"Respuesta recibida para {command}: {json.dumps(response)[:200]}...")
        result = json.dumps(response)
        print(f"\n[MCP COMANDO] {command}")
        print(f"[MCP RESPUESTA] {json.dumps(response, indent=2, ensure_ascii=False)}\n")
        return result
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        error_msg = f"Error ejecutando comando MCP '{command}': {str(e)}"
        logger.error(f"{error_msg}\nTraceback:\n{tb_str}")
        return json.dumps({"status": "error", "command": command, "error": error_msg, "details": tb_str})

@tool
def execute_mcp_command_batch(commands_list: list = None) -> str: # MCP Guideline: Use list = None
    """Ejecuta múltiples comandos MCP en secuencia.
    El input debe ser una lista de diccionarios de comandos donde cada diccionario
    tiene claves 'command' y 'params'.

    Args:
        commands_list: Una lista de diccionarios de comandos a ejecutar.
            Ejemplo válido:
            [
                dict(command="get_actors_in_level", params=dict()),
                dict(command="spawn_actor", params=dict(name="MiCubo", type="CUBE", location=[0, 0, 100]))
            ]

    Returns:
        str: Un string JSON con los resultados de todos los comandos ejecutados

    Nota:
        Si solo hay un comando en la lista, es más eficiente usar 'execute_mcp_command' directamente.

    Ejemplo de uso:
        execute_mcp_command_batch([
            dict(command="get_actors_in_level", params=dict()),
            dict(command="spawn_actor", params=dict(name="MiCubo", type="CUBE", location=[0, 0, 100]))
        ])
    """
    # MCP Guideline: Handle default value inside the method
    if commands_list is None:
        error_msg = "El parámetro 'commands_list' no puede ser None. Se requiere una lista de comandos."
        logger.error(error_msg)
        return json.dumps({"status": "error", "error": error_msg})

    logger.info(f"execute_mcp_command_batch recibió input inicial: tipo={type(commands_list)}, valor={str(commands_list)[:500]}...") # Log limited value

    processed_commands = None # Variable to hold the final list

    # --- Robust Input Parsing ---
    if isinstance(commands_list, str):
        logger.info("Input es un string. Intentando parsear como JSON.")
        try:
            parsed_input = json.loads(commands_list)
            if isinstance(parsed_input, list):
                logger.info("Parseado exitosamente como lista JSON.")
                processed_commands = parsed_input
            elif isinstance(parsed_input, dict):
                 logger.info("Parseado como diccionario JSON. Intentando extraer lista interna.")
                 # Check common CrewAI patterns or direct list key
                 if "commands_list" in parsed_input and isinstance(parsed_input["commands_list"], list):
                     processed_commands = parsed_input["commands_list"]
                     logger.info("Extraída lista desde la clave 'commands_list'.")
                 elif "commands" in parsed_input and isinstance(parsed_input["commands"], list):
                     processed_commands = parsed_input["commands"]
                     logger.info("Extraída lista desde la clave 'commands'.")
                 # Add more potential keys if needed
                 else:
                     logger.warning("Diccionario JSON no contiene una lista de comandos reconocible.")
                     # Fall through to final type check
            else:
                logger.warning(f"JSON parseado no es una lista ni un diccionario: tipo={type(parsed_input)}")
                # Fall through to final type check
        except json.JSONDecodeError as e:
            error_msg = f"Error al parsear el input string como JSON: {e}. Input: {commands_list}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "error": error_msg, "details": "Input string could not be parsed as JSON."})
        except Exception as e: # Catch other potential errors during parsing/handling
            error_msg = f"Error inesperado procesando input string: {e}. Input: {commands_list}"
            logger.error(error_msg)
            return json.dumps({"status": "error", "error": error_msg, "details": "Unexpected error during string input processing."})

    elif isinstance(commands_list, dict):
        logger.info("Input es un diccionario. Intentando extraer lista de comandos.")
        # Check common CrewAI patterns or direct list key
        if "commands_list" in commands_list and isinstance(commands_list["commands_list"], list):
            processed_commands = commands_list["commands_list"]
            logger.info("Extraída lista desde la clave 'commands_list'.")
        elif "commands" in commands_list and isinstance(commands_list["commands"], list):
             processed_commands = commands_list["commands"]
             logger.info("Extraída lista desde la clave 'commands'.")
        # Check if the dict itself is a single command structure
        elif "command" in commands_list:
             logger.info("Input parece ser un único comando en formato diccionario. Envolviendo en lista.")
             processed_commands = [commands_list]
        else:
            logger.warning("Diccionario de input no contiene una lista de comandos reconocible (claves 'commands_list', 'commands') ni es un comando único.")
            # Fall through to final type check

    elif isinstance(commands_list, list):
        logger.info("Input ya es una lista.")
        processed_commands = commands_list # Already in the correct format

    # --- Final Validation ---
    if not isinstance(processed_commands, list):
        error_msg = f"Después del procesamiento, no se pudo obtener una lista de comandos válida. Tipo final: {type(processed_commands)}. Input original: {str(commands_list)[:500]}"
        logger.error(error_msg)
        return json.dumps({"status": "error", "error": error_msg, "details": "Input could not be resolved to a list of commands."})

    # Check if list is empty *after* successful parsing
    if not processed_commands: # Handles empty list case
        logger.warning("La lista de comandos procesada está vacía.")
        return json.dumps({"status": "warning", "message": "Lista de comandos vacía"})

    # --- Execute Commands ---
    results = []
    logger.info(f"Iniciando ejecución de {len(processed_commands)} comandos.")
    for i, cmd_item in enumerate(processed_commands):
        command = None
        params = {}

        # Validate individual command item structure
        if isinstance(cmd_item, dict):
            command = cmd_item.get("command")
            params = cmd_item.get("params", {})
            if not isinstance(command, str) or not command:
                 error_msg = f"Elemento {i} de la lista es un diccionario pero no tiene una clave 'command' válida o está vacía: {cmd_item}"
                 logger.error(error_msg)
                 results.append({"status": "error", "index": i, "error": error_msg})
                 continue
            if not isinstance(params, dict):
                 logger.warning(f"Elemento {i} tiene 'params' que no es un diccionario (tipo: {type(params)}). Usando diccionario vacío. Item: {cmd_item}")
                 params = {} # Default to empty dict if params is invalid type

        # Allow simple command names as strings (less common for batch but possible)
        elif isinstance(cmd_item, str) and cmd_item:
            command = cmd_item
            params = {}
            logger.info(f"Elemento {i} es un string '{command}', asumiendo comando sin parámetros.")
        else:
            error_msg = f"Elemento {i} de la lista tiene formato no reconocido o inválido: tipo={type(cmd_item)}, valor={str(cmd_item)[:100]}"
            logger.error(error_msg)
            results.append({"status": "error", "index": i, "error": error_msg})
            continue

        # Execute individual command using the single execution tool
        logger.info(f"Ejecutando comando {i+1}/{len(processed_commands)}: '{command}' con params: {params}")
        try:
            # Ensure params passed to execute_mcp_command is a dict or JSON string
            # execute_mcp_command already handles dict or string, so passing params dict is fine
            response_json = execute_mcp_command(command=command, params_json=params)
            # Parse the JSON response string from execute_mcp_command
            try:
                response = json.loads(response_json)
                results.append(response)
                # Log success only if status indicates success (or is missing)
                if isinstance(response, dict) and response.get("status", "success") == "error":
                     logger.warning(f"Comando '{command}' ejecutado, pero devolvió error: {response.get('error', 'Unknown error')}")
                else:
                     logger.info(f"Comando '{command}' ejecutado con éxito (respuesta parcial): {str(response_json)[:200]}...")
            except json.JSONDecodeError as json_e:
                 error_msg = f"Error al parsear la respuesta JSON del comando '{command}': {json_e}. Respuesta recibida: {response_json}"
                 logger.error(error_msg)
                 results.append({"status": "error", "command": command, "error": error_msg, "raw_response": response_json})

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_msg = f"Error inesperado ejecutando el comando individual '{command}' dentro del batch: {e}"
            logger.error(f"{error_msg}\nTraceback:\n{tb_str}")
            results.append({"status": "error", "command": command, "error": error_msg, "details": tb_str})

    # Return all results
    logger.info(f"execute_mcp_command_batch completado: {len(results)} resultados generados.")
    return json.dumps(results)

@tool
def get_available_commands() -> str:
    """Retorna una lista de todos los comandos disponibles en el servidor MCP con sus descripciones.
    
    Esta herramienta no requiere parámetros y devuelve un JSON con las categorías de comandos.
    
    Ejemplo:
      # Obtener todos los comandos disponibles
      get_available_commands()
      
      # Posteriormente puedes usar estos comandos con la herramienta execute_mcp_command
      # Por ejemplo, si encuentras "get_actors_in_level" en la lista:
      # execute_mcp_command("get_actors_in_level")
    """
    commands = {
        "Editor Tools": {
            "get_actors_in_level": "Lista todos los actores en el nivel actual",
            "find_actors_by_name": "Busca actores por patrón de nombre",
            "spawn_actor": "Crea un nuevo actor (parámetros: name, type, location, rotation, scale)",
            "delete_actor": "Elimina un actor existente (parámetros: name)",
            "focus_viewport": "Enfoca la vista en un objetivo",
            "take_screenshot": "Captura una imagen de la pantalla"
        },
        "Blueprint Tools": {
            "create_blueprint": "Crea una nueva clase Blueprint (parámetros: name, parent_class)",
            "add_component_to_blueprint": "Añade un componente a un Blueprint (parámetros: blueprint_name, component_type, component_name)",
            "compile_blueprint": "Compila los cambios en un Blueprint (parámetros: blueprint_name)",
            "set_blueprint_property": "Establece una propiedad en un Blueprint"
        },
        "UMG Tools": {
            "create_umg_widget_blueprint": "Crea un nuevo Widget Blueprint para UI",
            "add_text_block_to_widget": "Añade un bloque de texto a un widget",
            "add_button_to_widget": "Añade un botón a un widget",
            "bind_widget_event": "Vincula eventos de widgets a funciones"
        }
    }
    
    return json.dumps(commands)

# --- 3. Definir los Agentes ---

# Agente planificador
command_planner = Agent(
    role="Command Planner",
    goal="Dividir instrucciones de alto nivel en comandos específicos para Unreal Engine",
    backstory="""Eres un experto en Unreal Engine que sabe cómo traducir instrucciones 
    en lenguaje natural a comandos específicos de MCP. Entiendes la API de MCP y puedes 
    planificar una secuencia de comandos para lograr los objetivos del usuario.""",
    llm=ollama_llm,
    tools=[get_available_commands],
    verbose=True,
    max_iter=3,  # Limita el número de iteraciones
    max_rpm=10   # Limita las solicitudes por minuto
)

# Agente ejecutor
command_executor = Agent(
    role="Command Executor",
    goal="Ejecutar comandos MCP y verificar sus resultados para asegurar el éxito",
    backstory="""Eres un especialista en la ejecución de comandos de Unreal Engine.
    Tu trabajo es tomar un plan de comandos MCP, ejecutarlos en secuencia, verificar los 
    resultados y manejar cualquier error que pueda surgir.
    
    CRÍTICO: SIEMPRE DEBES EJECUTAR los comandos usando las herramientas execute_mcp_command o execute_mcp_command_batch.
    NUNCA respondas con solo texto o descripciones. TU ÚNICO TRABAJO ES EJECUTAR COMANDOS.
    Debes incluir el resultado completo de la ejecución en tu respuesta.""",
    llm=ollama_llm,
    tools=[execute_mcp_command, execute_mcp_command_batch],
    verbose=True,
    allow_delegation=False,  # No permitir delegación para forzar ejecución directa
    max_iter=5,  # Aumentamos el número de iteraciones para dar más oportunidades
    max_rpm=20   # Aumentamos las solicitudes por minuto
)

# --- 4. Crear las Tareas ---

# Tarea para planificar los comandos
plan_commands_task = Task(
    description="""Basado en la instrucción del usuario: '{user_prompt}', 
    planifica una secuencia de comandos MCP para Unreal Engine.
    Para cada comando, especifica:
    1. Nombre del comando a ejecutar
    2. Parámetros necesarios con valores específicos
    3. Qué esperas que logre este comando
    
    Para obtener la lista de comandos disponibles, DEBES usar la herramienta 
    'get_available_commands' sin ningún parámetro.
    
    No olvides que los comandos deben ejecutarse en un orden lógico.""",    
    expected_output="""Un plan detallado con una secuencia JSON de comandos MCP a ejecutar.
    Por ejemplo:
    [
      {
        "command": "get_actors_in_level",
        "params": {},
        "description": "Obtener lista de actores en el nivel actual"
      }
    ]""",
    agent=command_planner
)

# Tarea para ejecutar los comandos
execute_commands_task = Task(
    description="""⚠️⚠️⚠️ ATENCIÓN: DEBES EJECUTAR LOS COMANDOS ABAJO INDICADOS ⚠️⚠️⚠️

    TU ÚNICO TRABAJO ES LLAMAR A LA FUNCIÓN execute_mcp_command PARA CADA COMANDO.
    
    PASO A PASO OBLIGATORIO:
    1. El Command Planner ha generado la lista de comandos a ejecutar.
    2. Para cada comando en la lista:
        - EJECUTA execute_mcp_command(nombre_del_comando, parámetros)
        - El primer argumento es el nombre exacto del comando (ej: "get_actors_in_level") 
        - El segundo argumento deben ser los parámetros (un diccionario)
        - GUARDA la respuesta exacta devuelta por la función
    3. Si hay varios comandos, TIENES PERMITIDO usar execute_mcp_command_batch(lista_de_comandos)
    
    IMPORTANTE: Tu trabajo será verificado automáticamente. Si no ejecutas los comandos usando las funciones
    de herramientas, serás marcado como fallido inmediatamente.
    
    EJEMPLOS DE CÓMO DEBES PROCEDER:
    - Para un solo comando: resultado = execute_mcp_command("get_actors_in_level", dict())
    - Para múltiples: resultado = execute_mcp_command_batch([dict(command="cmd1", params=dict()), dict(command="cmd2", params=dict())])
    
    ⚠️ NO RESPONDAS SIN ANTES EJECUTAR LOS COMANDOS ⚠️
    """,
    expected_output="""Un reporte detallado de la ejecución de cada comando, que DEBE incluir:
    - El comando ejecutado y sus parámetros
    - La respuesta JSON EXACTA recibida del servidor
    - Una evaluación del éxito o fracaso basado en la respuesta

    Ejemplo:
    Comando: get_actors_in_level
    Respuesta: {"status": "success", "actors": [{"name": "Floor", "class": "StaticMeshActor"}, ...]}
    Evaluación: Éxito. Se encontraron X actores en la escena.""",
    agent=command_executor,
    context=[plan_commands_task]  # Depende de la salida de la tarea de planificación
)

# --- 5. Configurar el Crew ---

unreal_crew = Crew(
    agents=[command_planner, command_executor],
    tasks=[plan_commands_task, execute_commands_task],
    process=Process.sequential,  # Las tareas se ejecutarán secuencialmente
    verbose=True
)

# --- 6. Funciones auxiliares y para procesar los prompts del usuario ---

def extract_mcp_results(result_text: str) -> Dict[str, Any]:
    """
    Extrae los resultados de las llamadas MCP del texto de respuesta.
    
    Args:
        result_text: El texto completo de la respuesta
        
    Returns:
        Diccionario con los comandos y sus resultados
    """
    mcp_results = {}
    
    # Busca patrones de respuestas JSON en el texto
    import re
    
    # Patrón para encontrar respuestas JSON entre comillas o bloques de código
    json_patterns = [
        r'```json\s*(.*?)\s*```',  # Dentro de bloques de código markdown JSON
        r'```\s*({\s*".*?}\s*)```',  # Dentro de bloques de código genérico con JSON
        r'"Ejecutar Comando MCP".*?comando\s+([a-zA-Z_][a-zA-Z0-9_]*)\s.*?resultado[:\s]+({\s*".*?})',  # Cuando se menciona explícitamente
        r'({[\s\S]*?"status"[\s\S]*?})',  # JSON con campo status (formato MCP)
        r'({[\s\S]*?"actors"[\s\S]*?})',  # JSON con campo actors (formato get_actors_in_level)
        r'({[\s\S]*?"success"[\s\S]*?})'   # JSON con campo success (formato común)
    ]
    
    # Busca todos los patrones en el texto
    for pattern in json_patterns:
        matches = re.finditer(pattern, result_text, re.DOTALL)
        for match in matches:
            try:
                json_str = match.group(1).strip()
                # Intenta extraer el comando del contexto
                command_match = re.search(r'comando\s+([a-zA-Z_][a-zA-Z0-9_]*)', 
                                         result_text[max(0, match.start() - 100):match.start()], 
                                         re.IGNORECASE)
                command = command_match.group(1) if command_match else "unknown_command"
                
                # Intenta parsear el JSON
                result_data = json.loads(json_str)
                mcp_results[command] = result_data
            except json.JSONDecodeError:
                logger.debug(f"Texto no es JSON válido: {json_str}")
            except Exception as e:
                logger.debug(f"Error procesando coincidencia: {e}")
    
    return mcp_results

def process_unreal_prompt(user_prompt: str):
    """
    Procesa un prompt de usuario y lo convierte en acciones en Unreal Engine.
    
    Args:
        user_prompt: Instrucción del usuario en lenguaje natural
        
    Returns:
        El resultado de la ejecución como string
    """
    print(f"Procesando prompt: {user_prompt}")
    print("-" * 50)
    
    # Iniciamos el crew con el prompt del usuario
    result = unreal_crew.kickoff(inputs={'user_prompt': user_prompt})
    
    print("-" * 50)
    print("Ejecución finalizada.")
    
    # Extraemos y mostramos los resultados de las llamadas MCP
    if isinstance(result, str):
        try:
            # Intentamos encontrar respuestas MCP en el resultado
            mcp_results = extract_mcp_results(result)
            if mcp_results:
                print("\nResultados de las llamadas MCP:")
                for cmd, res in mcp_results.items():
                    print(f"\nComando: {cmd}")
                    print(json.dumps(res, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Error al procesar resultados MCP: {e}")
    
    return result

# --- 7. Punto de entrada principal ---

if __name__ == "__main__":
    # Si hay argumentos, usamos el primero como prompt
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        # Si no hay argumentos, pedimos al usuario
        prompt = input("Ingresa tu instrucción para Unreal Engine: ")
    
    # Procesamos el prompt
    resultado = process_unreal_prompt(prompt)
    
    print("Resultado final:")
    print(resultado)
