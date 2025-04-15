"""
Unreal Engine Crew AI

Un sistema basado en CrewAI para interactuar con Unreal Engine a través del MCP Server.
"""

import os
import sys
import json
import ast
import logging
from typing import Dict, List, Any
from crewai import Agent, Task, Crew, Process
from langchain_ollama import OllamaLLM
from crewai.tools import tool

# Configuramos logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('unreal_crew.log'), logging.StreamHandler()
])
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
    timeout=6000,  # Aumentado de 600 a 6000 segundos
    system="Enable deep thinking subroutine."  # Activación del modo de pensamiento profundo
)

# --- 2. Herramientas personalizadas para MCP ---

@tool
def execute_mcp_command(command: str, params_json = None) -> str:
    """Ejecuta un comando específico en el servidor MCP de Unreal Engine.
    El input debe ser el nombre del comando y opcionalmente parámetros como string JSON o diccionario.
    
    Args:
        command: El nombre del comando MCP a ejecutar
        params_json: Los parámetros del comando. Puede ser un string JSON o un diccionario (default: None)
    
    Returns:
        str: La respuesta del servidor como string JSON
    
    COMANDOS PRINCIPALES Y SUS PARÁMETROS OBLIGATORIOS:
    
    1. create_blueprint:
       - REQUIERE: 'name' (string con el nombre del blueprint)
       - REQUIERE: 'parent_class' (string con la clase base, ej: 'Actor')
    
    2. spawn_actor:
       - REQUIERE: 'name' (string con nombre único)
       - REQUIERE: 'type' (string con tipo: 'CUBE', 'SPHERE', etc.)
       - OPCIONAL: 'location' ([x,y,z]), 'rotation' ([pitch,yaw,roll]), 'scale'
    
    3. add_component_to_blueprint:
       - REQUIERE: 'blueprint_name', 'component_type', 'component_name'
    
    Ejemplos:
      execute_mcp_command("get_actors_in_level")      
      execute_mcp_command("spawn_actor", '{"name": "MyCube", "type": "CUBE", "location": [0, 0, 100]}')
      execute_mcp_command("spawn_actor", {"name": "MyCube", "type": "CUBE", "location": [0, 0, 100]}')
      execute_mcp_command("find_actors_by_name", '{"pattern": "Player*"}')
      execute_mcp_command("delete_actor", '{"name": "MyCube"}')
      execute_mcp_command("set_actor_transform", '{"name": "Floor", "location": [0, 0, 0], "rotation": [0, 90, 0]}')
      execute_mcp_command("create_blueprint", '{"name": "BP_Character", "parent_class": "Character"}')
    """
    logger.info(f"--- TOOL CALL: execute_mcp_command ---")
    logger.info(f"Input command: {command}")
    logger.info(f"Input params_json (type: {type(params_json)}): {str(params_json)[:500]}...") # Log limited params
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
        
    # Validación de parámetros obligatorios para comandos críticos
    required_params = {}
    if command == "create_blueprint":
        required_params = {"name": "nombre del blueprint", "parent_class": "clase base (ej: Actor, Pawn, Character)"}
    elif command == "spawn_actor":
        required_params = {"name": "nombre único del actor", "type": "tipo de actor (ej: CUBE, SPHERE)"}
    elif command == "add_component_to_blueprint":
        required_params = {
            "blueprint_name": "nombre del blueprint",
            "component_type": "tipo de componente",
            "component_name": "nombre del componente"
        }
    
    # Verificar parámetros obligatorios
    for param_name, description in required_params.items():
        if param_name not in params or not params[param_name]:
            error_msg = f"Parámetro obligatorio '{param_name}' ({description}) no proporcionado para el comando '{command}'"
            logger.error(error_msg)
            return json.dumps({"status": "error", "error": error_msg})

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
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        error_msg = f"Error ejecutando comando MCP '{command}': {str(e)}"
        logger.error(f"{error_msg}\nTraceback:\n{tb_str}")
        result = json.dumps({"status": "error", "command": command, "error": error_msg, "details": tb_str})
    
    logger.info(f"--- TOOL RETURN: execute_mcp_command ---")
    logger.info(f"Returning: {result[:500]}...") # Log limited result
    return result

@tool
def execute_mcp_command_batch(input_data=None) -> str:
    """Ejecuta múltiples comandos MCP en secuencia de forma robusta.
    Intenta interpretar la entrada como una lista de comandos, ya sea directamente,
    como un string JSON, o extraída de un diccionario.

    Args:
        input_data: La entrada que contiene los comandos. Puede ser:
            - Una lista de diccionarios: [dict(command="...", params=dict(...)), ...]
            - Un string JSON representando la lista anterior: '[dict(command="...", params=dict(...)), ...]'
            - Un diccionario que contenga la lista bajo claves como 'commands_list' o 'commands':
              dict(commands_list=[dict(command="...", params=dict(...))])

    Returns:
        str: Un string JSON con los resultados de todos los comandos ejecutados.

    Ejemplo de uso:
        # Pasando una lista directamente
        execute_mcp_command_batch([
            dict(command="get_actors_in_level", params=dict()),
            dict(command="spawn_actor", params=dict(name="MiCubo", type="CUBE", location=[0, 0, 100]))
        ])

        # Pasando un string JSON
        execute_mcp_command_batch('[dict(command="get_actors_in_level", params=dict())]')

        # Pasando un diccionario (CrewAI podría hacer esto)
        execute_mcp_command_batch(dict(commands_list=[dict(command="get_actors_in_level", params=dict())]))
    """
    logger.info(f"--- TOOL CALL: execute_mcp_command_batch ---")
    logger.info(f"Input input_data (type: {type(input_data)}): {str(input_data)[:500]}...") # Log limited input

    # MCP Guideline: Handle default value inside the method
    if input_data is None:
        # Añadir mejor manejo de errores para debug
        import traceback
        logger.warning("Input 'input_data' es None. Mostrando stack trace para debug:")
        logger.warning(traceback.format_stack())
        # Si estamos en una ejecución real (con contexto), intentar recuperar comandos desde la tarea anterior
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'context' in frame.f_locals and isinstance(frame.f_locals['context'], list):
                    for item in frame.f_locals['context']:
                        if hasattr(item, 'output') and item.output:
                            logger.info(f"Recuperando output de contexto anterior: {str(item.output)[:500]}")
                            input_data = item.output
                            break
                frame = frame.f_back
        except Exception as e:
            logger.error(f"Error intentando recuperar contexto: {e}")
            
        if input_data is None:
            logger.warning("No se pudo recuperar input. Devolviendo lista de resultados vacía.")
            return json.dumps([]) # Return empty results list for None input

    processed_commands = None # Variable to hold the final list

    # --- Robust Input Parsing ---
    if isinstance(input_data, list):
        logger.info("Input ya es una lista.")
        processed_commands = input_data    
    elif isinstance(input_data, str):
        logger.info("Input es un string. Intentando parsear como JSON.")
        try:
            # Registrar entrada exacta para debug
            logger.info(f"String crudo recibido: '{input_data}'")
            # Mostrar en consola para debug
            print(f"Repaired JSON: {input_data}")

            # Intenta decodificar el string JSON a una lista Python
            parsed_data = json.loads(input_data)
            
            # --- MANEJO ROBUSTO PARA ESTRUCTURAS ANIDADAS ---
            # Case 1: Input es directamente una lista de comandos
            if isinstance(parsed_data, list):
                processed_commands = parsed_data
                logger.info("String JSON parseado exitosamente a lista.")
            
            # Case 2: El formato es [{"commands_list": [...]}] (lo que vemos en los logs)
            elif isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
                # Comprobar si hay una clave 'commands_list' o similar que contenga la lista real
                for key in ["commands_list", "commands", "command_list", "input", "data"]:
                    if key in parsed_data[0] and isinstance(parsed_data[0][key], list):
                        processed_commands = parsed_data[0][key]
                        logger.info(f"Extraída lista de comandos anidada desde parsed_data[0]['{key}']")
                        break
                
                # Si no encontramos una lista anidada pero cada elemento tiene 'command' y 'params'
                if processed_commands is None:
                    command_candidates = []
                    for item in parsed_data:
                        if isinstance(item, dict) and "command" in item:
                            command_candidates.append(item)
                    
                    if command_candidates:
                        processed_commands = command_candidates
                        logger.info(f"Extraídos {len(command_candidates)} comandos directamente del array de objetos")
            
            # Case 3: Input es un objeto único con una clave que contiene la lista
            elif isinstance(parsed_data, dict):
                for key in ["commands_list", "commands", "command_list", "input", "data"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        processed_commands = parsed_data[key]
                        logger.info(f"Extraída lista de comandos desde raíz['{key}']")
                        break
                
                # Si es un comando único, lo envolvemos en lista
                if processed_commands is None and "command" in parsed_data:
                    processed_commands = [parsed_data]
                    logger.info("Input es un único comando como objeto. Envuelto en lista.")
            
            # Si después de todo esto no tenemos comandos, reporte el error
            if processed_commands is None:
                error_msg = f"No se pudo extraer una lista de comandos válida del JSON: {parsed_data}"
                logger.error(error_msg)
                # Return error consistent with CrewAI's expectation if parsing fails structurally
                return json.dumps({"status": "error", "message": "Invalid input format. Expected a list of command dictionaries, but received an unrecognized structure."})
                
        except json.JSONDecodeError as e:
            error_msg = f"Error al parsear el input string como JSON: {e}. Input: {input_data}"
            logger.error(error_msg)
            logger.error(f"Posición del error: {e.pos}, línea: {e.lineno}, columna: {e.colno}")
            logger.error(f"Contexto del error: {input_data[max(0, e.pos-20):e.pos]}|ERROR|{input_data[e.pos:min(len(input_data), e.pos+20)]}")
            
            # Intento de reparación de JSON mal formado (común con agentes LLM)
            try:
                # Tratar de reparar JSON si parece que fue generado por un LLM
                import ast
                import re
                
                # PRIMERO: Intentar evaluar como literal Python si contiene dict() o []
                # Esto es más seguro y directo para el formato [dict(...), dict(...)]
                if ("dict(" in input_data and "[" in input_data and "]" in input_data) or \
                   (input_data.strip().startswith("[") and input_data.strip().endswith("]")):
                    logger.info("Detectado posible formato Python literal [dict(...)]. Intentando evaluar con ast.literal_eval.")
                    try:
                        # Usar ast.literal_eval para evaluar de forma segura el literal Python
                        evaluated_data = ast.literal_eval(input_data)
                        if isinstance(evaluated_data, list):
                            # Verificar si cada elemento es un diccionario con 'command'
                            is_valid_list = True
                            for item in evaluated_data:
                                if not isinstance(item, dict) or "command" not in item:
                                    is_valid_list = False
                                    logger.warning(f"Elemento en lista evaluada no es un dict válido con 'command': {item}")
                                    break
                            if is_valid_list:
                                processed_commands = evaluated_data
                                logger.info("Input evaluado exitosamente como lista Python literal.")
                            else:
                                logger.warning("Lista evaluada no contiene diccionarios de comando válidos.")
                        else:
                            logger.warning(f"Literal Python evaluado no es una lista (tipo: {type(evaluated_data)}).")
                    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as eval_error:
                        logger.warning(f"ast.literal_eval falló: {eval_error}")
                    except Exception as eval_ex: # Captura otras posibles excepciones de ast
                         logger.error(f"Error inesperado durante ast.literal_eval: {eval_ex}")

                # SEGUNDO: Si literal_eval falló o no aplicó, intentar reparar formato dict() a JSON
                if processed_commands is None and "dict(" in input_data:
                    logger.info("Detectado posible formato Python dict() en lugar de JSON (literal_eval falló o no aplicó). Intentando reparar a JSON.")
                    # Reemplazar dict(...) con {"...": ...}
                    repaired = input_data.replace("dict(", "{").replace(")", "}")
                    # Reemplazar comillas finales que no son válidas en JSON
                    repaired = re.sub(r',\\s*}', '}', repaired)
                    # Convertir comillas simples a dobles (manejar casos con comillas escapadas si es necesario)
                    # Esta conversión simple puede ser problemática si hay comillas dentro de strings
                    repaired = repaired.replace("\\'", "'") # Desescapar primero si es necesario
                    repaired = repaired.replace("'", '"') 
                    
                    logger.info(f"JSON reparado (intento 1): {repaired}")
                    try:
                        parsed_data = json.loads(repaired)
                        
                        if isinstance(parsed_data, list):
                            processed_commands = parsed_data
                            logger.info("JSON reparado parseado exitosamente a lista.")
                        elif isinstance(parsed_data, dict):
                            # Buscar una lista dentro del dict
                            for key in ["commands_list", "commands", "command_list"]:
                                if key in parsed_data and isinstance(parsed_data[key], list):
                                    processed_commands = parsed_data[key]
                                    logger.info(f"Lista encontrada en clave '{key}' del JSON reparado")
                                    break
                        # Añadir manejo para el caso [{"commands_list": [...]}] después de reparar
                        elif isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], dict):
                             for key in ["commands_list", "commands", "command_list", "input", "data"]:
                                if key in parsed_data[0] and isinstance(parsed_data[0][key], list):
                                    processed_commands = parsed_data[0][key]
                                    logger.info(f"Extraída lista de comandos anidada desde JSON reparado parsed_data[0]['{key}']")
                                    break
                                    
                    except json.JSONDecodeError as repair_json_error:
                         logger.error(f"Falló el parseo del JSON reparado: {repair_json_error}")
                    except Exception as repair_ex:
                         logger.error(f"Error inesperado durante el procesamiento del JSON reparado: {repair_ex}")

                # TERCERO: Si todo lo demás falla, intentar literal_eval como último recurso general
                if processed_commands is None:
                    logger.info("Intentando parsear como literal Python genérico (último recurso)")
                    try:
                        result = ast.literal_eval(input_data)
                        if isinstance(result, list):
                            processed_commands = result
                            logger.info("Input parseado como literal Python a lista (último recurso).")
                        else:
                            logger.error(f"Literal Python no es una lista: {type(result)}")
                    except Exception as final_eval_error:
                         logger.error(f"Falló el último intento con ast.literal_eval: {final_eval_error}")

            except Exception as repair_error:
                logger.error(f"Error general durante el intento de reparación/evaluación: {repair_error}")
            
            # Si todavía no tenemos comandos, devolver error
            if processed_commands is None:
                # Mejorar mensaje de error para indicar que se intentó reparar/evaluar
                return json.dumps({"status": "error", "message": f"Could not parse command list from input string, even after attempting repair/evaluation: {str(e)}"})
        except Exception as e:
            error_msg = f"Error inesperado procesando input string: {e}. Input: {input_data}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return json.dumps({"status": "error", "error": "Unexpected error processing string input", "details": str(e)})

    elif isinstance(input_data, dict):
        logger.info("Input es un diccionario. Intentando extraer lista de comandos.")
        found_list = None
        # Check common keys CrewAI might use or that make sense
        possible_keys = ["commands_list", "commands", "command_list", "batch", "input", "data"]
        for key in possible_keys:
            if key in input_data and isinstance(input_data[key], list):
                found_list = input_data[key]
                logger.info(f"Extraída lista desde la clave '{key}'.")
                break
            # Check if the value associated with the key is a JSON string list
            elif key in input_data and isinstance(input_data[key], str):
                 logger.info(f"Valor para la clave '{key}' es un string. Intentando parsear como JSON.")
                 try:
                     parsed_value = json.loads(input_data[key])
                     if isinstance(parsed_value, list):
                         found_list = parsed_value
                         logger.info(f"String JSON para la clave '{key}' parseado exitosamente a lista.")
                         break
                     else:
                         logger.warning(f"String JSON para la clave '{key}' no representa una lista.")
                 except json.JSONDecodeError:
                     logger.warning(f"String para la clave '{key}' no es JSON válido.")

        if found_list is not None:
            processed_commands = found_list
        # If no list found under common keys, check if the dict itself is a single command
        elif "command" in input_data and "params" in input_data:
             logger.info("Input parece ser un único comando en formato diccionario. Envolviendo en lista.")
             processed_commands = [input_data]
        # Check if the dictionary has exactly one value and that value is a list
        elif len(input_data) == 1:
             single_value = next(iter(input_data.values()))
             if isinstance(single_value, list):
                 logger.info("Input es un diccionario con un solo valor que es una lista. Usando esa lista.")
                 processed_commands = single_value
             elif isinstance(single_value, str):
                 logger.info("Input es un diccionario con un solo valor que es un string. Intentando parsear como JSON list.")
                 try:
                     parsed_value = json.loads(single_value)
                     if isinstance(parsed_value, list):
                         processed_commands = parsed_value
                         logger.info("String JSON de valor único parseado exitosamente a lista.")
                     else:
                         logger.warning("String JSON de valor único no representa una lista.")
                 except json.JSONDecodeError:
                     logger.warning("String de valor único no es JSON válido.")

        if processed_commands is None:
             logger.error("Diccionario de input no contiene una lista de comandos reconocible o interpretable.")
             return "Error: Could not extract a valid command list from the dictionary input."


    # --- Final Validation ---
    if not isinstance(processed_commands, list):
        # This case should ideally be unreachable now due to prior checks and returns
        error_msg = f"Después de un procesamiento robusto, no se pudo obtener una lista de comandos válida. Tipo final: {type(processed_commands)}. Input original: {str(input_data)[:500]}"
        logger.error(error_msg)
        # Return error consistent with CrewAI's expectation
        return "Error: the Action Input is not a valid key, value dictionary."

    # Check if list is empty *after* successful parsing
    if not processed_commands: # Handles empty list case
        logger.warning("La lista de comandos procesada está vacía.")
        return json.dumps({"status": "warning", "message": "Lista de comandos vacía"})

    # --- Execute Commands ---
    results = []
    logger.info(f"Iniciando ejecución de {len(processed_commands)} comandos.")
    for i, cmd_item in enumerate(processed_commands):
        command = None
        params = {}        # Validate individual command item structure
        if isinstance(cmd_item, dict):
            command = cmd_item.get("command")
            params = cmd_item.get("params", {})
            if not isinstance(command, str) or not command:
                error_msg = f"Elemento {i} de la lista es un diccionario pero no tiene una clave 'command' válida o está vacía: {cmd_item}"
                logger.error(error_msg)
                results.append(dict(status="error", index=i, error=error_msg))
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
            results.append(dict(status="error", index=i, error=error_msg))
            continue

        # Execute individual command using the single execution tool
        logger.info(f"Ejecutando comando {i+1}/{len(processed_commands)}: '{command}' con params: {params}")
        try:
            # execute_mcp_command already handles dict or string params, so passing params dict is fine
            response_json = execute_mcp_command(command=command, params_json=params)
            # Parse the JSON response string from execute_mcp_command
            try:
                response = json.loads(response_json)
                results.append(response)
                # Log success/error based on status
                if isinstance(response, dict) and response.get("status", "success") == "error":
                    logger.warning(f"Comando '{command}' ejecutado, pero devolvió error: {response.get('error', 'Unknown error')}")
                else:
                    logger.info(f"Comando '{command}' ejecutado con éxito (respuesta parcial): {str(response_json)[:200]}...")
            except json.JSONDecodeError as json_e:
                error_msg = f"Error al parsear la respuesta JSON del comando '{command}': {json_e}. Respuesta recibida: {response_json}"
                logger.error(error_msg)
                results.append(dict(status="error", command=command, error=error_msg, raw_response=response_json))

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_msg = f"Error inesperado ejecutando el comando individual '{command}' dentro del batch: {e}"
            logger.error(f"{error_msg}\nTraceback:\n{tb_str}")
            results.append(dict(status="error", command=command, error=error_msg, details=tb_str))

    # Return all results
    logger.info(f"execute_mcp_command_batch completado: {len(results)} resultados generados.")
    final_result = json.dumps(results)
    logger.info(f"--- TOOL RETURN: execute_mcp_command_batch ---")
    logger.info(f"Returning: {final_result[:500]}...") # Log limited result
    return final_result

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
    logger.info(f"--- TOOL CALL: get_available_commands ---")
    commands = {
        "Editor Tools": {
            "get_actors_in_level": "Lista todos los actores en el nivel actual. No requiere parámetros.",
            "find_actors_by_name": "Busca actores por patrón de nombre. Parámetros: 'pattern' (string con patrón de búsqueda, ej: 'Player*').",
            "spawn_actor": "Crea un nuevo actor en el nivel. Parámetros OBLIGATORIOS: 'name' (string con nombre único), 'type' (string con tipo de actor: 'CUBE', 'SPHERE', 'CYLINDER', etc). Parámetros OPCIONALES: 'location' (array [x,y,z] en cm), 'rotation' (array [pitch,yaw,roll] en grados), 'scale' (array [x,y,z] o número).",
            "delete_actor": "Elimina un actor existente por su nombre. Parámetros OBLIGATORIOS: 'name' (string con nombre exacto del actor).",
            "focus_viewport": "Enfoca la vista en un objetivo. Parámetros: 'target' (string con nombre del actor) o 'location' (array [x,y,z]). Si no se especifica, enfoca en el origen.",
            "take_screenshot": "Captura una imagen de la pantalla. Parámetros OPCIONALES: 'filename' (string con nombre de archivo), 'width', 'height' (números)."
        },
        "Blueprint Tools": {
            "create_blueprint": "Crea una nueva clase Blueprint. Parámetros OBLIGATORIOS: 'name' (string, nombre sin BP_ ni espacios), 'parent_class' (string, clase base como 'Actor', 'Pawn', 'Character', etc).",
            "add_component_to_blueprint": "Añade un componente a un Blueprint. Parámetros OBLIGATORIOS: 'blueprint_name' (string, nombre del blueprint), 'component_type' (string, tipo del componente como 'StaticMeshComponent'), 'component_name' (string, nombre para el componente).",
            "compile_blueprint": "Compila los cambios en un Blueprint. Parámetros OBLIGATORIOS: 'blueprint_name' (string, nombre del blueprint a compilar).",
            "set_blueprint_property": "Establece una propiedad en un Blueprint. Parámetros OBLIGATORIOS: 'blueprint_name' (string), 'property_name' (string), 'property_value' (valor apropiado para el tipo)."
        },
        "UMG Tools": {
            "create_umg_widget_blueprint": "Crea un nuevo Widget Blueprint para UI. Parámetros OBLIGATORIOS: 'name' (string, nombre sin WB_ ni espacios).",
            "add_text_block_to_widget": "Añade un bloque de texto a un widget. Parámetros OBLIGATORIOS: 'widget_name' (string), 'text_block_name' (string), 'text' (string, texto a mostrar).",
            "add_button_to_widget": "Añade un botón a un widget. Parámetros OBLIGATORIOS: 'widget_name' (string), 'button_name' (string). OPCIONALES: 'text' (string, texto del botón).",
            "bind_widget_event": "Vincula eventos de widgets a funciones. Parámetros OBLIGATORIOS: 'widget_name' (string), 'widget_element' (string, nombre del elemento), 'event_name' (string, nombre del evento como 'OnClicked'), 'function_name' (string, función a vincular)."
        }
    }
    
    final_result = json.dumps(commands)
    logger.info(f"--- TOOL RETURN: get_available_commands ---")
    logger.info(f"Returning: {final_result[:500]}...") # Log limited result
    return final_result

# --- 3. Definir los Agentes ---

# Agente planificador
command_planner = Agent(
    role="Command Planner",
    goal="Dividir instrucciones de alto nivel en comandos específicos para Unreal Engine",
    backstory="""Eres un experto en Unreal Engine que sabe cómo traducir instrucciones 
    en lenguaje natural a comandos específicos de MCP. Entiendes la API de MCP y puedes 
    planificar una secuencia de comandos para lograr los objetivos del usuario.
    
    **PROCESO OBLIGATORIO:**
    1. Llama a la herramienta 'get_available_commands' UNA SOLA VEZ para obtener la lista de comandos.
    2. USA INMEDIATAMENTE la lista obtenida para planificar la secuencia de comandos MCP necesarios.
    3. NO vuelvas a llamar a 'get_available_commands' después de haber obtenido la lista.
    4. Asegúrate de que TODOS los parámetros requeridos para cada comando estén presentes.
    5. Usa el formato dict() para los parámetros.
    
    GUÍA DE COMANDOS MCP PRINCIPALES:
    
    1. CREAR BLUEPRINTS:
       - Comando: 'create_blueprint'
       - Parámetros OBLIGATORIOS: 
         * 'name': String con el nombre del blueprint (sin espacios ni prefijos BP_)
         * 'parent_class': String con la clase base (como 'Actor', 'Pawn', 'Character')
       - Ejemplo: create_blueprint con params=dict(name="MyActor", parent_class="Actor")
    
    2. AÑADIR COMPONENTES A BLUEPRINTS:
       - Comando: 'add_component_to_blueprint'
       - Parámetros OBLIGATORIOS:
         * 'blueprint_name': Nombre exacto del blueprint al que añadir
         * 'component_type': Tipo de componente (ej: 'StaticMeshComponent')
         * 'component_name': Nombre para el nuevo componente
       - Ejemplo: add_component_to_blueprint con params=dict(blueprint_name="MyActor", component_type="StaticMeshComponent", component_name="Mesh")
    
    3. SPAWNER ACTORES:
       - Comando: 'spawn_actor'
       - Parámetros OBLIGATORIOS:
         * 'name': String con nombre único para el actor
         * 'type': Tipo de actor ('CUBE', 'SPHERE', etc.)
       - Parámetros OPCIONALES:
         * 'location': Array [x,y,z] en cm (usa [0,0,0] para el centro)
         * 'rotation': Array [pitch,yaw,roll] en grados
         * 'scale': Array [x,y,z] o número único
       - Ejemplo: spawn_actor con params=dict(name="MiCubo", type="CUBE", location=[0,0,100])
    
    4. LISTAR ACTORES EN NIVEL:
       - Comando: 'get_actors_in_level'
       - Sin parámetros
       - Ejemplo: get_actors_in_level con params=dict()
    
    IMPORTANTE: 
    - Asegúrate de que TODOS los parámetros requeridos estén presentes en tus comandos.
    - Siempre usa el formato correcto: dict(param1="valor1", param2=[0,0,0])
    - Para ubicar elementos en el centro de la escena, usa location=[0,0,0]
    - Al crear blueprints, siempre proporciona 'name' y 'parent_class'.
    """,
    llm=ollama_llm,
    tools=[get_available_commands],
    verbose=True,
    max_iter=5,  # Aumentar el número de iteraciones para mejor planificación
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
    
    PASOS OBLIGATORIOS:
    1. Llama a la herramienta 'get_available_commands' UNA SOLA VEZ para ver los comandos disponibles.
    2. ANALIZA la lista de comandos obtenida.
    3. USA esa lista para crear tu plan final con la secuencia de comandos MCP.
    4. NO llames a 'get_available_commands' más de una vez.
    
    Para cada comando en tu plan, especifica:
    1. Nombre del comando a ejecutar (debe existir en la lista obtenida).
    2. Parámetros necesarios con valores específicos (verifica los requeridos).
    3. Qué esperas que logre este comando.
    
    No olvides que los comandos deben ejecutarse en un orden lógico.""",    
    expected_output="""Un plan detallado con una secuencia de comandos MCP a ejecutar, usando notación dict().
    Por ejemplo:
    [
      dict(
        command="get_actors_in_level",
        params=dict(),
        description="Obtener lista de actores en el nivel actual"
      )
    ]""",
    agent=command_planner
)

# Tarea para ejecutar los comandos
execute_commands_task = Task(
    description="""⚠️⚠️⚠️ ATENCIÓN: DEBES EJECUTAR LOS COMANDOS MCP PROPORCIONADOS ⚠️⚠️⚠️

    TU ÚNICO TRABAJO ES LLAMAR A LAS HERRAMIENTAS 'execute_mcp_command' o 'execute_mcp_command_batch'.
    NO GENERES TEXTO EXPLICATIVO, SOLO LLAMA A LA HERRAMIENTA ADECUADA.

    Contexto: El 'Command Planner' ha generado una lista de comandos (disponible en el contexto de esta tarea).

    PASOS OBLIGATORIOS:
    1. Revisa la lista de comandos proporcionada por el 'Command Planner' en el contexto.
    2. Si hay UN SOLO comando en la lista:
        - Llama a la herramienta 'execute_mcp_command'.
        - El primer argumento 'command' DEBE ser el nombre del comando (string).
        - El segundo argumento 'params_json' DEBE ser el diccionario de parámetros directamente (NO un string JSON).
        - Ejemplo de llamada: 'execute_mcp_command'(command="get_actors_in_level", params_json=dict())
    3. Si hay MÚLTIPLES comandos en la lista:
        - Llama a la herramienta 'execute_mcp_command_batch'.
        - El argumento 'input_data' DEBE ser la LISTA Python COMPLETA de diccionarios de comandos, tal como la proporcionó el planificador.
        - **CRÍTICO**: NO pases un string JSON a 'input_data'. Pasa la lista Python directamente.
        - Ejemplo de llamada CORRECTA: 'execute_mcp_command_batch'(input_data=[dict(command="cmd1", params=dict()), dict(command="cmd2", params=dict(key="value"))])
        - Ejemplo de llamada INCORRECTA: 'execute_mcp_command_batch'(input_data='[dict(command="cmd1", params=dict())]') <-- ¡NO HACER ESTO!

    IMPORTANTE: Tu trabajo será verificado. Si no llamas a las herramientas 'execute_mcp_command' o 'execute_mcp_command_batch' con los argumentos en el formato correcto (diccionarios y listas Python, NO strings JSON), la tarea fallará.

    ⚠️ NO RESPONDAS NADA MÁS QUE LA LLAMADA A LA HERRAMIENTA. ⚠️
    """,
    expected_output="""La salida JSON EXACTA devuelta por la herramienta 'execute_mcp_command' o 'execute_mcp_command_batch'.
    No añadas ningún texto adicional, formato o explicación. Solo el JSON puro de la respuesta de la herramienta.

    Ejemplo si se usó 'execute_mcp_command':
    dict(status="success", actors=[dict(name="Floor", class="StaticMeshActor")])

    Ejemplo si se usó 'execute_mcp_command_batch':
    [dict(status="success", command="cmd1", result=dict()), dict(status="error", command="cmd2", error="Detalles del error")]
    """,
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
