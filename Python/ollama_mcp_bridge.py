"""
Ollama MCP Bridge - Servidor puente entre Ollama y el Model Context Protocol para Unreal Engine.

Este servidor permite utilizar modelos de Ollama para controlar Unreal Engine a través del MCP.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Union, Any
import pathlib
import re

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import httpx
import ollama

# Obtener la ruta base del directorio actual del script
BASE_DIR = pathlib.Path(__file__).parent.absolute()
STATIC_DIR = BASE_DIR / 'static'
# Ruta a la documentación de las herramientas
DOCS_TOOLS_DIR = pathlib.Path(BASE_DIR.parent) / 'Docs' / 'Tools'
# Definir ruta del archivo de log
LOG_FILE = pathlib.Path(BASE_DIR.parent) / 'ollama_mcp_bridge.log'

# Importamos la conexión de Unreal del servidor MCP existente
from unreal_mcp_server import UnrealConnection, get_unreal_connection

# Configuración de logging
# Crear un directorio para los logs si no existe
log_dir = pathlib.Path(BASE_DIR.parent)
LOG_FILE = log_dir / 'ollama_mcp_bridge.log'

# Configuración de logging más explícita
# Obtener el logger
logger = logging.getLogger("OllamaMCPBridge")
logger.setLevel(logging.INFO)  # Establecer nivel a INFO

# Eliminar handlers existentes para evitar duplicación
if logger.handlers:
    logger.handlers.clear()

# Crear handlers
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
console_handler = logging.StreamHandler()

# Crear formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Añadir handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configuración de Ollama
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "cogito:8b")

# Configuración del servidor
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Estado global
message_history: List[Dict] = []
# Añadimos variables para controlar la frecuencia de verificación de estado
last_status_check = 0
STATUS_CHECK_INTERVAL = 60  # Intervalo en segundos (60 = 1 minuto)
cached_status = {
    "mcp_connected": False,
    "ollama_available": False,
    "message_history": [],
    "timestamp": 0
}

# Función para cargar la documentación de las herramientas MCP
def load_mcp_tools_documentation():
    """
    Lee los archivos de documentación en la carpeta Docs/Tools y genera
    un prompt de sistema con la documentación de las herramientas MCP.
    """
    tool_files = [
        "actor_tools.md",
        "blueprint_tools.md",
        "node_tools.md",
        "editor_tools.md",
        "content_tools.md"
    ]
    
    doc_content = "# Unreal MCP Tools Reference\n\n"
    
    # Verificar si la ruta existe
    if not DOCS_TOOLS_DIR.exists():
        logger.warning(f"Directorio de documentación no encontrado: {DOCS_TOOLS_DIR}")
        # Proporcionar documentación básica fallback
        return """
## Actor Management
- `get_actors_in_level()` - Get all actors in the current level
- `create_actor(name, type, location, rotation, scale)` - Create actors (e.g. CUBE, SPHERE, CAMERA, LIGHT)
- `delete_actor(name)` - Remove actors
- `set_actor_transform(name, location, rotation, scale)` - Modify actor position, rotation, and scale

## Content Management
- `get_project_assets(path_pattern, asset_type)` - Get assets in the project
"""
    
    has_files = False
    for file_name in tool_files:
        file_path = DOCS_TOOLS_DIR / file_name
        if file_path.exists():
            has_files = True
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Eliminar la primera línea si es un comentario de filepath
                    content = re.sub(r'^<!--.*?-->\s*', '', content, flags=re.DOTALL)
                    # Añadir el contenido al documento final
                    doc_content += content + "\n\n"
            except Exception as e:
                logger.error(f"Error leyendo documentación de {file_name}: {str(e)}")
    
    # Si no se encontró ningún archivo, usar documentación básica
    if not has_files:
        logger.warning("No se encontraron archivos de documentación. Usando documentación básica fallback.")
        return """
## Actor Management
- `get_actors_in_level()` - Get all actors in the current level
- `create_actor(name, type, location, rotation, scale)` - Create actors (e.g. CUBE, SPHERE, CAMERA, LIGHT)
- `delete_actor(name)` - Remove actors
- `set_actor_transform(name, location, rotation, scale)` - Modify actor position, rotation, and scale

## Content Management
- `get_project_assets(path_pattern, asset_type)` - Get assets in the project
"""
    
    # Añadir ejemplos de uso al final
    doc_content += """
Ejemplos de comandos válidos:
1. Para crear un actor: 
{"command": "create_actor", "parameters": {"name": "MiEsfera", "type": "SPHERE", "location": [0, 0, 100]}}

2. Para mover un actor:
{"command": "set_actor_transform", "parameters": {"name": "MiEsfera", "location": [200, 0, 100]}}

3. Para consultar actores:
{"command": "get_actors_in_level", "parameters": {}}

4. Para crear un blueprint:
{"command": "create_blueprint", "parameters": {"name": "MiBlueprint", "parent_class": "Actor"}}

5. Para añadir un componente a un blueprint:
{"command": "add_component_to_blueprint", "parameters": {"blueprint_name": "MiBlueprint", "component_type": "StaticMesh", "component_name": "MiComponente"}}

6. Para obtener assets del proyecto:
{"command": "get_project_assets", "parameters": {"path_pattern": "/Game/", "asset_type": "Blueprint"}}

7. Para obtener la estructura del proyecto:
{"command": "get_project_structure", "parameters": {"root_path": "/Game/"}}

8. Para buscar assets:
{"command": "search_assets", "parameters": {"search_term": "Player", "exact_match": false}}

9. Para obtener detalles de un asset:
{"command": "get_asset_details", "parameters": {"asset_path": "/Game/Blueprints/BP_Player"}}

10. Para obtener clases de actor de un plugin:
{"command": "get_plugin_actor_classes", "parameters": {"plugin_name": "MyPlugin"}}

11. Para instanciar un blueprint existente:
{"command": "spawn_blueprint_actor", "parameters": {"blueprint_name": "MyBlueprint", "actor_name": "MiActorInstancia", "location": [0, 0, 100]}}

Basándote en la instrucción del usuario, genera el comando MCP apropiado en formato JSON.
"""
    
    # No olvidarse de retornar la documentación
    return doc_content

# Cargar la documentación MCP para el prompt del sistema
MCP_TOOLS_DOCUMENTATION = load_mcp_tools_documentation()

# Aplicación FastAPI
app = FastAPI(title="Ollama MCP Bridge")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos
class NLRequest(BaseModel):
    text: str
    model: Optional[str] = None

class DirectCommandRequest(BaseModel):
    command: str
    parameters: Dict = {}

class ConnectionRequest(BaseModel):
    url: Optional[str] = None

# Rutas de la API
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/status")
async def get_status():
    """Devuelve el estado actual de las conexiones"""
    global last_status_check, cached_status
    current_time = time.time()
    
    # Verificar si es necesario actualizar el estado (solo cada minuto)
    if current_time - last_status_check > STATUS_CHECK_INTERVAL:
        logger.info("Actualizando estado (verificación programada)")
        unreal_conn = get_unreal_connection()
        ollama_available = await check_ollama()
        
        cached_status = {
            "mcp_connected": unreal_conn is not None and unreal_conn.connected,
            "ollama_available": ollama_available,
            "message_history": message_history[-20:] if message_history else [],
            "timestamp": current_time
        }
        last_status_check = current_time
    else:
        # Usar caché para evitar llamadas frecuentes
        logger.debug(f"Usando estado en caché (próxima actualización en {int(STATUS_CHECK_INTERVAL - (current_time - last_status_check))} segundos)")
    
    # Siempre actualizar el historial de mensajes aunque usemos caché para otros valores
    cached_status["message_history"] = message_history[-20:] if message_history else []
    
    return cached_status

@app.post("/api/connect")
async def connect_to_mcp(request: ConnectionRequest):
    """Conecta al servidor MCP de Unreal Engine"""
    try:
        unreal_conn = get_unreal_connection()
        if (unreal_conn and unreal_conn.connected):
            return {"success": True, "message": "Ya conectado al servidor MCP"}
        
        unreal_conn = UnrealConnection()
        success = unreal_conn.connect()
        
        if success:
            # Actualiza la conexión global
            global _unreal_connection
            _unreal_connection = unreal_conn
            return {"success": True, "message": "Conectado al servidor MCP"}
        else:
            return {"success": False, "message": "Error al conectar con el servidor MCP"}
    except Exception as e:
        logger.error(f"Error conectando al servidor MCP: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/api/send-command")
async def send_command(request: DirectCommandRequest):
    """Envía un comando directo al servidor MCP"""
    try:
        unreal_conn = get_unreal_connection()
        if not unreal_conn or not unreal_conn.connected:
            raise HTTPException(status_code=400, detail="No hay conexión con el servidor MCP")
        
        response = unreal_conn.send_command(request.command, request.parameters)
        
        # Registrar en el historial
        message_history.append({"source": "client", "data": {"command": request.command, "parameters": request.parameters}})
        message_history.append({"source": "unreal", "data": response})
        
        return response
    except Exception as e:
        logger.error(f"Error enviando comando: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-nl")
async def process_natural_language(request: NLRequest):
    """Procesa lenguaje natural usando Ollama y envía comandos MCP"""    
    try:
        
        # Asignar el modelo correctamente
        model = request.model or OLLAMA_MODEL        
        logger.info(f"Procesando solicitud NL con modelo: {model}")
        logger.info(f"Texto de entrada: {request.text}")
          # Sistema de prompt para Ollama - usamos la documentación cargada dinámicamente
        system_prompt = """Eres un asistente experto en Unreal Engine. 
Tu tarea es analizar instrucciones en lenguaje natural y convertirlas en comandos específicos que Unreal Engine pueda ejecutar.
Enable deep thinking subroutine.
Debes responder con uno o varios JSON válidos que contenga uno o varios comandos con los parámetros adecuados para el MCP (Model Context Protocol).

 # Unreal MCP Server Tools and Best Practices    
    ## Actor Management
    - `get_actors_in_level()` - Obtiene todos los actores en el nivel actual
      Ejemplo: {"command": "get_actors_in_level", "parameters": {}}
    
    - `find_actors_by_name(name)` - Busca actores por nombre (admite comodines)
      Ejemplo: {"command": "find_actors_by_name", "parameters": {"name": "Player*"}}
      
    - `get_actor_properties(name)` - Obtiene las propiedades de un actor específico
      Ejemplo: {"command": "get_actor_properties", "parameters": {"name": "PlayerCharacter"}}
      
    - `create_actor(name, type, location, rotation, scale)` - Crea actores en el nivel
      Ejemplo: {"command": "create_actor", "parameters": {"name": "MiCubo", "type": "CUBE", "location": [0, 0, 100]}}
      
    - `delete_actor(name)` - Elimina un actor del nivel
      Ejemplo: {"command": "delete_actor", "parameters": {"name": "MiCubo"}}
      
    - `set_actor_transform(name, location, rotation, scale)` - Modifica la transformación de un actor
      Ejemplo: {"command": "set_actor_transform", "parameters": {"name": "MiCubo", "location": [100, 0, 50], "rotation": [0, 45, 0]}}
      
    - `get_selected_actors()` - Obtiene los actores seleccionados actualmente en el editor
      Ejemplo: {"command": "get_selected_actors", "parameters": {}}
    
    ## Content Discovery Tools
    - `get_project_assets(path_pattern, asset_type)` - Obtiene assets del proyecto
      Ejemplo: {"command": "get_project_assets", "parameters": {"path_pattern": "/Game/", "asset_type": "Blueprint"}}
      
    - `get_project_structure(root_path)` - Obtiene la estructura de carpetas del proyecto
      Ejemplo: {"command": "get_project_structure", "parameters": {"root_path": "/Game/"}}
      
    - `search_assets(search_term, exact_match)` - Busca assets por nombre
      Ejemplo: {"command": "search_assets", "parameters": {"search_term": "Player", "exact_match": false}}
      
    - `get_asset_details(asset_path)` - Obtiene detalles de un asset específico
      Ejemplo: {"command": "get_asset_details", "parameters": {"asset_path": "/Game/Blueprints/BP_Player"}}
    
    ## Blueprint Management
    - `create_blueprint(name, parent_class)` - Crea un nuevo Blueprint
      Ejemplo: {"command": "create_blueprint", "parameters": {"name": "BP_MiPersonaje", "parent_class": "Character"}}
      
    - `add_component_to_blueprint(blueprint_name, component_type, component_name)` - Añade un componente a un Blueprint
      Ejemplo: {"command": "add_component_to_blueprint", "parameters": {"blueprint_name": "BP_MiPersonaje", "component_type": "StaticMesh", "component_name": "MiMalla"}}
      
    - `set_component_property(blueprint_name, component_name, property_name, property_value)` - Establece propiedades de un componente
      Ejemplo: {"command": "set_component_property", "parameters": {"blueprint_name": "BP_MiPersonaje", "component_name": "MiMalla", "property_name": "StaticMesh", "property_value": "/Game/Meshes/Cube"}}
      
    - `spawn_blueprint_actor(blueprint_name, actor_name, location, rotation, scale)` - Instancia un Blueprint en el nivel
      Ejemplo: {"command": "spawn_blueprint_actor", "parameters": {"blueprint_name": "BP_MiPersonaje", "actor_name": "Jugador1", "location": [0, 0, 100]}}
    
    ## Blueprint Node Management
    - `add_function_to_blueprint(blueprint_name, function_name)` - Añade una función a un Blueprint
      Ejemplo: {"command": "add_function_to_blueprint", "parameters": {"blueprint_name": "BP_MiPersonaje", "function_name": "Saltar"}}
      
    - `add_variable_to_blueprint(blueprint_name, variable_name, variable_type, default_value)` - Añade una variable a un Blueprint
      Ejemplo: {"command": "add_variable_to_blueprint", "parameters": {"blueprint_name": "BP_MiPersonaje", "variable_name": "Salud", "variable_type": "float", "default_value": 100.0}}
      
    - `add_event_to_blueprint(blueprint_name, event_name)` - Añade un evento a un Blueprint
      Ejemplo: {"command": "add_event_to_blueprint", "parameters": {"blueprint_name": "BP_MiPersonaje", "event_name": "BeginPlay"}}
      
    - `add_node_to_graph(blueprint_name, graph_name, node_type, node_name, node_position)` - Añade un nodo al grafo
      Ejemplo: {"command": "add_node_to_graph", "parameters": {"blueprint_name": "BP_MiPersonaje", "graph_name": "Saltar", "node_type": "K2_AddActorLocalOffset", "node_name": "MoverArriba", "node_position": [300, 200]}}
    
    ## Editor Tools
    - `get_editor_selection()` - Obtiene los elementos seleccionados en el editor
      Ejemplo: {"command": "get_editor_selection", "parameters": {}}
      
    - `get_editor_viewport_info()` - Obtiene información sobre las vistas del editor
      Ejemplo: {"command": "get_editor_viewport_info", "parameters": {}}
      
    - `set_editor_viewport_camera(viewport_index, location, rotation)` - Controla la cámara de una vista
      Ejemplo: {"command": "set_editor_viewport_camera", "parameters": {"viewport_index": 0, "location": [0, 0, 500], "rotation": [0, -90, 0]}}
      
    - `get_plugin_actor_classes(plugin_name)` - Obtiene las clases de actores de un plugin
      Ejemplo: {"command": "get_plugin_actor_classes", "parameters": {"plugin_name": "MyPlugin"}}
  
  
    ## Best Practices
    ### Actor Creation and Management
    - When creating actors, always provide a unique name to avoid conflicts
    - Valid actor types include: CUBE, SPHERE, PLANE, CYLINDER, CONE, CAMERA, LIGHT, POINT_LIGHT, SPOT_LIGHT
    - Location is specified as [x, y, z] in Unreal units
    - Rotation is specified as [pitch, yaw, roll] in degrees
    - Scale is specified as [x, y, z] multipliers (1.0 is default scale)
    - Always clean up temporary actors when no longer needed

    ### Content Discovery Best Practices
    - Use specific path patterns to narrow down asset searches: `get_project_assets("/Game/Characters/", "Blueprint")`
    - Fetch plugin classes before implementation: `get_plugin_actor_classes("MyCustomPlugin")`
    - Explore project structure to understand the content organization: `get_project_structure("/Game/Levels/")`
    - Search for existing assets before creating new ones: `search_assets("PlayerCharacter", false)`
    - Get detailed asset information for referencing properties: `get_asset_details("/Game/Blueprints/BP_Player")`
    - Use asset search results to find dependencies or related content
    - Verify asset paths are correct by checking project structure first
    - Remember to include the full path when referencing assets in other commands
    
    ### Blueprint Development
    - Always compile Blueprints after making changes
    - Use meaningful names for variables and functions
    - Organize nodes in the graph for better readability
    - Test Blueprint functionality in a controlled environment
    - Use proper variable types for different data needs
    - Consider performance implications when adding nodes
    
    ### Node Graph Management
    - Position nodes logically to maintain graph readability
    - Use appropriate node types for different operations
    - Connect nodes with proper pin types
    - Document complex node setups with comments
    - Test node connections before finalizing
    
    ### Input Mapping
    - Use descriptive names for input actions
    - Consider platform-specific input needs
    - Test input mappings thoroughly
    - Document input bindings for team reference
    
    ### Error Handling
    - Always check command responses for success status
    - Handle error cases gracefully
    - Log important operations and errors
    - Validate parameters before sending commands
    - Clean up resources in error cases
    
Basándote en la instrucción del usuario, genera el comando MCP apropiado en formato JSON.
"""
        print(system_prompt)  # Debugging: imprimir el prompt del sistema para verificar su contenido
        
        # Llamada a la biblioteca oficial de Ollama para procesar el texto con streaming
        logger.info(f"Usando la biblioteca de Ollama para conectarse a {OLLAMA_API_BASE}")
        try:
            # Configurar el cliente de Ollama con la URL correcta
            client = ollama.Client(host=OLLAMA_API_BASE.replace('http://', ''))
              # Recoger la respuesta completa mientras se hace streaming
            logger.info("Iniciando streaming de la respuesta de Ollama")
            full_response = ""
     
            # Preparar los mensajes para la API de chat
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": request.text
                }
            ]
              # Usar la API de chat de Ollama con streaming
            for chunk in client.chat(
                model=model,
                messages=messages,
                stream=True,
            ):
                if chunk.get('message') and chunk['message'].get('content') and chunk['message']['content'] is not None:
                    full_response += str(chunk['message']['content'])  # Convertir explícitamente a string
                # Podríamos implementar streaming al cliente aquí si fuera necesario
            
            generated_text = full_response
            
            # Log the full response for debugging
            logger.info(f"Texto generado por Ollama (primeros 100 caracteres): {generated_text[:100]}...")
            logger.debug(f"Texto completo: {generated_text}")
            
            # Improved JSON extraction with better regex that handles formatted output
            # This pattern looks for a JSON object potentially surrounded by markdown code fences or other text
            # Try multiple patterns to extract JSON
            json_patterns = [
                r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',  # JSON in code blocks with optional language
                r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})',      # JSON object with potential nested objects
                r'({.*})',                              # Simple fallback pattern
            ]
            
            command_json = None
            for pattern in json_patterns:
                matches = re.findall(pattern, generated_text, re.DOTALL)
                if matches:
                    for potential_json in matches:
                        try:
                            # Try to parse each potential JSON match
                            parsed = json.loads(potential_json.strip())
                            if isinstance(parsed, dict) and "command" in parsed:
                                command_json = parsed
                                logger.info(f"JSON extraído con éxito usando patrón: {pattern}")
                                logger.info(f"JSON extraído: {command_json}")
                                break
                        except json.JSONDecodeError as e:
                            error_msg = f"No se pudo parsear el JSON generado: {str(e)}"
                            logger.error(error_msg)
                            logger.error(f"JSON problemático: {potential_json}")
                            logger.error(f"Posición del error: {e.pos}, línea: {e.lineno}, columna: {e.colno}")
                            continue
                if command_json:
                    break
            
            if not command_json:
                error_msg = "No se pudo extraer un comando JSON válido de la respuesta de Ollama"
                logger.error(f"{error_msg}: {generated_text}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "ollama_response": generated_text
                }
                
            try:
                logger.info(f"JSON extraído: {command_json}")
                
                # Validar estructura
                if "command" not in command_json:
                    error_msg = "Comando generado no válido: falta el campo 'command'"
                    logger.error(error_msg)
                    return {
                        "status": "error",
                        "message": error_msg,
                        "ollama_response": generated_text
                    }
                    
                # Registrar en el historial la solicitud a Ollama
                message_history.append({
                    "source": "ollama_request",
                    "data": {"text": request.text, "model": model}
                })
                
                # Registrar en el historial la respuesta de Ollama
                message_history.append({
                    "source": "ollama_response",
                    "data": {"generated_command": command_json}
                })
                
                # Enviar el comando a Unreal Engine
                command = command_json.get("command")
                parameters = command_json.get("parameters", {})
                
                # Registrar en el historial el comando a Unreal
                message_history.append({
                    "source": "client", 
                    "data": {"command": command, "parameters": parameters}
                })
                # Verificar la conexión con Unreal Engine
                unreal_conn = get_unreal_connection()
                if not unreal_conn or not unreal_conn.connected:
                    logger.error("No hay conexión con el servidor MCP")
                    raise HTTPException(status_code=400, detail="No hay conexión con el servidor MCP")
        
                # Enviar comando a Unreal
                logger.info(f"Enviando comando a Unreal: {command} con parámetros: {parameters}")
                mcp_response = unreal_conn.send_command(command, parameters)
                logger.info(f"Respuesta de Unreal recibida: {mcp_response}")
                
                # Registrar respuesta de Unreal
                message_history.append({
                    "source": "unreal", 
                    "data": mcp_response
                })
                
                return {
                    "status": "success",
                    "ollama_response": generated_text,
                    "command_extracted": command_json,
                    "mcp_response": mcp_response
                }
                
            except json.JSONDecodeError as e:
                error_msg = f"No se pudo parsear el JSON generado: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "ollama_response": generated_text
                }
        except Exception as e:
            error_msg = f"Error en la solicitud a Ollama: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except Exception as e:
        error_msg = f"Error inesperado procesando lenguaje natural: {str(e)}"
        logger.error(error_msg)
        logger.exception("Detalles del error:")
        
        # Add more context to help with debugging
        try:
            # If we have the generated_text, log it to help debugging
            if 'generated_text' in locals():
                logger.error(f"Texto generado que causó el error: {generated_text}")
            
            # If this is a JSON error, provide more context
            if isinstance(e, json.JSONDecodeError):
                logger.error(f"Error de JSON: posición {e.pos}, línea {e.lineno}, columna {e.colno}")
                logger.error(f"Documento JSON con problema: {e.doc[:100]}...")
        except:
            # Don't let our additional logging cause more errors
            pass
            
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/models")
async def get_ollama_models():
    """Obtiene la lista de modelos disponibles en Ollama"""
    try:
        client = ollama.Client(host=OLLAMA_API_BASE.replace('http://', ''))
        models_response = client.list()
        
        # Extraer los nombres de los modelos directamente de la respuesta
        model_names = []
        
        # La API de Ollama devuelve un objeto con un atributo 'models' que es una lista de objetos Model
        if hasattr(models_response, 'models'):
            for model_obj in models_response.models:
                if hasattr(model_obj, 'model'):
                    model_names.append(model_obj.model)
        
        if model_names:
            logger.info(f"Modelos disponibles: {model_names}")
            return {"status": "success", "models": model_names}
        else:
            logger.warning("No se encontraron modelos disponibles")
            return {"status": "error", "message": "No hay modelos disponibles", "models": []}
    except Exception as e:
        logger.error(f"Error obteniendo modelos de Ollama: {str(e)}")
        logger.exception("Detalles completos del error:")
        return {"status": "error", "message": str(e), "models": []}

@app.get("/api/debug-models")
async def debug_ollama_models():
    """Versión de debug para obtener los modelos disponibles en Ollama"""
    try:
        client = ollama.Client(host=OLLAMA_API_BASE.replace('http://', ''))
        models_response = client.list()
        
        # Devolver la respuesta completa para análisis
        return {
            "status": "debug",
            "raw_response": models_response,
            "response_type": str(type(models_response)),
            "has_models_key": 'models' in models_response if isinstance(models_response, dict) else False
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def check_ollama():
    """Verifica si Ollama está disponible"""
    try:
        client = ollama.Client(host=OLLAMA_API_BASE.replace('http://', ''))
        # La API de Ollama lanzará una excepción si no puede conectarse
        client.list()
        return True
    except Exception as e:
        logger.error(f"Error conectando con Ollama: {str(e)}")
        return False

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    logger.info(f"Iniciando servidor Ollama MCP Bridge en {HOST}:{PORT}")
    uvicorn.run("ollama_mcp_bridge:app", host=HOST, port=PORT, reload=True)