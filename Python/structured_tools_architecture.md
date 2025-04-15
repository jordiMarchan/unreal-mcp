# Arquitectura recomendada: Structured Tools y Agentes Especializados para MCP en CrewAI

## Introducción

En sistemas complejos como la integración de CrewAI con Unreal Engine vía MCP, la robustez y claridad en la definición de herramientas (tools) es clave para evitar errores de formato, facilitar la validación y mejorar la experiencia de los agentes LLM. Esta guía describe una arquitectura basada en:

- Uso de `StructuredTool` de CrewAI para cada comando MCP relevante.
- División de agentes especializados por dominio (ej: actores, blueprints, UMG).
- Consejos y pseudocódigo para implementar este enfoque.

---

## Ventajas del enfoque StructuredTool + Agentes Especializados

- **Validación automática:** Cada herramienta define sus parámetros y tipos exactos usando Pydantic, evitando errores de formato y parámetros faltantes.
- **Interfaz clara para el LLM:** El agente ve solo las herramientas relevantes para su dominio, con firmas explícitas.
- **Mantenimiento modular:** Añadir o modificar comandos es sencillo y localizado.
- **Escalabilidad:** Permite crecer el sistema sin sobrecargar a un solo agente con demasiadas herramientas.

---

## Ejemplo de organización

### 1. Definir herramientas estructuradas (StructuredTool)

```python
from crewai.tools import StructuredTool
from pydantic import BaseModel

class CreateBlueprintArgs(BaseModel):
    name: str
    parent_class: str

class CreateBlueprintTool(StructuredTool):
    name = 'create_blueprint'
    description = 'Crea un blueprint en Unreal Engine.'
    args_schema = CreateBlueprintArgs

    def _run(self, name: str, parent_class: str) -> str:
        """Crea un blueprint. Ejemplo: create_blueprint(name='BP_Test', parent_class='Character')"""
        # Lógica para llamar al MCP aquí
        ...
```

### 2. Crear agentes especializados

```python
from crewai import Agent

actor_tools = [spawn_actor_tool, delete_actor_tool, ...]
blueprint_tools = [create_blueprint_tool, add_component_tool, ...]

actor_agent = Agent(
    role='Actor Agent',
    goal='Gestionar actores en Unreal Engine',
    tools=actor_tools,
    ...
)

blueprint_agent = Agent(
    role='Blueprint Agent',
    goal='Gestionar blueprints en Unreal Engine',
    tools=blueprint_tools,
    ...
)
```

### 3. Orquestador (opcional)

Un agente principal puede analizar la instrucción del usuario y delegar subtareas a los agentes especializados.

```python
orchestrator = Agent(
    role='Orquestador',
    goal='Dividir instrucciones complejas y delegar a agentes especializados',
    tools=[actor_agent, blueprint_agent, ...],
    ...
)
```

---

## Orquestador personalizado (Manager Agent)

CrewAI permite definir un agente orquestador personalizado (manager agent) que puede analizar la instrucción del usuario, dividirla en subtareas y delegar cada una al agente especializado correspondiente. Esto maximiza la flexibilidad y el control en sistemas complejos.

### Ventajas
- Centraliza la toma de decisiones y la delegación de tareas.
- Permite lógica avanzada de coordinación, validaciones cruzadas y manejo de errores globales.
- Facilita la integración de lógica de negocio y presentación de resultados finales.

### Ejemplo de integración

```python
from crewai import Agent, Crew, Process

# Definir agentes especializados
actor_agent = Agent(...)
blueprint_agent = Agent(...)
umg_agent = Agent(...)

# Definir el manager agent
manager = Agent(
    role="Project Manager",
    goal="Coordinar y supervisar la ejecución de tareas MCP en Unreal Engine",
    backstory="Eres responsable de delegar y supervisar tareas entre agentes expertos en diferentes dominios de Unreal Engine.",
    allow_delegation=True,
)

# Crear el crew con manager personalizado
crew = Crew(
    agents=[actor_agent, blueprint_agent, umg_agent],
    tasks=[...],  # Tareas de alto nivel
    manager_agent=manager,
    process=Process.hierarchical,
)

result = crew.kickoff(inputs={"user_prompt": "Crea un personaje y añade un UI Widget"})
```

### Consejos para el orquestador
- Sé explícito en el `backstory` y la descripción del manager sobre cómo debe decidir a qué agente delegar cada subtarea.
- Usa nombres de herramientas exactos y consistentes en los prompts.
- El manager puede supervisar el progreso, reasignar tareas o manejar errores globales.

---

## Consejos prácticos

- **Define los parámetros de cada herramienta con Pydantic y docstrings claros.**
- **Evita el uso de tipos ambiguos (`Any`, `object`, `Optional`, `Union`).**
- **Agrupa herramientas por dominio funcional y asígnalas solo al agente correspondiente.**
- **En los prompts de los agentes, usa siempre el nombre exacto de la herramienta entre comillas simples.**
- **Incluye ejemplos de uso en los docstrings usando notación Python (`dict()`).**
- **Si usas un orquestador, asegúrate de que pueda decidir correctamente a qué agente delegar cada subtarea.**

---

## Pseudocódigo de flujo

```python
# 1. El usuario da una instrucción compleja
user_prompt = "Crea un blueprint de personaje y añade un actor en la escena."

# 2. El orquestador analiza y divide la tarea
plan = orchestrator.plan(user_prompt)

# 3. El orquestador delega subtareas
for subtask in plan:
    if subtask.tipo == 'blueprint':
        blueprint_agent.execute(subtask)
    elif subtask.tipo == 'actor':
        actor_agent.execute(subtask)
    ...

# 4. Cada agente usa solo sus StructuredTools
# 5. Se recopilan y presentan los resultados
```

---

## Conclusión

Este enfoque modular y estructurado maximiza la robustez, la claridad y la mantenibilidad de la integración CrewAI + MCP. Es especialmente recomendable para proyectos grandes o equipos que busquen minimizar errores y facilitar la evolución del sistema.
