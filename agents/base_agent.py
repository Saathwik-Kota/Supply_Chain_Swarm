"""
Base ReAct Agent class for the Emergency Response Supply Chain Swarm.

Implements the ReAct (Reasoning + Acting) loop:
  1. Thought: Agent reasons about current state and task objective.
  2. Action: Agent chooses specific tools with typed arguments.
  3. Observation: Agent observes tool execution output and latency.
  4. Synthesis: Agent reflects and decides whether further actions or state updates are needed.
"""

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from llm.client import LLMClient, get_llm_client
from llm.schemas import get_agent_tool_schemas
from observability.logger import log_agent_event
from swarm.state import AgentLog, SwarmState

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Autonomous ReAct agent equipped with an LLM brain, system prompt, and bound tools.
    """

    def __init__(
        self,
        name: str,
        role_key: str,
        system_prompt: str,
        tool_names: List[str],
        tool_registry: Dict[str, Callable],
        llm_client: Optional[LLMClient] = None,
        max_react_steps: int = 4,
    ):
        self.name = name
        self.role_key = role_key
        self.system_prompt = system_prompt
        self.tool_names = tool_names
        self.tool_registry = tool_registry
        self.llm_client = llm_client or get_llm_client()
        self.max_react_steps = max_react_steps
        self.tool_schemas = get_agent_tool_schemas(tool_names)

    def log(
        self,
        state: SwarmState,
        step: str,
        message: str,
        data: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Structured logging to SwarmState."""
        data_dict = dict(data) if data is not None else None
        entry = AgentLog(
            agent=self.name,
            step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=message,
            data=data_dict,
            error=error,
            latency_ms=latency_ms,
        )
        state["agent_logs"].append(entry)
        logger.info(f"[{self.name}] ({step}): {message}")
        log_agent_event(
            agent=self.name,
            step=step,
            message=message,
            data=data_dict,
            error=error,
            latency_ms=latency_ms,
        )

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Any, float]:
        """Execute a tool function by name and measure execution latency."""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Tool '{tool_name}' not found in {self.name}'s registry")

        func = self.tool_registry[tool_name]
        start_time = time.time()
        try:
            result = func(**args)
        except TypeError as e:
            # Fallback if arguments differ slightly from schema
            logger.warning(f"Tool {tool_name} argument mismatch ({e}); attempting no-arg invocation")
            result = func()

        latency_ms = (time.time() - start_time) * 1000
        return result, round(latency_ms, 2)

    def build_task_context(self, state: SwarmState) -> str:
        """Override in subclasses to provide domain-specific context to the LLM."""
        return (
            f"Destination: {state.get('destination_name')} "
            f"({state.get('destination_lat')}, {state.get('destination_lon')}). "
            f"Requested items: {state.get('requested_items')}."
        )

    def run(self, state: SwarmState) -> SwarmState:
        """
        Execute the autonomous ReAct reasoning and tool-calling cycle.
        """
        state["current_agent"] = self.name
        self.log(state, "start", f"Activated {self.name} with {len(self.tool_names)} registered tools.")

        start_time = time.time()
        if "llm_reasoning" not in state or state["llm_reasoning"] is None:
            state["llm_reasoning"] = {}
        state["llm_reasoning"][self.role_key] = []

        history: List[Dict[str, Any]] = []
        task_context = self.build_task_context(state)

        for step_idx in range(self.max_react_steps):
            try:
                # 1. Query LLM Brain for next Thought + Tool Calls
                llm_response = self.llm_client.call_agent_step(
                    system_prompt=self.system_prompt,
                    task_context=task_context,
                    tool_schemas=self.tool_schemas,
                    history=history,
                    agent_role=self.role_key,
                )

                thought = llm_response.get("thought", "")
                tool_calls = llm_response.get("tool_calls", [])
                provider = llm_response.get("provider_used", "LLM")

                # Record LLM thought in state and logs
                reasoning_entry = {
                    "step": step_idx + 1,
                    "thought": thought,
                    "tool_calls": [tc["name"] for tc in tool_calls],
                    "provider": provider,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                state["llm_reasoning"][self.role_key].append(reasoning_entry)

                self.log(
                    state,
                    f"react_thought_step_{step_idx+1}",
                    f"🧠 {thought}",
                    data={"provider": provider, "tools_invoked": [tc["name"] for tc in tool_calls]},
                )

                # If no tools invoked, agent has concluded its reasoning
                if not tool_calls:
                    break

                # 2. Execute requested tools
                step_observations = []
                for tc in tool_calls:
                    tool_name = tc.get("name")
                    tool_args = tc.get("arguments", {})

                    self.log(
                        state,
                        "tool_invoke",
                        f"🔧 Invoking tool `{tool_name}` with args: {tool_args}",
                    )

                    try:
                        tool_result, tool_lat = self.execute_tool(tool_name, tool_args)
                        self.log(
                            state,
                            "tool_result",
                            f"✅ Tool `{tool_name}` completed in {tool_lat:.1f}ms",
                            data=tool_result if isinstance(tool_result, dict) else {"result": str(tool_result)[:300]},
                            latency_ms=tool_lat,
                        )
                        # Process and write domain outputs into SwarmState
                        self.process_tool_result(state, tool_name, tool_result, tool_args)
                        step_observations.append({
                            "tool": tool_name,
                            "output": tool_result,
                        })
                    except Exception as e:
                        error_str = f"Tool `{tool_name}` execution error: {str(e)}"
                        self.log(state, "tool_error", error_str, error=traceback.format_exc())
                        step_observations.append({
                            "tool": tool_name,
                            "error": str(e),
                        })

                # Append to conversation history for next ReAct loop
                history.append({
                    "role": "assistant",
                    "content": f"THOUGHT: {thought}\nACTION: Invoked {[tc['name'] for tc in tool_calls]}",
                })
                history.append({
                    "role": "user",
                    "content": f"OBSERVATION: {json.dumps(step_observations, default=str)[:1000]}",
                })

            except Exception as e:
                err_msg = f"ReAct step {step_idx+1} failed: {str(e)}"
                self.log(state, "react_error", err_msg, error=traceback.format_exc())
                state["errors"].append({"agent": self.name, "error": err_msg})
                break

        # Post-execution wrap up
        self.post_run_cleanup(state)
        total_latency = round((time.time() - start_time) * 1000, 2)
        self.log(
            state,
            "complete",
            f"🎯 {self.name} completed mission execution.",
            latency_ms=total_latency,
        )
        return state

    def process_tool_result(
        self,
        state: SwarmState,
        tool_name: str,
        tool_result: Any,
        tool_args: Dict[str, Any],
    ) -> None:
        """Subclasses override this to update state keys with tool outputs."""
        pass

    def post_run_cleanup(self, state: SwarmState) -> None:
        """Subclasses can enforce consistency before handing off state."""
        pass

