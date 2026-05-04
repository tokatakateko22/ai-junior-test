from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging

from src.config import OSS_MODEL_NAME
from src.rag_pipeline import NovaBiteRAGStore
from src.tools import ToolsRegistry, build_operations_tools


_LLM_CACHE: HuggingFacePipeline | None = None


def _must_get_llm() -> HuggingFacePipeline:
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(OSS_MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(OSS_MODEL_NAME, local_files_only=True)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    _LLM_CACHE = HuggingFacePipeline(pipeline=hf_pipeline)
    return _LLM_CACHE


def _render_prompt(messages: list[SystemMessage | HumanMessage | AIMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "SYSTEM"
        elif isinstance(m, HumanMessage):
            role = "USER"
        else:
            role = "ASSISTANT"
        lines.append(f"{role}: {m.content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _invoke_text(llm: HuggingFacePipeline, messages: list[SystemMessage | HumanMessage | AIMessage]) -> str:
    prompt = _render_prompt(messages)
    return str(llm.invoke(prompt)).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass
class AgentResult:
    answer: str
    metadata: dict[str, Any]


class RestaurantKnowledgeRAGAgent:
    def __init__(self) -> None:
        self.llm = _must_get_llm()
        self.store = NovaBiteRAGStore()
        self.store.load_or_create()

    def handle(self, user_query: str) -> AgentResult:
        docs = self.store.retrieve(user_query)
        if not docs:
            return AgentResult(
                answer=(
                    "I could not find verified information for that in NovaBite records. "
                    "Please ask staff for confirmation."
                ),
                metadata={"grounded": False, "sources": []},
            )

        response = self._extractive_answer(user_query, docs)
        sources = [f"{d.metadata.get('source')}#chunk-{d.metadata.get('chunk_id')}" for d in docs]
        return AgentResult(answer=str(response), metadata={"grounded": True, "sources": sources})

    @staticmethod
    def _extractive_answer(user_query: str, docs: list[Any]) -> str:
        q = _normalize(user_query)
        if any(k in q for k in ["recommend", "suggest", "recommendation"]):
            menu_items: list[str] = []
            for doc in docs:
                for raw_line in doc.page_content.splitlines():
                    line = raw_line.strip().lstrip("- ").strip()
                    if line.startswith("**") and "**:" in line:
                        menu_items.append(line)
            if "vegan" in q:
                vegan_items = [item for item in menu_items if "vegan" in item.lower()]
                if vegan_items:
                    return "Recommended vegan option: " + vegan_items[0]
            if menu_items:
                return "Recommended menu options: " + "; ".join(menu_items[:3])

        if "weekend" in q and "hour" in q:
            weekend_lines: list[str] = []
            for doc in docs:
                in_weekend = False
                for raw_line in doc.page_content.splitlines():
                    line = raw_line.strip()
                    if "weekends (saturday and sunday)" in line.lower():
                        in_weekend = True
                        continue
                    if in_weekend and line.startswith("### "):
                        break
                    if in_weekend and line.startswith("- ") and ":" in line:
                        weekend_lines.append(line.lstrip("- ").strip())
                if weekend_lines:
                    break
            if weekend_lines:
                return "Weekend hours: " + "; ".join(weekend_lines[:3]) + "."

        query_terms = [w for w in _normalize(user_query).split() if len(w) > 3]
        candidates: list[str] = []
        for doc in docs:
            for raw_line in doc.page_content.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                clean = line.lstrip("- ").strip()
                if not clean:
                    continue
                lowered = clean.lower()
                if any(t in lowered for t in query_terms):
                    candidates.append(clean)

        if not candidates:
            for doc in docs:
                for raw_line in doc.page_content.splitlines():
                    line = raw_line.strip().lstrip("- ").strip()
                    if line and not line.startswith("#"):
                        candidates.append(line)
                    if len(candidates) >= 2:
                        break
                if len(candidates) >= 2:
                    break

        selected = candidates[:3]
        if not selected:
            return (
                "I found related records but could not extract a precise answer. "
                "Please ask staff for confirmation."
            )
        return " ".join(selected)


class OperationsAgent:
    def __init__(self) -> None:
        self.llm = _must_get_llm()
        self.registry: ToolsRegistry = build_operations_tools()
        self.tools_by_name = {tool.name: tool for tool in self.registry.tools}

    def handle(self, user_query: str, hint: str = "") -> AgentResult:
        rule_route = self._rule_based_route(user_query)
        if rule_route is not None:
            return self._run_tool_and_format(user_query, rule_route)

        tool_names = ", ".join(self.tools_by_name.keys())
        extraction_prompt = [
            SystemMessage(
                content=(
                    "You are the operations tool router. Pick exactly one tool and arguments.\n"
                    f"Allowed tools: {tool_names}\n"
                    "Return strict JSON with keys: tool, args.\n"
                    "args must match the selected tool signature.\n"
                    "If unclear, return {\"tool\":\"clarify\",\"args\":{\"question\":\"...\"}}."
                )
            ),
            HumanMessage(content=f"Hint: {hint}\nUser request: {user_query}"),
        ]
        route_raw = _invoke_text(self.llm, extraction_prompt)
        route = self._safe_parse(route_raw)

        if route.get("tool") == "clarify":
            question = route.get("args", {}).get("question", "Could you clarify your request?")
            return AgentResult(answer=question, metadata={"tool_called": None})

        return self._run_tool_and_format(user_query, route)

    def _run_tool_and_format(self, user_query: str, route: dict[str, Any]) -> AgentResult:
        tool_name = route.get("tool", "")
        args = route.get("args", {})
        tool = self.tools_by_name.get(tool_name)
        if not tool:
            return AgentResult(
                answer="I need a bit more detail before I can run an operation.",
                metadata={"tool_called": None},
            )

        result = tool.invoke(args)
        natural_answer = self._format_tool_response(tool_name, args, result)
        return AgentResult(
            answer=str(natural_answer),
            metadata={"tool_called": tool_name, "tool_output": result},
        )

    @staticmethod
    def _safe_parse(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {"tool": "clarify", "args": {"question": "Could you share more details?"}}

    @staticmethod
    def _extract_branch(text: str) -> str | None:
        lowered = _normalize(text)
        if "downtown" in lowered:
            return "Downtown"
        if "riverside" in lowered:
            return "Riverside"
        if "tech park" in lowered:
            return "Tech Park"
        return None

    @staticmethod
    def _extract_date(text: str) -> str | None:
        match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if not match:
            return None
        date = match.group(0)
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return date
        except ValueError:
            return None

    @staticmethod
    def _extract_time(text: str) -> str | None:
        match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
        if not match:
            return None
        hh = int(match.group(1))
        mm = int(match.group(2))
        return f"{hh:02d}:{mm:02d}"

    @staticmethod
    def _extract_name(text: str) -> str | None:
        match = re.search(r"\bfor\s+([A-Za-z][A-Za-z\s'-]{1,40})", text, re.IGNORECASE)
        if not match:
            return None
        candidate = re.sub(r"\s+(on|at)\b.*$", "", match.group(1), flags=re.IGNORECASE).strip()
        return candidate if candidate else None

    @staticmethod
    def _extract_user_id(text: str) -> str | None:
        match = re.search(r"\bNB-\d{4}\b", text, re.IGNORECASE)
        return match.group(0).upper() if match else None

    def _rule_based_route(self, user_query: str) -> dict[str, Any] | None:
        lowered = _normalize(user_query)
        branch = self._extract_branch(user_query)
        date = self._extract_date(user_query)
        time = self._extract_time(user_query)

        if any(k in lowered for k in ["loyalty", "points"]):
            user_id = self._extract_user_id(user_query)
            if not user_id:
                return {"tool": "clarify", "args": {"question": "Please provide a loyalty user ID like NB-1001."}}
            return {"tool": "check_loyalty_points", "args": {"user_id": user_id}}

        if "special" in lowered:
            if not branch:
                return {"tool": "clarify", "args": {"question": "Which branch should I check for today's special?"}}
            return {"tool": "get_today_special", "args": {"branch": branch}}

        if any(k in lowered for k in ["book", "reserve", "reservation"]):
            name = self._extract_name(user_query)
            missing = []
            if not name:
                missing.append("name")
            if not date:
                missing.append("date (YYYY-MM-DD)")
            if not time:
                missing.append("time (HH:MM)")
            if not branch:
                missing.append("branch")
            if missing:
                return {
                    "tool": "clarify",
                    "args": {"question": f"To book a table, please provide: {', '.join(missing)}."},
                }
            return {"tool": "book_table", "args": {"name": name, "date": date, "time": time, "branch": branch}}

        if any(k in lowered for k in ["availability", "available", "free table", "table"]):
            if not (date and time and branch):
                return {
                    "tool": "clarify",
                    "args": {
                        "question": "Please provide date (YYYY-MM-DD), time (HH:MM), and branch to check availability."
                    },
                }
            return {"tool": "check_table_availability", "args": {"date": date, "time": time, "branch": branch}}

        return None

    @staticmethod
    def _format_tool_response(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
        if tool_name == "check_table_availability":
            if result.get("available"):
                return (
                    f"Yes, {args['branch']} has availability on {args['date']} at {args['time']}. "
                    f"Available tables: {result.get('available_tables', 0)}."
                )
            return f"No table is available for {args['branch']} on {args['date']} at {args['time']}."
        if tool_name == "book_table":
            if result.get("success"):
                return f"Your table is booked. Booking ID: {result.get('booking_id')}."
            return str(result.get("message", "Could not complete the booking."))
        if tool_name == "get_today_special":
            if result.get("found"):
                return f"Today's special at {args['branch']} is {result.get('special')}."
            return str(result.get("message", "I could not find today's special."))
        if tool_name == "check_loyalty_points":
            if result.get("found"):
                return f"User {result.get('user_id')} has {result.get('points')} loyalty points."
            return str(result.get("message", "I could not find that user ID."))
        return json.dumps(result)


class MainOrchestratorAgent:
    def __init__(self) -> None:
        self.llm = _must_get_llm()
        self.rag_agent = RestaurantKnowledgeRAGAgent()
        self.ops_agent = OperationsAgent()
        self.history: list[HumanMessage | AIMessage] = []

    def process(self, user_query: str) -> AgentResult:
        rule_intent = self._rule_based_intent(user_query)
        if rule_intent is not None:
            result = self._dispatch_by_intent(rule_intent, user_query, "rule-based")
            self._remember(user_query, result.answer)
            return result

        history = self.history
        classifier_prompt = [
            SystemMessage(
                content=(
                    "You are a routing orchestrator for NovaBite.\n"
                    "Classify user intent into one of: rag, operations, mixed, clarify.\n"
                    "Return strict JSON with keys: intent, reason, needs_clarification, clarification_question.\n"
                    "Use chat history for follow-up understanding."
                )
            ),
            *history[-6:],
            HumanMessage(content=user_query),
        ]
        route_raw = _invoke_text(self.llm, classifier_prompt)
        route = OperationsAgent._safe_parse(route_raw)

        intent = route.get("intent", "clarify")
        needs_clarification = bool(route.get("needs_clarification", False))
        if needs_clarification:
            answer = route.get("clarification_question", "Could you clarify what you need?")
            result = AgentResult(answer=answer, metadata={"intent": "clarify"})
            self._remember(user_query, result.answer)
            return result

        result = self._dispatch_by_intent(intent, user_query, str(route.get("reason", "")))

        validated = self._validate_output(user_query, result)
        self._remember(user_query, validated.answer)
        return validated

    def _dispatch_by_intent(self, intent: str, user_query: str, hint: str) -> AgentResult:
        if intent == "rag":
            return self.rag_agent.handle(user_query)
        if intent == "operations":
            return self.ops_agent.handle(user_query, hint=hint)
        if intent == "mixed":
            rag_part = self.rag_agent.handle(user_query)
            ops_part = self.ops_agent.handle(user_query, hint=hint or "Mixed request")
            merged = self._merge_answers(user_query, rag_part.answer, ops_part.answer)
            return AgentResult(
                answer=merged,
                metadata={
                    "intent": "mixed",
                    "rag": rag_part.metadata,
                    "operations": ops_part.metadata,
                },
            )
        return AgentResult(
            answer="Could you clarify whether you need menu/policy information or an operation like booking?",
            metadata={"intent": "clarify"},
        )

    @staticmethod
    def _rule_based_intent(user_query: str) -> str | None:
        lowered = _normalize(user_query)
        rag_terms = [
            "menu",
            "recommend",
            "suggest",
            "vegan",
            "allergen",
            "opening hours",
            "weekend hours",
            "host birthday",
            "premium catering",
            "grilled",
            "fried",
        ]
        ops_terms = [
            "book",
            "reserve",
            "reservation",
            "availability",
            "available table",
            "special",
            "loyalty",
            "points",
            "check table",
        ]

        rag_hit = any(term in lowered for term in rag_terms)
        ops_hit = any(term in lowered for term in ops_terms)
        if rag_hit and ops_hit:
            return "mixed"
        if rag_hit:
            return "rag"
        if ops_hit:
            return "operations"
        return None

    def _merge_answers(self, query: str, rag_answer: str, ops_answer: str) -> str:
        prompt = [
            SystemMessage(
                content=(
                    "Merge two sub-agent outputs into one concise response. "
                    "Preserve factual constraints, no invented details."
                )
            ),
            HumanMessage(
                content=(
                    f"User query: {query}\n"
                    f"Knowledge answer: {rag_answer}\n"
                    f"Operations answer: {ops_answer}"
                )
            ),
        ]
        return _invoke_text(self.llm, prompt)

    def _validate_output(self, query: str, result: AgentResult) -> AgentResult:
        # Keep deterministic outputs stable for tiny local models.
        if result.metadata.get("tool_called"):
            result.metadata["validated"] = True
            return result

        guard_prompt = [
            SystemMessage(
                content=(
                    "You are an output validator. "
                    "If response contains unverified or speculative claims, rewrite safely."
                )
            ),
            HumanMessage(
                content=(
                    f"User query: {query}\n"
                    f"Candidate response: {result.answer}\n"
                    "Return strict JSON: {\"safe_answer\": \"...\", \"changed\": true|false}"
                )
            ),
        ]
        raw = _invoke_text(self.llm, guard_prompt)
        parsed = OperationsAgent._safe_parse(raw)
        safe = parsed.get("safe_answer")
        if isinstance(safe, str) and safe.strip():
            result.answer = safe
        result.metadata["validated"] = True
        return result

    def _remember(self, user_query: str, answer: str) -> None:
        self.history.append(HumanMessage(content=user_query))
        self.history.append(AIMessage(content=answer))
