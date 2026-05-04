from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


BRANCH_CAPACITY = {
    "downtown": 18,
    "riverside": 12,
    "tech park": 10,
}

SPECIALS = {
    "downtown": "Truffle Mushroom Risotto",
    "riverside": "Lemon Herb Salmon Bowl",
    "tech park": "Spicy Tofu Power Bowl",
}

LOYALTY_POINTS = {
    "NB-1001": 240,
    "NB-1002": 85,
    "NB-4040": 0,
}

BOOKINGS: list[dict[str, Any]] = []


@dataclass
class ToolsRegistry:
    tools: list[StructuredTool]


class CheckAvailabilityInput(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format.")
    time: str = Field(description="Time in HH:MM 24h format.")
    branch: str = Field(description="Branch name: Downtown, Riverside, or Tech Park.")


class BookTableInput(CheckAvailabilityInput):
    name: str = Field(description="Customer full name.")


class GetTodaySpecialInput(BaseModel):
    branch: str = Field(description="Branch name: Downtown, Riverside, or Tech Park.")


class CheckLoyaltyPointsInput(BaseModel):
    user_id: str = Field(description="Loyalty user id. Example: NB-1001")


def _slot_key(date: str, time: str, branch: str) -> tuple[str, str, str]:
    return (date.strip(), time.strip(), branch.strip().lower())


def check_table_availability(date: str, time: str, branch: str) -> dict[str, Any]:
    datetime.strptime(date, "%Y-%m-%d")
    datetime.strptime(time, "%H:%M")
    b = branch.strip().lower()
    if b not in BRANCH_CAPACITY:
        return {"available": False, "reason": "Unknown branch."}

    existing = sum(1 for x in BOOKINGS if _slot_key(date, time, branch) == x["slot"])
    available_tables = BRANCH_CAPACITY[b] - existing
    return {"available": available_tables > 0, "available_tables": max(available_tables, 0)}


def book_table(name: str, date: str, time: str, branch: str) -> dict[str, Any]:
    availability = check_table_availability(date=date, time=time, branch=branch)
    if not availability.get("available"):
        return {"success": False, "message": "No table available for this slot."}

    booking_id = f"NB-RES-{len(BOOKINGS) + 1:04d}"
    BOOKINGS.append(
        {
            "booking_id": booking_id,
            "name": name,
            "slot": _slot_key(date, time, branch),
        }
    )
    return {"success": True, "booking_id": booking_id}


def get_today_special(branch: str) -> dict[str, Any]:
    b = branch.strip().lower()
    if b not in SPECIALS:
        return {"found": False, "message": "Unknown branch."}
    return {"found": True, "special": SPECIALS[b]}


def check_loyalty_points(user_id: str) -> dict[str, Any]:
    points = LOYALTY_POINTS.get(user_id.upper())
    if points is None:
        return {"found": False, "message": "User ID not found."}
    return {"found": True, "user_id": user_id.upper(), "points": points}


def build_operations_tools() -> ToolsRegistry:
    tools = [
        StructuredTool.from_function(
            name="check_table_availability",
            func=check_table_availability,
            description="Check if a branch has table availability for a given date/time.",
            args_schema=CheckAvailabilityInput,
        ),
        StructuredTool.from_function(
            name="book_table",
            func=book_table,
            description="Book a table for a customer after checking availability.",
            args_schema=BookTableInput,
        ),
        StructuredTool.from_function(
            name="get_today_special",
            func=get_today_special,
            description="Get the special dish of the day for a branch.",
            args_schema=GetTodaySpecialInput,
        ),
        StructuredTool.from_function(
            name="check_loyalty_points",
            func=check_loyalty_points,
            description="Check loyalty points by customer id.",
            args_schema=CheckLoyaltyPointsInput,
        ),
    ]
    return ToolsRegistry(tools=tools)
