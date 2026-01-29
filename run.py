from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import random

# === 1. Data Structures ===
class NPCState(BaseModel):
    id: str
    name: str
    money: int = 0
    food: int = 50
    rest: int = 50
    location: str = "Home"
    current_hour: int = 0
    daily_hours_worked: int = 0
    action_history: List[str] = []
    personality: str = "balanced"  # workaholic, lazy, foodie

class Action(BaseModel):
    npc_id: str
    action: str
    place: str
    duration: int = 1
    reason: str
    confidence: float

class Place(BaseModel):
    name: str
    actions: List[Dict]

# === 2. World ===
WORLD = {
    "food_store": Place(
        name="Food Store",
        actions=[
            {"name": "eating", "effects": {"food": +100, "money": -60}, "duration": 1},
            {"name": "parttime", "effects": {"money": +20, "food": -5, "rest": -5}, "duration": 1},
            {"name": "shopping", "effects": {"money": -5}, "duration": 1}  # browse
        ]
    ),
    "wooden_factory": Place(
        name="Wooden Factory",
        actions=[
            {"name": "fulltime", "effects": {"money": +20, "food": -5, "rest": -5}, "duration": 1},
            {"name": "chat", "effects": {"rest": +5}, "duration": 1}  # chat
        ]
    ),
    "home": Place(
        name="Home",
        actions=[
            {"name": "sleep", "effects": {"rest": +40}, "duration": 1},
            {"name": "relax", "effects": {"rest": +15}, "duration": 1}  # relax
        ]
    ),
    "park": Place(
        name="Park",
        actions=[
            {"name": "walk", "effects": {"rest": -0, "food": -5}, "duration": 1}  # walk
        ]
    )
}

# === 3. Hybrid Decision: Hard Rules + AI ===
def create_npc_agent(npc: NPCState):
    """Create NPC Agent (with personality)"""
    personality_traits = {
        "workaholic": "Loves work, prioritizes earning money, less rest",
        "lazy": "Prefers relaxation, avoids work, more rest",
        "foodie": "Loves food, eats frequently",
        "balanced": "Balanced life, work and rest equally"
    }
    
    return Agent(
        role=f"{npc.name} (Personality: {npc.personality})",
        goal=f"Live according to personality, Current: 💰{npc.money} 🍔{npc.food} 😴{npc.rest}",
        backstory=f"""{npc.name}'s character: {personality_traits.get(npc.personality, 'ordinary person')}
Within safe range (food > 30, rest > 40), freely choose actions.
Can work for money, eat, sleep, or relax and browse.""",
        llm="ollama/gemma3:4b",
        verbose=False,
        allow_delegation=False
    )

def hybrid_decision(npc: NPCState) -> Optional[Action]:
    """Hybrid decision: Check Hard Rules first, otherwise return None (let AI decide)"""
    
    # === HARD RULES (survival baseline, cannot violate) ===
    if npc.food < 15:
        return Action(
            npc_id=npc.id, action="eating", place="Food Store", duration=1,
            reason=f"🚨 Survival baseline: food={npc.food} < 15, forced eating", confidence=1.0
        )
    
    if npc.rest < 20:
        return Action(
            npc_id=npc.id, action="sleep", place="Home", duration=1,
            reason=f"🚨 Survival baseline: rest={npc.rest} < 20, forced sleep", confidence=1.0
        )
    
    if npc.money < 5:  # bankruptcy protection
        return Action(
            npc_id=npc.id, action="parttime", place="Food Store", duration=1,
            reason=f"🚨 Bankruptcy protection: money={npc.money} < 5, emergency work", confidence=1.0
        )
    
    # === Within safe range → return None, let AI decide freely ===
    return None

def create_ai_decision_task(npc: NPCState, hour: int):
    """Create AI decision task (free choice)"""
    
    # Available actions
    available_actions = []
    for place_key, place in WORLD.items():
        for action in place.actions:
            available_actions.append({
                "place": place.name,
                "action": action["name"],
                "effects": action["effects"],
                "duration": action["duration"]
            })
    
    description = f"""# Hour {hour}: {npc.name}'s Free Time

## Current Status (Safe Range)
- 💰 Money: {npc.money}
- 🍔 Food: {npc.food}
- 😴 Rest: {npc.rest}
- 📍 Location: {npc.location}
- ⏰ Work Hours: {npc.daily_hours_worked}h

## Your Personality: {npc.personality}
- workaholic: Prioritize work and earning money
- lazy: Prioritize rest and relaxation
- foodie: Eat food frequently
- balanced: Balance all activities

## Available Actions (Free Choice)
{json.dumps(available_actions, indent=2, ensure_ascii=False)}

## Decision Requirements
Based on your personality and current status, choose an action you **want to do**.
You don't always have to work, you can:
- Walk in the park (walk)
- Relax at home (relax)
- Chat at the factory (chat)
- Browse at the store (shopping)
- Or any other action

Return JSON:
{{
  "npc_id": "{npc.id}",
  "action": "walk",
  "place": "Park",
  "duration": 1,
  "reason": "Explain why you chose this based on personality (within 30 words)",
  "confidence": 0.85
}}
"""
    
    agent = create_npc_agent(npc)
    return Task(
        description=description,
        expected_output="JSON action decision",
        agent=agent,
        output_pydantic=Action
    )

# === 4. State Update ===
def apply_action(npc: NPCState, action: Action) -> NPCState:
    """Execute action"""
    place_key = action.place.lower().replace(" ", "_")
    place = WORLD.get(place_key)
    
    if not place:
        print(f"❌ {action.place} does not exist")
        return npc
    
    place_action = None
    for act in place.actions:
        if act["name"] == action.action or act["name"] in action.action:
            place_action = act
            break
    
    if not place_action:
        print(f"❌ {action.action} does not exist at {action.place}")
        return npc
    
    # 應用效果
    for stat, change in place_action["effects"].items():
        current = getattr(npc, stat)
        new_value = max(0, min(200, current + change))
        setattr(npc, stat, new_value)
    
    npc.location = action.place
    
    # Only count work hours
    if place_action["name"] in ["fulltime", "parttime"]:
        npc.daily_hours_worked += place_action.get("duration", 1)
    
    npc.action_history.append(f"H{npc.current_hour}:{place_action['name']}")
    
    print(f"✅ {npc.name}: {place_action['name']} @ {action.place} | 💰{npc.money} 🍔{npc.food} 😴{npc.rest}")
    return npc

# === 5. Main Loop ===
def simulate_day(npcs: List[NPCState], hours: int = 24):
    """Simulate one day (hybrid mode)"""
    print("🌍 NPC Sim Start (Hard Rules + AI Freedom)\n")
    
    for hour in range(hours):
        print(f"\n⏰ === Hour {hour} ===")
        
        if hour == 0:
            for npc in npcs:
                npc.daily_hours_worked = 0
                print(f"🌅 {npc.name} ({npc.personality}) new day")
        
        # System metabolism
        for npc in npcs:
            old_food, old_rest = npc.food, npc.rest
            npc.food = max(0, npc.food - 5)
            npc.rest = max(0, npc.rest - 5)
            print(f"⏳ {npc.name} metabolism: 🍔{old_food}→{npc.food} 😴{old_rest}→{npc.rest}")
        
        # Decision
        for npc in npcs:
            npc.current_hour = hour
            
            # 1. Check Hard Rules
            forced_action = hybrid_decision(npc)
            
            if forced_action:
                # Force execute (survival baseline)
                print(f"🤖 {npc.name} [FORCED]: {forced_action.action} - {forced_action.reason}")
                npc = apply_action(npc, forced_action)
            else:
                # 2. AI free decision
                try:
                    task = create_ai_decision_task(npc, hour)
                    crew = Crew(
                        agents=[task.agent],
                        tasks=[task],
                        process=Process.sequential,
                        verbose=False
                    )
                    
                    result = crew.kickoff()
                    action = result.pydantic if hasattr(result, 'pydantic') else result
                    
                    print(f"🤖 {npc.name} [AI]: {action.action} - {action.reason}")
                    npc = apply_action(npc, action)
                    
                except Exception as e:
                    print(f"❌ {npc.name} AI failed: {e}, random action")
                    # Fallback: random choice
                    fallback_actions = ["sleep", "eating", "relax", "walk"]
                    random_action = random.choice(fallback_actions)
                    action = Action(
                        npc_id=npc.id, action=random_action, 
                        place="Home" if random_action in ["sleep", "relax"] else "Park",
                        duration=1, reason="Random choice", confidence=0.5
                    )
                    npc = apply_action(npc, action)
        
        # Statistics
        if hour % 6 == 5:
            print(f"\n📊 === 6h Statistics ===")
            for npc in npcs:
                s = "💀" if npc.food < 10 or npc.rest < 10 else "✅"
                print(f"  {npc.name}: 💰{npc.money} 🍔{npc.food} 😴{npc.rest} Work{npc.daily_hours_worked}h | {s}")
    
    print(f"\n🌙 === Day End ===")
    for npc in npcs:
        s = "✅" if npc.food > 0 and npc.rest > 0 else "💀"
        w = "💎" if npc.money > 150 else "💸" if npc.money < 30 else "💰"
        diversity = len(set(npc.action_history)) / max(1, len(npc.action_history))
        print(f"  {npc.name} ({npc.personality}): 💰{npc.money} 🍔{npc.food} 😴{npc.rest} | {s} {w}")
        print(f"    Action diversity: {diversity:.2f} | Work {npc.daily_hours_worked}h")

# === 6. Start ===
if __name__ == "__main__":
    npcs = [
        NPCState(id="npc1", name="Max", money=50, food=50, rest=50, personality="workaholic"),
        NPCState(id="npc2", name="Alice", money=50, food=50, rest=50, personality="lazy"),
        NPCState(id="npc3", name="Bob", money=50, food=50, rest=50, personality="foodie"),
    ]
    
    simulate_day(npcs, hours=24)
    
    print("\n📊 === Detailed Summary ===")
    for npc in npcs:
        print(f"\n{npc.name} ({npc.personality}):")
        print(f"  Final: 💰{npc.money} 🍔{npc.food} 😴{npc.rest}")
        print(f"  Path: {' → '.join(npc.action_history[-12:])}")
