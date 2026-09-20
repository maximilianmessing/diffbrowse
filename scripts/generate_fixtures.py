"""Generate rich, realistic recorded browser decision states for offline replay."""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List

from jev_ultrafast.action_candidates import flatten_actions
from jev_ultrafast.recorder import DecisionRecorder


def build_synthetic_scenarios() -> List[Dict[str, Any]]:
    """Build a comprehensive set of grounded browser tasks across multiple domains."""
    scenarios = []

    # 1. Flights Search Domain
    flight_cities = [
        ("Zurich", "London", "2026-09-20"),
        ("New York", "Paris", "2026-10-15"),
        ("Tokyo", "San Francisco", "2026-11-01"),
        ("Berlin", "Rome", "2026-08-25"),
        ("Sydney", "Singapore", "2026-12-10"),
    ]
    for orig, dest, date in flight_cities:
        goal = f"Find one-way flights from {orig} to {dest} on {date} for one adult in economy."
        # Step 1: Switch roundtrip to one-way
        actions_1 = [
            {"id": "e1", "kind": "click", "label": "Change ticket type · Round trip", "role": "button", "node": 10},
            {"id": "e2", "kind": "fill", "label": "Where from?", "role": "combobox", "value": orig, "node": 11},
            {"id": "e3", "kind": "fill", "label": "Where to?", "role": "combobox", "value": "", "node": 12},
            {"id": "e4", "kind": "click", "label": "Departure date", "role": "button", "node": 13},
            {"id": "e5", "kind": "click", "label": "Search flights", "role": "button", "node": 14},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"flights_{orig}_{dest}",
                "goal": goal,
                "url": "https://www.google.com/travel/flights?hl=en",
                "title": "Google Flights - Search Flights",
                "text": f"Compare flights to {dest}. Book flexible tickets with no change fees.",
                "actions": actions_1,
                "history": [],
                "expected_action": "e1",
                "operation": "CLICK",
                "target": "1",
            }
        )

        # Step 2: Select destination
        actions_2 = [
            {"id": "e1", "kind": "click", "label": "Change ticket type · One way", "role": "button", "node": 10},
            {"id": "e2", "kind": "fill", "label": "Where from?", "role": "combobox", "value": orig, "node": 11},
            {"id": "e3", "kind": "fill", "label": "Where to?", "role": "combobox", "value": "", "node": 12},
            {"id": "e4", "kind": "click", "label": "Departure date", "role": "button", "node": 13},
            {"id": "e5", "kind": "click", "label": "Search flights", "role": "button", "node": 14},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"flights_{orig}_{dest}",
                "goal": goal,
                "url": "https://www.google.com/travel/flights?hl=en",
                "title": "Google Flights - Search Flights",
                "text": "One-way selected. Enter destination city or airport.",
                "actions": actions_2,
                "history": [
                    {"step": 1, "action": "Change ticket type · One way", "choice": "e1", "page_changed": True}
                ],
                "expected_action": "e3",
                "operation": "TYPE_TEXT",
                "target": "3",
            }
        )

        # Step 3: Pick autocomplete destination airport
        actions_3 = [
            {
                "id": "e1",
                "kind": "click",
                "label": f"{dest} All airports ({dest[:3].upper()})",
                "role": "option",
                "node": 30,
            },
            {"id": "e2", "kind": "click", "label": f"{dest} International Airport", "role": "option", "node": 31},
            {"id": "e3", "kind": "click", "label": "Close suggestions", "role": "button", "node": 32},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"flights_{orig}_{dest}",
                "goal": goal,
                "url": "https://www.google.com/travel/flights?hl=en",
                "title": "Google Flights - Destination Selection",
                "text": f"Suggestions matching {dest}: Select an airport from the list.",
                "actions": actions_3,
                "history": [
                    {"step": 1, "action": "Change ticket type · One way", "choice": "e1", "page_changed": True},
                    {"step": 2, "action": "Where to?", "choice": "e3", "text": dest, "page_changed": True},
                ],
                "expected_action": "e1",
                "operation": "CLICK",
                "target": "1",
            }
        )

        # Step 4: Click Search after dates and route are filled
        actions_4 = [
            {"id": "e1", "kind": "fill", "label": "Where from?", "role": "combobox", "value": orig, "node": 11},
            {"id": "e2", "kind": "fill", "label": "Where to?", "role": "combobox", "value": dest, "node": 12},
            {"id": "e3", "kind": "click", "label": f"Departure: {date}", "role": "button", "node": 13},
            {"id": "e4", "kind": "click", "label": "Search flights", "role": "button", "node": 14},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"flights_{orig}_{dest}",
                "goal": goal,
                "url": "https://www.google.com/travel/flights?hl=en",
                "title": "Google Flights - Ready to Search",
                "text": f"Trip: One-way from {orig} to {dest} on {date}. Ready to search.",
                "actions": actions_4,
                "history": [
                    {"step": 1, "action": "Change ticket type", "choice": "e1", "page_changed": True},
                    {"step": 2, "action": "Where to?", "choice": "e2", "page_changed": True},
                    {"step": 3, "action": "Departure date", "choice": "e3", "page_changed": True},
                ],
                "expected_action": "e4",
                "operation": "CLICK",
                "target": "4",
            }
        )

        # Step 5: Flights results visible -> DONE
        actions_5 = [
            {
                "id": "e1",
                "kind": "click",
                "label": f"Nonstop flight Swiss Int $280 · Departing {date}",
                "role": "button",
                "node": 50,
            },
            {
                "id": "e2",
                "kind": "click",
                "label": f"1 stop flight British Airways $220 · Departing {date}",
                "role": "button",
                "node": 51,
            },
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"flights_{orig}_{dest}",
                "goal": goal + " Stop when matching flight options are visible.",
                "url": f"https://www.google.com/travel/flights/search?tfs={orig}_{dest}",
                "title": f"Flights from {orig} to {dest}",
                "text": f"Matching flight options: Nonstop flights from {orig} to {dest} on {date}. Options displayed.",
                "actions": actions_5,
                "history": [
                    {"step": 4, "action": "Search flights", "choice": "e4", "page_changed": True},
                ],
                "expected_action": "DONE",
                "operation": "DONE",
                "target": None,
            }
        )

    # 2. Hotel & Accommodation Filter Domain (Forma Fixture)
    cities = ["Lisbon", "Porto", "Barcelona", "Kyoto", "Berlin"]
    for city in cities:
        goal = (
            f"Use destination search and filters to find Design stays in {city} with Free cancellation, "
            "then open Casa Flora."
        )
        # Filter scenario
        actions_hotel = [
            {"id": "e1", "kind": "fill", "label": "Destination", "role": "searchbox", "value": "", "node": 101},
            {"id": "e2", "kind": "click", "label": "Search", "role": "button", "node": 102},
            {
                "id": "e3",
                "kind": "click",
                "label": "Free cancellation",
                "role": "checkbox",
                "checked": "false",
                "node": 103,
            },
            {
                "id": "e4",
                "kind": "select",
                "label": "Category → Design",
                "value": "Design",
                "role": "combobox",
                "node": 104,
                "current_value": "All",
            },
            {"id": "e5", "kind": "click", "label": "Instant book", "role": "checkbox", "checked": "false", "node": 105},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"hotel_{city}",
                "goal": goal,
                "url": "http://127.0.0.1:8766/fixture.html?scenario=travel",
                "title": "Forma · Find a place to slow down",
                "text": "Browse boutique stays. Filter by design, coastal, retreat. Destination search.",
                "actions": actions_hotel,
                "history": [],
                "expected_action": "e1",
                "operation": "TYPE_TEXT",
                "target": "1",
            }
        )

        # After destination entered, toggle free cancellation
        actions_hotel_2 = [
            {"id": "e1", "kind": "fill", "label": "Destination", "role": "searchbox", "value": city, "node": 101},
            {"id": "e2", "kind": "click", "label": "Search", "role": "button", "node": 102},
            {
                "id": "e3",
                "kind": "click",
                "label": "Free cancellation",
                "role": "checkbox",
                "checked": "false",
                "node": 103,
            },
            {
                "id": "e4",
                "kind": "select",
                "label": "Category → Design",
                "value": "Design",
                "role": "combobox",
                "node": 104,
                "current_value": "All",
            },
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"hotel_{city}",
                "goal": goal,
                "url": "http://127.0.0.1:8766/fixture.html?scenario=travel",
                "title": "Forma · Stays in " + city,
                "text": f"Destination: {city}. Filter options available below.",
                "actions": actions_hotel_2,
                "history": [{"step": 1, "action": "Destination", "choice": "e1", "text": city, "page_changed": True}],
                "expected_action": "e3",
                "operation": "CLICK",
                "target": "3",
            }
        )

        # After free cancellation, select Category -> Design
        actions_hotel_3 = [
            {"id": "e1", "kind": "fill", "label": "Destination", "role": "searchbox", "value": city, "node": 101},
            {
                "id": "e2",
                "kind": "click",
                "label": "Free cancellation",
                "role": "checkbox",
                "checked": "true",
                "node": 103,
            },
            {
                "id": "e3",
                "kind": "select",
                "label": "Category → Design",
                "value": "Design",
                "role": "combobox",
                "node": 104,
                "current_value": "All",
            },
            {"id": "e4", "kind": "click", "label": "Apply filters", "role": "button", "node": 105},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"hotel_{city}",
                "goal": goal,
                "url": "http://127.0.0.1:8766/fixture.html?scenario=travel",
                "title": "Forma · Stays in " + city,
                "text": f"Destination: {city}. Free cancellation enabled. Choose category dropdown.",
                "actions": actions_hotel_3,
                "history": [
                    {"step": 1, "action": "Destination", "choice": "e1", "page_changed": True},
                    {"step": 2, "action": "Free cancellation", "choice": "e2", "page_changed": True},
                ],
                "expected_action": "e3",
                "operation": "SELECT",
                "target": "3:1",
            }
        )

        # Open Casa Flora from filtered results
        actions_hotel_4 = [
            {
                "id": "e1",
                "kind": "click",
                "label": "Casa Flora · $180/night · Design stay",
                "role": "link",
                "node": 201,
            },
            {
                "id": "e2",
                "kind": "click",
                "label": "Villa Rosa · $220/night · Design stay",
                "role": "link",
                "node": 202,
            },
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"hotel_{city}",
                "goal": goal,
                "url": "http://127.0.0.1:8766/fixture.html?scenario=travel#filtered",
                "title": "Forma · Filtered Results",
                "text": "Results: Casa Flora (Design, Free cancellation). Villa Rosa.",
                "actions": actions_hotel_4,
                "history": [
                    {"step": 3, "action": "Category → Design", "choice": "e3", "page_changed": True},
                ],
                "expected_action": "e1",
                "operation": "CLICK",
                "target": "1",
            }
        )

    # 3. Wikipedia Research Domain
    topics = [
        (
            "Gödel's incompleteness theorems",
            "Find and open the Wikipedia article about Gödel’s incompleteness theorems.",
        ),
        ("Quantum computing", "Find and open the Wikipedia article about Quantum computing."),
        ("James Webb Space Telescope", "Find and open the Wikipedia article about James Webb Space Telescope."),
        ("Turing machine", "Find and open the Wikipedia article about Turing machine."),
        ("Neural network", "Find and open the Wikipedia article about Neural network."),
    ]
    for topic, goal in topics:
        # Step 1: Main page search box
        actions_wiki_1 = [
            {"id": "e1", "kind": "fill", "label": "Search Wikipedia", "role": "searchbox", "value": "", "node": 301},
            {"id": "e2", "kind": "click", "label": "Search", "role": "button", "node": 302},
            {"id": "e3", "kind": "click", "label": "Main page", "role": "link", "node": 303},
            {"id": "e4", "kind": "click", "label": "Current events", "role": "link", "node": 304},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"wiki_{topic[:10]}",
                "goal": goal,
                "url": "https://en.wikipedia.org/wiki/Main_Page",
                "title": "Wikipedia, the free encyclopedia",
                "text": "Welcome to Wikipedia. The free encyclopedia that anyone can edit. Search Wikipedia.",
                "actions": actions_wiki_1,
                "history": [],
                "expected_action": "e1",
                "operation": "TYPE_TEXT",
                "target": "1",
            }
        )

        # Step 2: Open matching article from search suggestions
        actions_wiki_2 = [
            {"id": "e1", "kind": "click", "label": f"{topic} (mathematical logic)", "role": "option", "node": 310},
            {"id": "e2", "kind": "click", "label": f"{topic} in popular culture", "role": "option", "node": 311},
            {
                "id": "e3",
                "kind": "click",
                "label": "Search for pages containing " + topic,
                "role": "option",
                "node": 312,
            },
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"wiki_{topic[:10]}",
                "goal": goal,
                "url": "https://en.wikipedia.org/wiki/Main_Page",
                "title": "Wikipedia Search Suggestions",
                "text": f"Search results matching '{topic}'. Select the primary topic article.",
                "actions": actions_wiki_2,
                "history": [
                    {"step": 1, "action": "Search Wikipedia", "choice": "e1", "text": topic, "page_changed": True}
                ],
                "expected_action": "e1",
                "operation": "CLICK",
                "target": "1",
            }
        )

        # Step 3: Article opened -> DONE
        actions_wiki_3 = [
            {"id": "e1", "kind": "click", "label": "Jump to content", "role": "link", "node": 401},
            {"id": "e2", "kind": "click", "label": "Edit this page", "role": "link", "node": 402},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
        scenarios.append(
            {
                "task_id": f"wiki_{topic[:10]}",
                "goal": goal,
                "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                "title": f"{topic} - Wikipedia",
                "text": f"{topic} is an encyclopedic topic. First proved by Kurt Gödel in 1931.",
                "actions": actions_wiki_3,
                "history": [
                    {"step": 2, "action": f"{topic} (mathematical logic)", "choice": "e1", "page_changed": True},
                ],
                "expected_action": "DONE",
                "operation": "DONE",
                "target": None,
            }
        )

    # 4. Large Action Space Scaling Variations (with 10, 25, 50, 100, 120 elements)
    for num_elems in [10, 25, 50, 100, 120]:
        gen_actions = []
        for i in range(1, num_elems + 1):
            if i == 5:
                gen_actions.append(
                    {"id": f"e{i}", "kind": "click", "label": "Confirm and Submit", "role": "button", "node": 1000 + i}
                )
            elif i % 4 == 0:
                gen_actions.append(
                    {
                        "id": f"e{i}",
                        "kind": "fill",
                        "label": f"Field {i}",
                        "role": "textbox",
                        "value": "",
                        "node": 1000 + i,
                    }
                )
            elif i % 3 == 0:
                gen_actions.append(
                    {
                        "id": f"e{i}",
                        "kind": "click",
                        "label": f"Option {i}",
                        "role": "checkbox",
                        "checked": "false",
                        "node": 1000 + i,
                    }
                )
            else:
                gen_actions.append(
                    {"id": f"e{i}", "kind": "click", "label": f"Navigation link {i}", "role": "link", "node": 1000 + i}
                )
        gen_actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560})
        gen_actions.append({"id": "wait", "kind": "wait", "label": "Wait"})

        scenarios.append(
            {
                "task_id": f"scaling_candidates_{num_elems}",
                "goal": "Submit the form by clicking Confirm and Submit.",
                "url": f"https://portal.example.test/form?size={num_elems}",
                "title": f"Configuration Form ({num_elems} controls)",
                "text": "Please complete the required options and click Confirm and Submit when ready.",
                "actions": gen_actions,
                "history": [],
                "expected_action": "e5",
                "operation": "CLICK",
                "target": "5",
            }
        )

    return scenarios


def generate_dataset(output_path: str, count: int = 500) -> None:
    recorder = DecisionRecorder(output_path)
    base_scenarios = build_synthetic_scenarios()
    print(f"Base scenarios count: {len(base_scenarios)}")

    recorded_count = 0
    rng = random.Random(42)

    while recorded_count < count:
        scenario = rng.choice(base_scenarios)
        # Create small variation
        goal = scenario["goal"]
        page = {
            "url": scenario["url"],
            "title": scenario["title"],
            "text": scenario["text"],
            "actions": scenario["actions"],
            "fingerprint": f"fp_{recorded_count:06d}",
        }
        history = scenario["history"]
        expected_act = scenario["expected_action"]
        op = scenario["operation"]
        target = scenario["target"]

        # Synthesize Jev choice & probabilities matching ground truth
        candidates = flatten_actions(page["actions"])
        probs = {}
        for c in candidates:
            if c.action_id == expected_act:
                probs[c.action_id] = round(rng.uniform(0.85, 0.98), 4)
            else:
                probs[c.action_id] = round(rng.uniform(0.001, 0.05), 4)
        total_p = sum(probs.values())
        probs = {k: round(v / total_p, 4) for k, v in probs.items()}

        jev_decision = {
            "choice": expected_act,
            "operation": op,
            "target": target,
            "confidence": probs[expected_act],
            "probabilities": probs,
            "operation_probabilities": {op: probs[expected_act]},
            "target_probabilities": {target: probs[expected_act]} if target else {},
            "latency_ms": rng.randint(45, 120),
            "model": "jev-latest",
        }

        recorder.record_decision(
            task_id=scenario["task_id"],
            goal=goal,
            page=page,
            history=history,
            decision=jev_decision,
            executed_action=expected_act,
            execution_result={"executed": expected_act},
            page_changed=True,
            eventual_task_success=True,
            decision_id=f"dec_{recorded_count:05d}",
        )
        recorded_count += 1

    print(f"Generated {recorded_count} decision states in {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="fixtures/recorded_decisions.jsonl")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    generate_dataset(args.output, args.count)


if __name__ == "__main__":
    main()
