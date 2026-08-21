"""Designed event-group taxonomy over ESC-50.

The 50 ESC-50 categories are mapped into six project-specific event groups so
the frozen probe can evaluate event-group accessibility from compact audio
representation interfaces.

This is an exploratory mapping, not a canonical event taxonomy or evidence of
embodied-agent utility. Each of the 50 categories maps to exactly one group.
"""
from __future__ import annotations

# Canonical ESC-50 category strings (as they appear in meta/esc50.csv).
ESC50_CATEGORIES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
    "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
    "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
    "glass_breaking",
    "helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells",
    "airplane", "fireworks", "hand_saw",
]

# group -> list of ESC-50 categories
EVENT_GROUPS = {
    # human presence and activity sounds
    "human_activity": [
        "crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring",
    ],
    # liquids, pouring, flow, and related environmental sounds
    "water_liquid": [
        "water_drops", "pouring_water", "toilet_flush", "drinking_sipping",
        "sea_waves", "rain",
    ],
    # sharp transient impacts / contact events
    "impact_contact": [
        "glass_breaking", "mouse_click", "keyboard_typing", "clock_tick",
        "hand_saw", "fireworks",
    ],
    # doors / containers being manipulated
    "object_interaction": [
        "door_wood_knock", "door_wood_creaks", "can_opening",
    ],
    # machines, appliances, vehicles, and related mechanical sounds
    "appliance_mechanical": [
        "washing_machine", "vacuum_cleaner", "clock_alarm", "engine", "chainsaw",
        "helicopter", "airplane", "train", "siren", "car_horn", "church_bells",
    ],
    # ambient, natural, and animal background sounds
    "ambient_natural": [
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep",
        "crow", "crackling_fire", "crickets", "chirping_birds", "wind", "thunderstorm",
    ],
}

GROUP_NAMES = list(EVENT_GROUPS.keys())

# inverse map: category -> group
CATEGORY_TO_GROUP = {}
for _g, _cats in EVENT_GROUPS.items():
    for _c in _cats:
        CATEGORY_TO_GROUP[_c] = _g


def group_of(category: str) -> str:
    return CATEGORY_TO_GROUP[category]


def group_index(category: str) -> int:
    return GROUP_NAMES.index(CATEGORY_TO_GROUP[category])


def validate() -> None:
    """Assert the taxonomy covers all 50 categories exactly once."""
    mapped = sorted(CATEGORY_TO_GROUP.keys())
    canon = sorted(ESC50_CATEGORIES)
    assert mapped == canon, (
        "Taxonomy mismatch.\n"
        f"missing={sorted(set(canon) - set(mapped))}\n"
        f"extra={sorted(set(mapped) - set(canon))}"
    )
    total = sum(len(v) for v in EVENT_GROUPS.values())
    assert total == 50, f"expected 50 categories, mapped {total}"


if __name__ == "__main__":
    validate()
    print("OK: 50 ESC-50 categories ->", len(GROUP_NAMES), "designed event groups")
    for g, cats in EVENT_GROUPS.items():
        print(f"  {g:22s} ({len(cats):2d}): {', '.join(cats)}")
