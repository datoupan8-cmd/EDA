"""OCR token roles used by the V3 component and pin front ends."""
from __future__ import annotations
from dataclasses import dataclass
import re
from .schema import Text

V4_DESIGNATOR_RE = re.compile(r"^((?:USB|IC|LED|SW|TP|CN|U|R|C|L|Q|D|J|P|Y|X|F)\d+[A-Za-z]?)(.*)$", re.I)
PREFIX_TYPE = {
    "R": "r", "RC": "r", "RES": "r", "RP": "r", "RT": "r", "RV": "r", "RVC": "r",
    "C": "c", "L": "l", "U": "box", "IC": "box",
    "Q": "mosfet", "D": "d", "LED": "led", "SW": "switch", "S": "switch",
    "X": "crystal", "Y": "crystal", "F": "fuse", "TP": "testpoint",
    "J": "pin", "P": "pin", "CN": "pin", "USB": "pin",
    "JP": "jumper", "M": "motor", "ANT": "ant", "AE": "ant", "BZ": "buzzer",
    "BT": "battary", "BAT": "battary", "B": "battary", "T": "transfomer",
    "H": "m3螺丝", "MH": "m3螺丝", "DS": "seg",
}
PREFIX_COMPATIBLE_TYPES = {
    "R": frozenset({"r"}), "RC": frozenset({"r"}), "RES": frozenset({"r"}),
    "RP": frozenset({"r"}), "RT": frozenset({"r"}), "RV": frozenset({"r"}), "RVC": frozenset({"r"}),
    "C": frozenset({"c"}), "L": frozenset({"l"}),
    "U": frozenset({"box", "block", "amp", "amp_3pin", "amp_5pin", "opt", "other"}),
    "IC": frozenset({"box", "block", "amp", "amp_3pin", "amp_5pin", "opt", "other"}),
    "Q": frozenset({"mosfet", "mosfet_npn", "mosfet_pnp", "bjt", "bjt_npn", "bjt_pnp"}),
    "D": frozenset({"d", "led", "esd"}), "LED": frozenset({"led"}),
    "SW": frozenset({"switch"}), "S": frozenset({"switch"}), "F": frozenset({"fuse"}),
    "TP": frozenset({"testpoint", "pin"}),
    "X": frozenset({"crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin"}),
    "Y": frozenset({"crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin"}),
    "J": frozenset({"pin", "circle_header"}), "P": frozenset({"pin", "circle_header"}),
    "CN": frozenset({"pin", "circle_header"}), "USB": frozenset({"pin", "circle_header"}),
    "JP": frozenset({"jumper"}), "M": frozenset({"motor"}), "ANT": frozenset({"ant"}),
    "AE": frozenset({"ant"}), "BZ": frozenset({"buzzer"}), "BT": frozenset({"battary"}),
    "BAT": frozenset({"battary"}), "B": frozenset({"battary"}), "T": frozenset({"transfomer"}),
    "H": frozenset({"m3螺丝"}), "MH": frozenset({"m3螺丝"}), "DS": frozenset({"seg"}),
}
_PREFIX_ALTERNATION = "|".join(sorted((re.escape(key) for key in PREFIX_TYPE), key=len, reverse=True))
DESIGNATOR_RE = re.compile(rf"^((?:{_PREFIX_ALTERNATION})\d+[A-Za-z]?)(.*)$", re.I)
VALUE_RE = re.compile(r"^[<>]?[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)\s*(?:[pnuµμmkKMG]?(?:F|H|Hz|Ω|ohm|V|A)?|R|MZ|MHz|kHz|GHz)$", re.I)
PIN_NUMBER_RE = re.compile(r"^(?:\d{1,4}|[A-Z]{1,2}\d{1,3}|EP|PAD)$", re.I)
SIGNAL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_./+\-]{1,40}$")

QUANTITY_PATTERNS = {
    "capacitance": re.compile(r"^[<>]?[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:p|n|u|m)?f$", re.I),
    "inductance": re.compile(r"^[<>]?[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:p|n|u|m)?h$", re.I),
    "resistance": re.compile(r"^[<>]?[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:(?:p|n|u|m|k|g)?(?:Ω|ohm)|r|k|m|g)?$", re.I),
    "voltage": re.compile(r"^[<>]?[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:u|m|k)?v$", re.I),
    "current": re.compile(r"^[<>]?[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:u|m|k)?a$", re.I),
    "frequency": re.compile(r"^[<>]?[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:k|m|g)?hz$", re.I),
}
COMPONENT_QUANTITY = {
    "r": "resistance", "c": "capacitance", "l": "inductance",
    "v": "voltage", "battary": "voltage", "motor": "voltage",
    "fuse": "current", "circle_header": "current",
    "crystal": "frequency", "crystal_2pin": "frequency",
    "crystal_3pin": "frequency", "crystal_4pin": "frequency",
}


@dataclass
class TokenRole:
    index: int
    token: Text
    role: str
    normalized: str
    designator: str | None = None
    component_type: str | None = None
    tail: str = ""
    confidence: float = 0.0
    reason: str = ""


def prefix_type(key: str) -> str:
    match = re.match(r"[A-Za-z]+", key)
    return PREFIX_TYPE.get(match.group(0).upper() if match else "", "other")


def compatible_types_for_designator(key: str) -> frozenset[str]:
    match = re.match(r"[A-Za-z]+", key)
    return PREFIX_COMPATIBLE_TYPES.get(match.group(0).upper() if match else "", frozenset())


def value_compatible(component_type: str, value: str, expanded: bool = True) -> bool:
    value = re.sub(r"\s+", "", str(value)).replace(",", ".").replace("µ", "u").replace("μ", "u")
    if not VALUE_RE.fullmatch(value):
        return False
    if expanded:
        quantity = COMPONENT_QUANTITY.get(component_type)
        return bool(quantity and QUANTITY_PATTERNS[quantity].fullmatch(value))
    upper = value.upper()
    if component_type == "r":
        return not upper.endswith(("F", "H", "V"))
    if component_type == "c":
        return upper.endswith(("F", "P", "N", "U")) or bool(re.fullmatch(r"\d{2,4}", upper))
    if component_type == "l":
        return upper.endswith("H")
    return True


def classify_tokens(texts: list[Text], large_boxes=(), expanded_rules: bool = True) -> list[TokenRole]:
    roles = []
    for index, token in enumerate(texts):
        raw = token.text.strip()
        normalized = re.sub(r"\s+", "", raw)
        upper = normalized.upper()
        match = (DESIGNATOR_RE if expanded_rules else V4_DESIGNATOR_RE).fullmatch(normalized)
        if match:
            key, tail = match.group(1).upper(), match.group(2).strip()
            typ = prefix_type(key)
            # Tokens at a detected box edge are commonly IC pin labels (R8/P9/N7),
            # whereas U/IC identifiers may legitimately sit inside or below a box.
            edge_box = False
            if typ != "box":
                cx, cy = token.center
                for box in large_boxes:
                    x1, y1, x2, y2 = box
                    near = x1 - 28 <= cx <= x2 + 28 and y1 - 28 <= cy <= y2 + 28
                    inside = x1 + 2 <= cx <= x2 - 2 and y1 + 2 <= cy <= y2 - 2
                    if near or inside:
                        edge_box = True
                        break
            if edge_box:
                roles.append(TokenRole(index, token, "PIN_NUMBER", upper, confidence=token.score, reason="near_large_box_boundary"))
            else:
                roles.append(TokenRole(index, token, "DESIGNATOR", upper, key, typ, tail, token.score, "designator_pattern"))
            continue
        if VALUE_RE.fullmatch(normalized):
            roles.append(TokenRole(index, token, "VALUE", upper, confidence=token.score, reason="value_pattern"))
        elif PIN_NUMBER_RE.fullmatch(normalized):
            roles.append(TokenRole(index, token, "PIN_NUMBER", upper, confidence=token.score, reason="pin_number_pattern"))
        elif SIGNAL_RE.fullmatch(normalized):
            role = "PIN_NAME" if "_" in normalized or upper.startswith(("GPIO", "VDD", "GND", "CLK", "DATA", "XTAL")) else "MODEL_TEXT"
            roles.append(TokenRole(index, token, role, upper, confidence=token.score, reason="signal_or_model_pattern"))
        else:
            roles.append(TokenRole(index, token, "OTHER", upper, confidence=token.score, reason="unclassified"))
    return roles
