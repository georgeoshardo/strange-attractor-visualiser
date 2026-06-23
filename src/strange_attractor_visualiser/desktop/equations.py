import re


_DOTTED_VARIABLES = {
    "x": "ẋ",
    "y": "ẏ",
    "z": "ż",
}

_SUPERSCRIPTS = {
    "2": "²",
    "3": "³",
}


def format_equation_text(equation_text: str) -> str:
    text = equation_text.strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    text = re.sub(r"\\+dot\{([^}]+)\}", _replace_dot, text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(
        r"\^([23])",
        lambda match: _SUPERSCRIPTS[match.group(1)],
        text,
    )
    text = text.replace("\\", "")

    lines = []
    for line in text.split(","):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\s*=\s*", " = ", line)
        line = re.sub(r"\s*([+-])\s*", r" \1 ", line)
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)

    return "\n".join(lines)


def _replace_dot(match: re.Match[str]) -> str:
    variable = match.group(1)
    return _DOTTED_VARIABLES.get(variable, f"{variable}\u0307")
