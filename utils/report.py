from datetime import datetime

__all__ = ["generate_report"]


def _append_section(lines, title, value):
    if value is None or value == "":
        return

    lines.append("")
    lines.append(title)

    if isinstance(value, (list, tuple, set)):
        for item in value:
            lines.append(f"- {item}")
    else:
        for line in str(value).splitlines():
            lines.append(line)


def generate_report(modality, diagnosis, severity, confidence=None, details=None):
    lines = [
        "AI Brain Imaging Report",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Modality: {modality}",
        f"Finding: {diagnosis}",
        f"Severity: {severity}",
    ]

    if confidence is not None:
        try:
            lines.append(f"Confidence: {float(confidence):.2f}")
        except (TypeError, ValueError):
            lines.append(f"Confidence: {confidence}")

    if details:
        _append_section(lines, "Summary", details.get("summary"))
        _append_section(lines, "Plain-Language Meaning", details.get("meaning"))
        _append_section(lines, "Quality / QA Notes", details.get("qa_notes"))
        _append_section(lines, "Recommended Next Steps", details.get("next_steps"))
        _append_section(lines, "Lifestyle Guidance", details.get("lifestyle"))
        _append_section(lines, "Questions to Ask", details.get("questions"))
        _append_section(lines, "Important Note", details.get("note"))

    return "\n".join(lines)
