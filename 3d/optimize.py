# optimize.py

# cot optimization
def cot_tips() -> str:
    """Chain-of-Thought tips for 3D structure reasoning."""
    return (
        "--------------------------------------------------\n"
        "Thinking tips (keep them in mind only):\n"
        "You may proceed step-by-step:\n"
        "1. Set Layer 1 = top-view (copy 0/1 exactly).\n"
        "2. For each next layer, only place 1 where the cell below is 1.\n"
        "3. Build Layer 1 first, then grow upward legally.\n"
        "4. Stop when no more valid 1s can be added or max_height is reached.\n"
        "5. Ensure your output structure is different from any previous attempt.\n"
    )

# temperature optimization
def stage_temp(query_idx: int, total_queries: int) -> float:
    third = total_queries / 3
    if query_idx < third:
        return 0.2
    if query_idx < 2 * third:
        return 0.5
    return 0.8

# feedback optimization
def feedback_prompt(history: list) -> str:
    """
    history: [
        {
            "is_valid": False,
            "is_novel": True,
            "error_msg": "Block at (1,2) Layer 2 has no support"
        },
        ...
    ]
    Returns English feedback string to append to prompt.
    """
    if not history:
        return ""

    latest = history[-1]

    if latest["is_valid"] and latest["is_novel"]:
        return ""
    
    lines = ["\n[System Feedback] Issues in the previous attempt:"]

    if not latest["is_valid"]:
        msg = latest.get("error_msg", "Unknown validity error")
        lines.append(f"- Invalid: {msg}.")
        lines.append("- The top-down view must match the input exactly.")
        lines.append("- Every block at layer L must have a block directly below it at layer L-1, or it collapses.")
        lines.append("- Tip: build from the bottom up; ensure Layer 1 satisfies the projection first.")

    if not latest["is_novel"]:
        lines.append("- Not novel: identical to an earlier solution.")
        lines.append("- Generate a *different* valid structure by changing block positions or layer count.")
    
    lines.append("\nPlease fix the above and produce a new valid 3-D structure.")
    return "\n".join(lines)