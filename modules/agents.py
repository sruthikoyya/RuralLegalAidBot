import ast
import logging
import operator
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Safe math operators only — no builtins, no imports
_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Regex patterns that suggest a math calculation is needed
_MATH_PATTERNS = [
    r"\d+\s*[\+\-\*\/\%\^]\s*\d+",          # basic: 100 + 200
    r"calculate|compute|how much|total|sum",  # intent words
    r"\d+\s*(acres?|hectares?|guntas?)",       # land measurement
    r"emi|interest|loan|principal|rate",       # financial
    r"\d+\s*%\s*of\s*\d+",                   # percentage of
    r"per\s*(day|month|year|acre|hectare)",   # rates
    r"₹\s*\d+|\d+\s*rupees?",               # rupee amounts
    # Telugu math-related
    r"లెక్కించు|మొత్తం|వడ్డీ|రుణం",
]

_MATH_RE = re.compile("|".join(_MATH_PATTERNS), re.IGNORECASE)
def _safe_eval(node) -> float:
    """Recursively evaluate an AST expression with whitelisted ops only."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left  = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator")
        return op_fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported node type: {type(node).__name__}")


def extract_math_expression(text: str) -> Optional[str]:
    # Match: numbers, operators, parentheses, dots (for decimals)
    pattern = r"[\d\s\+\-\*\/\%\^\(\)\.]{3,}"
    matches = re.findall(pattern, text)
    for m in matches:
        cleaned = m.strip()
        if cleaned and any(c in cleaned for c in "+-*/^%"):
            return cleaned
    return None

def run_calculator(query: str) -> Optional[str]:
    if not _MATH_RE.search(query):
        return None

    # Try percentage pattern: X% of Y
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?(\d+(?:\.\d+)?)", query, re.IGNORECASE)
    if pct_match:
        pct  = float(pct_match.group(1))
        base = float(pct_match.group(2))
        result = (pct / 100) * base
        return f" {pct}% of {base:,.0f} = ₹{result:,.2f}"

    # Try EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
    emi_match = re.search(
        r"(?:emi|loan|రుణం).*?(?:₹\s*)?(\d+(?:,\d+)*(?:\.\d+)?).*?(\d+(?:\.\d+)?)\s*%.*?(\d+)\s*(?:months?|years?|నెలలు|సంవత్సరాలు)",
        query, re.IGNORECASE
    )
    if emi_match:
        try:
            principal = float(emi_match.group(1).replace(",", ""))
            annual_rate = float(emi_match.group(2))
            term_str = emi_match.group(3)
            # Detect if years or months
            n = int(term_str)
            if "year" in query.lower() or "సంవత్సర" in query:
                n = n * 12
            r = annual_rate / 100 / 12
            if r == 0:
                emi = principal / n
            else:
                emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
            return (
                f" EMI Calculation:\n"
                f"  Principal: ₹{principal:,.0f}\n"
                f"  Rate: {annual_rate}% p.a.\n"
                f"  Tenure: {n} months\n"
                f"  Monthly EMI: ₹{emi:,.2f}"
            )
        except Exception:
            pass  # Fall through to generic calculator

    # Generic arithmetic expression
    expr = extract_math_expression(query)
    if expr:
        try:
            tree   = ast.parse(expr, mode="eval")
            result = _safe_eval(tree.body)
            # Format nicely
            if result == int(result):
                formatted = f"{int(result):,}"
            else:
                formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
            return f" {expr.strip()} = {formatted}"
        except (ValueError, ZeroDivisionError, SyntaxError) as e:
            logger.debug(f"Calculator eval failed for '{expr}': {e}")

    return None

def run_web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search>=6.0")
        return ""

    # Bias search toward Indian government sources
    search_query = f"{query} India government scheme law 2024 2025"
    logger.info(f"[WebSearch] Query: '{search_query[:100]}'")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                search_query,
                max_results=max_results,
                region="in-en",        # India English region
                safesearch="moderate",
            ))

        if not results:
            logger.info("[WebSearch] No results found")
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body  = r.get("body",  "")[:300]  # cap at 300 chars per result
            href  = r.get("href",  "")
            parts.append(f"[{i}] {title}\n{body}\nSource: {href}")

        combined = "\n\n".join(parts)
        logger.info(f"[WebSearch] Returned {len(results)} results")
        return combined

    except Exception as e:
        logger.warning(f"[WebSearch] Failed: {e}")
        return ""

_UNCERTAINTY_PHRASES = [
    "i don't have information",
    "not in the documents",
    "cannot find",
    "not available",
    "no information",
    "cannot answer",
    "consult a",
    "not mentioned",
]

_TIME_TRIGGERS = [
    "latest", "current", "today", "recent", "2024", "2025", "2026",
    "new scheme", "new law", "amendment", "notification", "update",
    "ఇప్పుడు", "తాజా", "కొత్త", "అమెండ్మెంట్",
]


def should_web_search(query: str, rag_answer: str, enabled: bool = True) -> bool:
    if not enabled:
        return False
    q_lower = query.lower()
    a_lower = rag_answer.lower()
    uncertain  = any(p in a_lower for p in _UNCERTAINTY_PHRASES)
    time_based = any(kw in q_lower for kw in _TIME_TRIGGERS)
    return uncertain or time_based

def route_agents(query: str, calculator_enabled: bool = True) -> dict:
    use_calc = calculator_enabled and bool(_MATH_RE.search(query))
    return {
        "calculator": use_calc,
        "web": False,  # web search decision made post-RAG
    }
