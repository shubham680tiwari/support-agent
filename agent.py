from dotenv import load_dotenv
import inspect
import os
import sys
from google import genai
from google.genai import types

# load secret from .env
load_dotenv()

# Default model. Verified 2026-10-22 against
# https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash and
# https://ai.google.dev/gemini-api/docs/deprecations: "gemini-3.8-flash" is the
# current stable general-purpose Flash endpoint. The "gemini-3.7-flash" ID
# previously pinned here is valid but deprecated (2026-08-13); requests to it
# are auto-routed to 3.8 Flash, so this is a forward-looking pin, not a bug fix.
# Override at runtime with the GEMINI_MODEL environment variable.
DEFAULT_MODEL = "gemini-3.8-flash"

# Upper bound on tool-calling round trips within a single user turn, so a model
# that keeps requesting tools cannot loop forever.
MAX_TOOL_ITERATIONS = 10

# Prices in ORDER_DB are rupee-denominated, so refund messages are labelled INR.
# ASCII label rather than the rupee sign to avoid terminal encoding problems.
CURRENCY_LABEL = "INR"

# Refunds above this value require a human to approve at the terminal.
# NOTE: 50 reads as a leftover test value - every order in ORDER_DB costs
# between 80,000 and 180,000 INR, so in practice every refund trips the gate.
# Deliberately left unchanged pending a product decision on refund policy.
REFUND_APPROVAL_THRESHOLD = 50

# Fake order database backing the search tool. Prices are in INR.
ORDER_DB = {
    "ORD-001": {"item": "Iphone 18 pro", "price": 180000, "status": "Delivered"},
    "ORD-002": {"item": "Gaming Monitor", "price": 80000, "status": "Damaged"},
}

_client = None


def get_client():
    """Return a cached Gemini client, building it on first use.

    Raises RuntimeError when GEMINI_API_KEY is missing so the caller can fail
    fast with a readable message. The key value is never printed or logged.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put GEMINI_API_KEY=<your key> in the "
                ".env file at the project root, or export it in your shell."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# define search db tool
def search_database(order_id: str):
    """Database search tool."""
    return ORDER_DB.get(order_id, "Order not found")


# define mock refund tool
def issue_refund(order_id: str, amount: float):
    """Mock refund tool"""
    return f"Success: Your refund of amount {CURRENCY_LABEL} {amount} is issued for {order_id}."


my_tools = [search_database, issue_refund]

# The real tool functions, keyed by name. Used only to derive each tool's
# expected parameter names when validating model-supplied arguments.
_TOOL_FUNCTIONS = {func.__name__: func for func in my_tools}


def _handle_search_database(args: dict) -> str:
    """Validate arguments for search_database, then call it."""
    order_id = args.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        return "Error: search_database requires a non-empty string 'order_id'."
    return search_database(order_id)


def _handle_issue_refund(args: dict) -> str:
    """Validate arguments for issue_refund, apply the approval gate, then call it.

    This handler is the only path from the agent loop to issue_refund(), so the
    approval gate cannot be skipped by omitting or malforming an argument: every
    rejection returns an error string to the model and never falls through to
    the refund call.
    """
    order_id = args.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        return "Error: issue_refund requires a non-empty string 'order_id'."

    if "amount" not in args:
        return "Error: issue_refund requires a numeric 'amount'; none was provided."

    raw_amount = args["amount"]
    # bool is a subclass of int, so reject it explicitly rather than treating
    # True as 1.
    if raw_amount is None or isinstance(raw_amount, bool):
        return f"Error: issue_refund received a non-numeric 'amount': {raw_amount!r}."
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError):
        return f"Error: issue_refund received a non-numeric 'amount': {raw_amount!r}."

    # NaN compares unequal to itself; neither NaN nor an infinity is a usable
    # refund amount.
    if amount != amount or amount in (float("inf"), float("-inf")):
        return f"Error: issue_refund received a non-finite 'amount': {raw_amount!r}."
    if amount <= 0:
        return f"Error: issue_refund requires a positive 'amount'; got {amount}."

    if amount > REFUND_APPROVAL_THRESHOLD:
        print(f"Human approval required for {order_id} ({CURRENCY_LABEL} {amount})")
        approved = input("Approve the refund? (yes/no): ").strip().lower()
        if approved != "yes":
            return "Error: Refund rejected by admin"

    return issue_refund(order_id=order_id, amount=amount)


TOOL_HANDLERS = {
    "search_database": _handle_search_database,
    "issue_refund": _handle_issue_refund,
}


def call_tool(name: str, args) -> str:
    """Dispatch a model-requested tool call, returning a string for the model.

    Never raises: an unknown tool name, bad arguments, or a failing tool all
    come back as an "Error: ..." string so the agent loop stays alive.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        available = ", ".join(sorted(TOOL_HANDLERS))
        return f"Error: unknown tool '{name}'. Available tools: {available}."

    # FunctionCall.args is Optional[dict] in the SDK
    # (google/genai/types.py: FunctionCall.args), so it can arrive as None.
    try:
        call_args = dict(args or {})
    except (TypeError, ValueError):
        return f"Error: tool '{name}' received arguments that are not a mapping."

    expected = set(inspect.signature(_TOOL_FUNCTIONS[name]).parameters)
    unexpected = sorted(set(call_args) - expected)
    if unexpected:
        return (
            f"Error: tool '{name}' received unexpected argument(s): "
            f"{', '.join(unexpected)}. Expected only: {', '.join(sorted(expected))}."
        )

    try:
        return handler(call_args)
    except Exception as e:
        return f"Error: tool '{name}' failed: {e!r}"


def run_gemini_agent(user_prompt: str, contents: list):
    # Remember where this turn starts, so an API failure can roll the turn's
    # entries back out and leave the next turn a valid history.
    baseline = len(contents)

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    )

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    config = types.GenerateContentConfig(
        system_instruction="You are an e-commerce customer support agent.",
        tools=my_tools,
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = get_client().models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            print(f"API error: {e}")
            del contents[baseline:]
            return

        if response.function_calls:
            # Answer every call the model made, in order. Dropping any of them
            # would leave a call with no matching response in the history.
            response_parts = []
            for function_call in response.function_calls:
                func_name = function_call.name
                print(f"calling function {func_name}")
                tool_result = call_tool(func_name, function_call.args)
                response_parts.append(
                    types.Part.from_function_response(
                        name=func_name or "unknown_tool",
                        response={"result": tool_result}
                    )
                )

            # One content for the model's calls, then a single content carrying
            # one function-response part per call - the same shape the SDK's own
            # tool loop builds (google/genai/models.py).
            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=response_parts))
            continue

        contents.append(response.candidates[0].content)
        answer = response.text or "(no text answer returned)"
        print(f"\nAgent: {answer}\n")
        return

    # Every call issued so far has a matching response, so the history stays
    # valid for the next turn - no rollback needed here.
    print(
        f"\nStopped after {MAX_TOOL_ITERATIONS} tool iterations "
        "without a final answer.\n"
    )


def main():
    try:
        get_client()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    chat_history = []
    print("Support Agent Ready! (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        run_gemini_agent(user_input, chat_history)


if __name__ == "__main__":
    main()
