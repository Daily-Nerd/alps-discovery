"""Local invoke: optional callable for single-process agents.

When agents live in the same Python process, you can register a
callable alongside capabilities. Discovery returns it on the result
so the caller can invoke without knowing the agent's internals.
"""

import alps_discovery as alps


# Simulate local agent functions
def translate(request: dict) -> dict:
    text = request.get("text", "")
    target = request.get("target_lang", "DE")
    return {"translated": f"[{target}] {text}", "chars": len(text)}


def summarize(request: dict) -> dict:
    text = request.get("text", "")
    return {"summary": text[:50] + "...", "reduction": 0.8}


network = alps.LocalNetwork()

network.register(
    "translate-agent",
    ["legal translation", "EN-DE", "EN-FR"],
    invoke=translate,
)

network.register(
    "summarize-agent",
    ["document summarization", "legal briefs"],
    invoke=summarize,
)

# Discover + invoke in one flow
query = "translate legal contract"
results = network.discover(query)
print(f"Query: '{query}'")
print(f"Best match: {results[0].agent_name} (similarity={results[0].similarity:.3f})")

if results[0].invoke:
    output = results[0].invoke({"text": "Vertragsbedingungen und Haftungsausschluss", "target_lang": "EN"})
    print(f"Output: {output}")
    network.record_success(results[0].agent_name)
else:
    print("No invoke callable registered — use endpoint instead")

# Try a summarization query
print()
query = "summarize legal document"
results = network.discover(query)
print(f"Query: '{query}'")
print(f"Best match: {results[0].agent_name} (similarity={results[0].similarity:.3f})")

if results[0].invoke:
    output = results[0].invoke({"text": "This agreement constitutes the entire understanding between the parties..."})
    print(f"Output: {output}")
