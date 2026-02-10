"""Scale test: many agents, randomized tie-breaking, and traffic distribution.

Tests how the SDK handles larger agent pools. When multiple agents
score within 5% of the top score, randomized tie-breaking shuffles
them so traffic distributes naturally — no external load balancer needed.
"""

import alps_discovery as alps
from collections import Counter

network = alps.LocalNetwork()

# Register 20 translation agents with identical capabilities
for i in range(20):
    network.register(f"translate-{i:02d}", ["legal translation", "EN-DE"],
                      endpoint=f"http://translate-{i:02d}:8000")

# Register 5 summarization agents
for i in range(5):
    network.register(f"summarize-{i:02d}", ["document summarization", "legal briefs"],
                      endpoint=f"http://summarize-{i:02d}:8000")

print(f"Registered {network.agent_count} agents\n")

# Run 100 discovery queries and count who gets picked first
first_picks = Counter()
for _ in range(100):
    results = network.discover("translate legal contract")
    if results:
        first_picks[results[0].agent_name] += 1

print("=== First-place distribution over 100 queries ===")
for agent, count in first_picks.most_common():
    bar = "#" * count
    print(f"  {agent}: {count:3d} {bar}")

unique_firsts = len(first_picks)
print(f"\nUnique first-place agents: {unique_firsts}/20")

if unique_firsts > 1:
    print("  Randomized tie-breaking distributes traffic across agents")
    print("  with similar scores (within 5% of the top score).")
else:
    print("  Only one winner — scores may differ enough to prevent ties.")

# Show that summarization queries go to different agents
print("\n=== Summarization queries ===")
sum_picks = Counter()
for _ in range(100):
    results = network.discover("summarize legal brief")
    if results:
        sum_picks[results[0].agent_name] += 1

for agent, count in sum_picks.most_common(5):
    bar = "#" * count
    print(f"  {agent}: {count:3d} {bar}")

# Show how feedback shifts distribution
print("\n=== Feedback shifts distribution ===")
# Boost one specific agent
for _ in range(20):
    network.record_success("translate-05", query="translate legal contract")

boosted_picks = Counter()
for _ in range(100):
    results = network.discover("translate legal contract")
    if results:
        boosted_picks[results[0].agent_name] += 1

print("After boosting translate-05 with 20 successes:")
for agent, count in boosted_picks.most_common(5):
    bar = "#" * count
    print(f"  {agent}: {count:3d} {bar}")
