"""Scale test: many agents, load balancing, and novelty exploration.

Tests how the SDK handles larger agent pools and whether the
multi-kernel voting (capability + load balancing + novelty)
actually distributes traffic or concentrates on one winner.
"""

import alps_discovery as alps
from collections import Counter

network = alps.LocalNetwork()

# Register 20 translation agents with identical capabilities
for i in range(20):
    network.register(f"translate-{i:02d}", ["legal translation", "EN-DE"])

# Register 5 summarization agents
for i in range(5):
    network.register(f"summarize-{i:02d}", ["document summarization", "legal briefs"])

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

# Check: does LoadBalancingKernel actually spread traffic?
unique_firsts = len(first_picks)
print(f"\nUnique first-place agents: {unique_firsts}/20")

if unique_firsts == 1:
    print("LIMITATION: All queries go to the same agent.")
    print("  LoadBalancingKernel uses forwards_count, but the")
    print("  CapabilityKernel dominates when all agents have")
    print("  identical similarity. The enzyme's majority vote")
    print("  breaks ties deterministically, not randomly.")
elif unique_firsts < 5:
    print("LIMITATION: Weak load balancing — only a few agents get traffic.")
else:
    print("Load balancing is distributing across agents.")

# Check if novelty kernel has any visible effect
print("\n=== Novelty effect ===")
# Discover once, record success for the winner, repeat
winner_streak = []
for i in range(10):
    results = network.discover("translate legal contract")
    winner = results[0].agent_name
    winner_streak.append(winner)
    network.record_success(winner)

unique_in_streak = len(set(winner_streak))
print(f"Winners over 10 rounds with success feedback: {winner_streak}")
print(f"Unique winners: {unique_in_streak}")
if unique_in_streak == 1:
    print("LIMITATION: No exploration — same agent wins every time.")
    print("  NoveltyKernel's sigma-based scoring doesn't overcome")
    print("  the CapabilityKernel + diameter advantage once an agent")
    print("  starts accumulating successes.")
