# grouding_guard

A minimal, production-tested reference implementation of a **grounding guard** for LLM-powered agents.

The guard ensures an agent's response contains only figures and claims that are explicitly present in the retrieved source text — **refusing to answer rather than fabricating** when sources are insufficient.

---

## The problem

RAG pipelines retrieve relevant chunks and pass them to an LLM. The LLM generates a fluent answer — but fluency is not accuracy. A model can produce confident, well-formatted figures that appear nowhere in the retrieved text. In regulated domains (legal, financial, medical), a hallucinated number is not a minor error.

## The pattern
```
retrieve(query) → sources
agent_draft = llm(query, sources)
grounding_check(agent_draft, sources)
  → pass: return agent_draft
  → fail: return refusal  ← never fabricate
```
  
## Why exact matching

Approximate matching can be gamed by the same failure mode it is meant to catch: a hallucinated figure close to the real one might pass a fuzzy check. Exact matching is conservative by design. The cost is that correctly rounded or reformatted figures can trigger the guard — the fix is to refusal-route those edge cases, not to loosen the check.

## Production results

This pattern was deployed in a production AI agent serving enterprise clients in a regulated industry (French real-estate law and compliance):

- **Zero hallucinations** across all client interactions
- Verified on a **55-question evaluation set**
- **91% first-rank retrieval accuracy** (100% within top 5 results)

## Usage

```python
from grounding_guard import agent_respond

sources = ["The average price per m² was €9,850 in Q3 2024, down 3.2% year-over-year."]
proposed = "In Q3 2024 the average was €9,850/m², down 3.2%."

response = agent_respond(proposed, sources)
# → returns proposed response (all figures verified)

hallucinated = "In Q3 2024 the average was €10,200/m², down 1.8%."
response = agent_respond(hallucinated, sources)
# → returns refusal message (€10,200 and 1.8% not found in sources)
```

Run the demo:

```bash
python grounding_guard.py
```

## Extending the guard

The numerical claim patterns in `extract_numerical_claims` cover the most common hallucination surface (currency, percentages, years, quantities). To extend:

- Add regex patterns for domain-specific formats (ICD codes, contract clause numbers)
- Replace exact matching with a stricter normalizer for locale-specific number formatting
- Log guard firings to a monitoring system to track hallucination attempts over time

## License

MIT
