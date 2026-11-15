# Free Model Aggregator

Entry for make.fun bounty `free-model-aggregator-01`: one Node 22 server
(stdlib only) that serves the OpenAI Chat Completions API
(`POST /v1/chat/completions`, JSON + SSE) and the Anthropic Messages API
(`POST /v1/messages`, JSON + SSE), plus `GET /v1/models` in the wire the
auth header names, and routes every request to a free model on one of the
ten simulated providers.

## How it works

- **Lifecycle.** The server reads the brief from `MAKEFUN_PROMPT_FILE`,
  probes every provider's catalog, registers with
  `POST $MAKEFUN_ENV_URL/register` (catalog document), then polls
  `GET $MAKEFUN_ENV_URL/work` until done and exits 0.
  `node main.ts --list` prints the catalog document byte for byte.
- **Routing (`lib/router.ts`).** For each request the router ranks every
  free model that meets the request's requirements (tools / vision / JSON /
  `min_context`), filtered by wire when the client asked for JSON mode
  (the Anthropic wire can never echo JSON mode, so JSON requests never go
  to an Anthropic-wire provider). Among providers whose best model is
  within 15 rating-milli of the top rating it picks the provider with the
  fewest calls so far, so load spreads across the free tiers instead of
  burning one provider's budget. A provider that reports its budget spent
  or an outage is dropped permanently; a transient refusal counts toward a
  3-strike exhaustion.
- **Failover.** A refused call (402 / 429 / 5xx) is never retried on the
  same provider for the same request: the router moves to the next
  candidate and stops at the first success, so exactly one provider
  answers each request.
- **Wire fidelity (`lib/wire.ts`, `lib/crosswire.ts`, `lib/gemini.ts`).**
  Client fields are forwarded exactly, translated across wires: system
  text, user text, tool names in order, `max_tokens`, stop sequences in
  order, image bytes / URL, JSON mode. The provider's answer text is
  returned whole and verbatim, its usage numbers copied exactly, and its
  stop reported in family 0 (`stop` / `end_turn` / `STOP`). OpenAI
  streaming always carries a usage chunk (`stream_options.include_usage`
  is set upstream). All three wires are served both ways: OpenAI,
  Anthropic, and Gemini (`:generateContent` / `:streamGenerateContent`,
  model in the URL path). The Gemini wire is load-bearing, not a nicety:
  it is the second tools+vision provider, so a vision request still has
  a fallback when the first one is refused.
- **Catalog.** The registered catalog lists every provider with its own
  path (`/p/<id>`), key env var, wire, and exactly its free models;
  `GET /v1/models` lists exactly the registered free ids under both auth
  headers. The only paid model anywhere (OpenRouter's
  `anthropic/claude-opus-5.5`) is never routed to.

## Local grades

`grade-local --driver process --worktree` (kit's sealed scorer):

| trial | score | quality | served | wire | failover | catalog | upstream_calls |
|-------|-------|---------|--------|------|----------|---------|----------------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 27             |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 30             |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 30             |

All three replays matched byte for byte. Quality 100 is provably
unreachable on these instances: the quality bound ignores provider
budgets, and the best model alone cannot serve every request inside its
budget, so some requests must take a model a few rating-milli lower.

## Layout

- `main.ts` — server, lifecycle, request serving, failover loop.
- `lib/providers.ts` — brief parsing, auth headers, catalog probe,
  requirements, ratings.
- `lib/wire.ts` — OpenAI/Anthropic shapes, model lists, SSE helpers.
- `lib/router.ts` — rating-aware, load-balancing router with
  budget/outage exhaustion.
- `lib/crosswire.ts` — OpenAI client ↔ Anthropic provider translation.
- `lib/gemini.ts` — Gemini wire, both directions, JSON + streaming.
