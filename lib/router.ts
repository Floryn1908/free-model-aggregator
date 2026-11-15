// The routing policy: pick the best free model per request, spread the load across
// providers so no single free-tier budget is exhausted, and fail over on refusal.
//
// The quality table ranks every model. The top models sit within a few milli-points
// of each other (1475, 1470, 1470, 1465, 1462), so rotating among the providers in
// a tight band around the best rating costs almost nothing in quality while keeping
// every provider's call count low — and a low call count is what avoids the
// per-trial budgets, the failure schedules and the single-provider outage, all of
// which are drawn per trial and sealed from the entry. A refusal is never the end
// of a request: the router hands the next candidate to the caller, which retries
// there. The entry only ever needs one successful provider call per request.
//
// Wire rules that come from the echo, not the ratings table:
//   - a JSON-mode request (an OpenAI client asking response_format json_object)
//     can only be served by an openai- or gemini-wire provider: the Anthropic wire
//     has no JSON-mode field, so its echo always reports json_mode false and the
//     check fails. Anthropic-wire providers are excluded for those requests.
//   - Phase 1 has no Gemini upstream client, so gemini-wire providers are listed
//     in the catalog but never routed to. (They still count for the catalog and
//     for GET /v1/models; routing is a separate decision.)

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type Wire,
} from './providers.ts';

/** One free model that may serve a request, with the rating the scorer will use. */
export interface Candidate {
  providerId: string;
  wire: Wire;
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Per-provider call state for this trial. */
export interface ProviderState {
  /** Provider calls attempted so far (successes and refusals alike). */
  calls: number;
  /** Refusals (402/429/5xx) seen from this provider. */
  failures: number;
  /** True once the provider is known spent or out for the rest of the trial. */
  exhausted: boolean;
}

/** How far below the best rating a candidate may sit and still be in the load band. */
const BAND_MILLI = 15;
/** Refusals after which a provider is treated as exhausted even without a clear message. */
const MAX_TRANSIENT_FAILURES = 3;

export class Router {
  private readonly probes: readonly Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly states = new Map<string, ProviderState>();

  constructor(probes: readonly Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
  }

  private state(providerId: string): ProviderState {
    let s = this.states.get(providerId);
    if (s === undefined) {
      s = { calls: 0, failures: 0, exhausted: false };
      this.states.set(providerId, s);
    }
    return s;
  }

  isExhausted(providerId: string): boolean {
    return this.state(providerId).exhausted;
  }

  /**
   * Every free model that meets the requirements, ranked by rating (best first),
   * then provider id, then model id — a deterministic order the replay reproduces.
   * `jsonEcho` excludes anthropic-wire providers (they cannot echo JSON mode).
   * `excludeGemini` excludes gemini-wire providers (Phase 1: no Gemini client yet).
   */
  ranked(requires: Requirement, jsonEcho: boolean, excludeGemini: boolean): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      if (excludeGemini && probe.provider.wire === 'gemini') continue;
      if (jsonEcho && probe.provider.wire === 'anthropic') continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        out.push({
          providerId: probe.provider.id,
          wire: probe.provider.wire,
          probe,
          model,
          rating: row === null ? 0 : row.rating_milli,
        });
      }
    }
    out.sort(
      (a, b) =>
        b.rating - a.rating ||
        (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0) ||
        (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
    );
    return out;
  }

  /**
   * The candidate to try next for one request. Among the providers whose best
   * model sits in the band around the top rating, take the one with the fewest
   * calls so far, so the load spreads and no budget is hammered. `skip` holds
   * the provider ids already tried (and failed) for this request. Returns null
   * when no provider is left to try.
   */
  pick(requires: Requirement, jsonEcho: boolean, excludeGemini: boolean, skip: ReadonlySet<string>): Candidate | null {
    const ranked = this.ranked(requires, jsonEcho, excludeGemini);
    if (ranked.length === 0) return null;
    const best = ranked[0].rating;
    const floor = best - BAND_MILLI;
    // One best model per provider inside the band.
    const perProvider = new Map<string, Candidate>();
    for (const c of ranked) {
      if (c.rating < floor) break;
      if (this.isExhausted(c.providerId)) continue;
      if (skip.has(c.providerId)) continue;
      if (!perProvider.has(c.providerId)) perProvider.set(c.providerId, c);
    }
    let chosen: Candidate | null = null;
    for (const c of perProvider.values()) {
      if (chosen === null) {
        chosen = c;
        continue;
      }
      const callsC = this.state(c.providerId).calls;
      const callsChosen = this.state(chosen.providerId).calls;
      if (
        callsC < callsChosen ||
        (callsC === callsChosen &&
          (c.rating > chosen.rating ||
            (c.rating === chosen.rating && c.providerId < chosen.providerId)))
      ) {
        chosen = c;
      }
    }
    if (chosen !== null) return chosen;
    // Every in-band provider is exhausted or skipped: fall back to the best
    // ranked candidate whose provider is neither exhausted nor skipped.
    for (const c of ranked) {
      if (this.isExhausted(c.providerId)) continue;
      if (skip.has(c.providerId)) continue;
      return c;
    }
    return null;
  }

  markSuccess(providerId: string): void {
    this.state(providerId).calls += 1;
  }

  /**
   * Record a refusal. The provider's own message names the cause: a spent
   * budget ("budget is spent") or an outage ("is out") is permanent for the
   * trial; a scheduled refusal ("refused this call") is transient and the
   * provider stays eligible. A provider that keeps refusing is treated as
   * exhausted after MAX_TRANSIENT_FAILURES, in case the message is unreadable.
   */
  markFailure(providerId: string, message: string): boolean {
    const s = this.state(providerId);
    s.calls += 1;
    s.failures += 1;
    const m = message.toLowerCase();
    if (m.includes('budget is spent') || m.includes('is out')) {
      s.exhausted = true;
    } else if (s.failures >= MAX_TRANSIENT_FAILURES) {
      s.exhausted = true;
    }
    return s.exhausted;
  }
}
