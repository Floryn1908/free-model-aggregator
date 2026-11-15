// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry reads the brief (provider directory, quality table, workload
// manifest), probes each provider's catalog route, registers one catalog, and
// serves the client's twenty requests. Every answer comes from a provider:
// the entry routes each request to the best free model that meets it, speaks
// that provider's wire, and fails over to the next candidate when a provider
// refuses. The provider's answer text is returned whole and verbatim in the
// wire the client used.
//
// Routing (lib/router.ts): candidates are ranked by the quality table's
// rating_milli; among the providers whose best model sits in a tight band
// around the top rating, the one with the fewest calls so far is picked, so
// the load spreads and no per-trial budget is hammered. A refusal marks the
// provider (a spent budget or an outage is permanent; a scheduled refusal is
// transient) and the request is retried on the next candidate. Only one
// successful provider call is ever made per request.
//
// Wires: an OpenAI client is answered by an openai-wire provider directly, by
// an anthropic-wire provider through lib/crosswire.ts, or by a gemini-wire
// provider through lib/gemini.ts; an Anthropic client is answered by an
// anthropic-wire provider directly, by an openai-wire provider through
// lib/wire.ts, or by a gemini-wire provider through lib/gemini.ts. A JSON-mode
// request is never sent to an anthropic-wire provider (that wire cannot echo
// JSON mode).
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers so far. No clock, no randomness. The catalog printed by
// --list is the same document the register body carries, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToOpenAIResponse,
  openAIEventsFromAnthropic,
  openAIToAnthropicRequest,
} from './lib/crosswire.ts';
import {
  anthropicEventsFromGemini,
  anthropicToGeminiRequest,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIEventsFromGemini,
  openAIToGeminiRequest,
} from './lib/gemini.ts';
import { Router, type Candidate } from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The Gemini wire is implemented (lib/gemini.ts): gemini-wire providers are
 *  routed to like any other. They hold the best vision models, and the two
 *  tools+vision requests have no other fallback when the anthropic-wire
 *  provider refuses. */
const EXCLUDE_GEMINI = false;
/** The Anthropic version header the anthropic-wire providers expect. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI request asks for JSON mode, which the echo must report. */
function asksJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The refusal message out of a provider's error envelope, in any wire's shape. */
async function errorMessageOf(res: Response): Promise<string> {
  let text = '';
  try {
    text = await res.text();
  } catch {
    return `HTTP ${res.status}`;
  }
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) {
      if (isRecord(parsed.error) && typeof parsed.error.message === 'string') return parsed.error.message;
      if (typeof parsed.message === 'string') return parsed.message;
    }
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return text !== '' ? text.slice(0, 200) : `HTTP ${res.status}`;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One inference call to a candidate provider, in that provider's own wire.
 * `clientWire` is the wire the client used; the upstream request is the
 * client's body translated into the provider's wire, with the routed model's
 * id. Streaming requests ask the provider for usage (an OpenAI provider only
 * reports it when asked), and the Anthropic wire gets its version header.
 */
async function callUpstream(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const streaming = body.stream === true;
  if (streaming) headers.accept = 'text/event-stream';
  if (clientWire === 'openai') {
    if (candidate.wire === 'openai') {
      const upstreamBody: Json = { ...body, model: candidate.model.id };
      if (streaming) upstreamBody.stream_options = { include_usage: true };
      return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
    }
    if (candidate.wire === 'anthropic') {
      headers['anthropic-version'] = ANTHROPIC_VERSION;
      return fetch(`${base}/v1/messages`, {
        method: 'POST',
        headers,
        body: JSON.stringify(openAIToAnthropicRequest(body, candidate.model.id)),
      });
    }
    if (candidate.wire === 'gemini') {
      const { stream, body: geminiBody } = openAIToGeminiRequest(body);
      const action = stream ? ':streamGenerateContent' : ':generateContent';
      return fetch(`${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${action}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(geminiBody),
      });
    }
    throw new Error(`the ${candidate.wire} upstream wire is not implemented in this build`);
  }
  // clientWire === 'anthropic'
  if (candidate.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    const upstreamBody: Json = { ...body, model: candidate.model.id };
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }
  if (candidate.wire === 'openai') {
    return fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(anthropicToOpenAI(body, candidate.model.id)),
    });
  }
  if (candidate.wire === 'gemini') {
    const { stream, body: geminiBody } = anthropicToGeminiRequest(body);
    const action = stream ? ':streamGenerateContent' : ':generateContent';
    return fetch(`${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${action}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(geminiBody),
    });
  }
  throw new Error(`the ${candidate.wire} upstream wire is not implemented in this build`);
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function sseHeaders(): Record<string, string> {
  return { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
}

/**
 * Serve one client request against the providers, in the client's wire.
 * Tries the router's candidates in order until one answers; a refusal marks
 * the provider and moves on, so a request is only lost when no free model is
 * left to try. Returns true when the response was sent.
 */
async function serve(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
  requires: Requirement,
  jsonEcho: boolean,
): Promise<boolean> {
  const active = router;
  if (active === null) {
    sendError(res, clientWire, 503, 'no_free_model', 'the router is not ready');
    return false;
  }
  const tried = new Set<string>();
  for (;;) {
    const candidate = active.pick(requires, jsonEcho, EXCLUDE_GEMINI, tried);
    if (candidate === null) {
      sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model could serve this request');
      return false;
    }
    tried.add(candidate.providerId);
    let upstream: Response;
    try {
      upstream = await callUpstream(candidate, clientWire, body);
    } catch (err) {
      active.markFailure(candidate.providerId, `fetch failed: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      active.markFailure(candidate.providerId, await errorMessageOf(upstream));
      continue;
    }
    active.markSuccess(candidate.providerId);
    const streaming = body.stream === true && upstream.body !== null;
    if (streaming) {
      res.writeHead(200, sseHeaders());
      if (clientWire === 'openai') {
        if (candidate.wire === 'openai') await pipeStream(upstream.body, res);
        else if (candidate.wire === 'anthropic') {
          for await (const event of openAIEventsFromAnthropic(upstream.body, candidate.model.id)) res.write(event);
          res.end();
        } else if (candidate.wire === 'gemini') {
          for await (const event of openAIEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
          res.end();
        }
      } else {
        if (candidate.wire === 'anthropic') await pipeStream(upstream.body, res);
        else if (candidate.wire === 'openai') {
          for await (const event of anthropicEventsFromOpenAI(upstream.body, candidate.model.id)) res.write(event);
          res.end();
        } else if (candidate.wire === 'gemini') {
          for await (const event of anthropicEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
          res.end();
        }
      }
      return true;
    }
    const answer: unknown = await upstream.json();
    const answerJson = isRecord(answer) ? answer : {};
    if (clientWire === 'openai') {
      if (candidate.wire === 'openai') sendJson(res, 200, answerJson);
      else if (candidate.wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(answerJson, candidate.model.id));
      else if (candidate.wire === 'gemini') sendJson(res, 200, geminiToOpenAIResponse(answerJson, candidate.model.id));
    } else {
      if (candidate.wire === 'anthropic') sendJson(res, 200, answerJson);
      else if (candidate.wire === 'openai') sendJson(res, 200, openAIToAnthropic(answerJson, candidate.model.id));
      else if (candidate.wire === 'gemini') sendJson(res, 200, geminiToAnthropicResponse(answerJson, candidate.model.id));
    }
    return true;
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  await serve('openai', body, res, requires, asksJsonMode(body));
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  // An Anthropic request carries no JSON-mode field, so the echo never checks
  // it: the manifest's json requirement (model capability) is enough.
  await serve('anthropic', body, res, requires, false);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
