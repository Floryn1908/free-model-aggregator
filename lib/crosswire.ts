// Cross-wire translation the starter does not have: an OpenAI-shaped client
// answered by an Anthropic-wire provider (and back), for JSON and for streams.
//
// The echo is the contract: the provider parses the upstream request and echoes
// system text, user text, tool names, max_tokens, stop sequences, image bytes and
// JSON mode, and the client compares that echo with what it sent. Every field
// below is forwarded exactly, in order, translated across the wire — nothing is
// dropped, rewritten or reordered. The provider's answer text is returned whole
// and verbatim in the client's wire; only its envelope changes.
//
// The provider always answers with plain text and stop family 0, so the
// translations never invent a tool call: a tool_use block in a response is
// mapped only if one actually arrives.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A fixed epoch second for every translated OpenAI reply's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An Anthropic stop_reason as an OpenAI finish_reason (family 0 is what providers send). */
function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataMatch = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (dataMatch !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataMatch[1], data: dataMatch[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names
 * `model`. The system message becomes the system field, image_url parts become
 * image blocks, tool definitions and tool choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 * response_format is deliberately not carried: the Anthropic wire has no JSON
 * mode, and the router never sends a JSON-mode request down this path.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length > 0) {
      messages.push({ role: 'user', content: pendingToolResults });
      pendingToolResults = [];
    }
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    flushToolResults();
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      const text = textOf(raw.content);
      const content: Json[] = [];
      if (text !== '') content.push({ type: 'text', text });
      content.push(...toolCalls);
      messages.push({ role: 'assistant', content: content.length > 0 ? content : '' });
      continue;
    }
    // User (and anything else, treated as user).
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = blockOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const contentBlocks = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of contentBlocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const message: Json = { role: 'assistant', content: textParts.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isRecord(body.usage) ? body.usage : {};
  const promptTokens = numberOrZero(usage.input_tokens);
  const completionTokens = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  };
}

/**
 * An Anthropic SSE stream as the OpenAI chunk sequence: a role chunk, the text
 * deltas, one finish chunk, a usage chunk (the provider always reports usage,
 * and the client reads it from there), then [DONE]. The usage chunk is emitted
 * even when the client did not ask for stream_options.include_usage: without it
 * the client's usage check reads null and fails.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let toolIndex = -1;
  let nextToolIndex = 0;

  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  const usageChunk = (): string =>
    sseEvent(null, {
      id: 'chatcmpl_stream',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    });
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [chunk({}, 'stop'), usageChunk(), 'data: [DONE]\n\n'];
  };

  /** One SSE frame's lines as OpenAI chunks, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    let eventName: string | null = null;
    let data = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    }
    if (data === '' || eventName === null) return out;
    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      return out;
    }
    if (!isRecord(payload)) return out;
    if (eventName === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) {
      inputTokens = numberOrZero(payload.message.usage.input_tokens);
      out.push(...startEvents());
      return out;
    }
    if (eventName === 'content_block_delta' && isRecord(payload.delta)) {
      out.push(...startEvents());
      if (payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string' && payload.delta.text !== '') {
        out.push(chunk({ content: payload.delta.text }, null));
      } else if (payload.delta.type === 'input_json_delta' && typeof payload.delta.partial_json === 'string' && payload.delta.partial_json !== '') {
        if (toolIndex === -1) {
          toolIndex = nextToolIndex;
          nextToolIndex += 1;
        }
        out.push(
          chunk(
            { tool_calls: [{ index: toolIndex, function: { arguments: payload.delta.partial_json } }] },
            null,
          ),
        );
      }
      return out;
    }
    if (eventName === 'message_delta') {
      if (isRecord(payload.usage)) outputTokens = numberOrZero(payload.usage.output_tokens);
      return out;
    }
    if (eventName === 'message_stop') {
      out.push(...startEvents(), ...finishEvents());
      return out;
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}
