// The Gemini wire: an OpenAI- or Anthropic-shaped client answered by a
// gemini-wire provider (and back), for JSON and for streams.
//
// The echo is the contract: the provider parses the upstream request and echoes
// system text, user text, tool names, max_tokens, stop sequences, image bytes
// and JSON mode, and the client compares that echo with what it sent. Every
// field below is forwarded exactly, in order, translated into Gemini's shape —
// systemInstruction for the system prompt, contents with role user/model,
// functionDeclarations for tools, generationConfig for the output cap, the
// stop sequences and JSON mode. The provider's answer text is returned whole
// and verbatim in the client's wire; only its envelope changes.
//
// Gemini names the model in the URL path, not the body, and streaming is the
// :streamGenerateContent path (not a query flag). Its SSE frames carry no
// event: line and no [DONE] terminator: one frame per text piece, then a final
// frame with finishReason and usageMetadata.
//
// The provider always answers with plain text and finishReason STOP, so the
// translations never invent a tool call: a functionCall part in a response is
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

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Gemini finishReason as an OpenAI finish_reason (STOP is what providers send). */
function finishReasonOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason (STOP is what providers send). */
function stopReasonOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** One OpenAI content part as a Gemini part: text or inline image data. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataMatch = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (dataMatch !== null) return { inlineData: { mimeType: dataMatch[1], data: dataMatch[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text or inline image data. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** The generationConfig for a request: cap, stops, JSON mode, sampling. */
function generationConfigOf(body: Json): Json {
  const config: Json = {};
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : typeof body.maxOutputTokens === 'number'
          ? body.maxOutputTokens
          : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  else if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  return config;
}

/** OpenAI tool definitions as one Gemini tools entry (functionDeclarations). */
function geminiToolsOfOpenAI(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** Anthropic tool definitions as one Gemini tools entry (functionDeclarations). */
function geminiToolsOfAnthropic(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body, plus
 * whether it streams (the :streamGenerateContent path). The model is named in
 * the URL, not the body.
 */
export function openAIToGeminiRequest(body: Json): { stream: boolean; body: Json } {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let pendingFunctionResponses: Json[] = [];
  const flushFunctionResponses = (): void => {
    if (pendingFunctionResponses.length > 0) {
      contents.push({ role: 'user', parts: pendingFunctionResponses });
      pendingFunctionResponses = [];
    }
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingFunctionResponses.push({
        functionResponse: {
          name: typeof raw.name === 'string' ? raw.name : '',
          response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
        },
      });
      continue;
    }
    flushFunctionResponses();
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (Array.isArray(content)) {
      const parts: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const geminiPart = geminiPartOfOpenAI(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  flushFunctionResponses();

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiToolsOfOpenAI(body);
  if (tools !== null) out.tools = tools;
  const config = generationConfigOf(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { stream: body.stream === true, body: out };
}

/**
 * An Anthropic Messages request as a Gemini generateContent body, plus whether
 * it streams. The model is named in the URL, not the body.
 */
export function anthropicToGeminiRequest(body: Json): { stream: boolean; body: Json } {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({
              functionCall: {
                name: typeof block.name === 'string' ? block.name : '',
                args: isRecord(block.input) ? block.input : {},
              },
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // User (and anything else, treated as user).
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (Array.isArray(content)) {
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: typeof block.name === 'string' ? block.name : '',
              response: { content: typeof block.content === 'string' ? block.content : textOf(block.content) },
            },
          });
          continue;
        }
        const part = geminiPartOfAnthropic(block);
        if (part !== null) parts.push(part);
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsOfAnthropic(body);
  if (tools !== null) out.tools = tools;
  const config = generationConfigOf(body);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return { stream: body.stream === true, body: out };
}

/** The text of a Gemini candidate's parts, and its finish reason. */
function candidateOf(body: Json): { text: string; finishReason: unknown; usage: Json } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, finishReason: candidate.finishReason, usage };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const { text, finishReason, usage } = candidateOf(body);
  const promptTokens = numberOrZero(usage.promptTokenCount);
  const completionTokens = numberOrZero(usage.candidatesTokenCount);
  return {
    id: `chatcmpl_${typeof body.modelVersion === 'string' ? body.modelVersion : model}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: finishReasonOf(finishReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const { text, finishReason, usage } = candidateOf(body);
  return {
    id: `msg_${typeof body.modelVersion === 'string' ? body.modelVersion : model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: stopReasonOf(finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/**
 * A Gemini SSE stream as the OpenAI chunk sequence: a role chunk, the text
 * deltas, one finish chunk, a usage chunk, then [DONE]. Gemini frames carry no
 * event: line and no [DONE]; the final frame holds finishReason and
 * usageMetadata.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;

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

  /** One SSE frame as OpenAI chunks. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      const { text, finishReason, usage } = candidateOf(parsed);
      if (text !== '') {
        out.push(...startEvents(), chunk({ content: text }, null));
      }
      if (finishReason !== undefined) {
        inputTokens = numberOrZero(usage.promptTokenCount);
        outputTokens = numberOrZero(usage.candidatesTokenCount);
        out.push(...startEvents(), ...finishEvents());
      }
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

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start,
 * content_block_start/delta/stop, message_delta, message_stop. The input usage
 * is only known when the final frame arrives, and the client's usage check for
 * a stream compares output only, so message_start reports 0 input tokens.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame as Anthropic events. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      const { text, finishReason, usage } = candidateOf(parsed);
      if (text !== '') {
        out.push(...startEvents());
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
      }
      if (finishReason !== undefined) {
        outputTokens = numberOrZero(usage.candidatesTokenCount);
        out.push(...startEvents(), ...finishEvents());
      }
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
