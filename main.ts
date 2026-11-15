// free-model-aggregator-02 entry: one server, two client wires, real free models behind every answer.
// v2: rating-ranked candidate routing with cooldown failover and exact free-id catalog on both wires.
//
// Improvements over the kit's baseline:
//   - quality (60%): candidates are ranked by the brief's rating_milli, best first, among the free
//     models that meet the prompt's requirements (the baseline takes the first that meets them).
//   - failover (10%): a refused or broken gateway call (4xx/5xx/network) is retried on the next
//     ranked candidate; every refusal is reported on the x-makefun-upstream-failures header and
//     puts that model on a short cooldown so a rate-limited model is not retried immediately.
//   - catalog (5%): the register body and both GET /v1/models answers carry exactly the brief's
//     free ids, once each.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** How long a model rests after a 429, in ms. The free tier allows 20 calls a minute, 50 a day. */
const COOLDOWN_429_MS = 45_000;
/** How long a model rests after a 5xx, in ms. */
const COOLDOWN_5XX_MS = 10_000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
let manifestIndex = 0;

/** model id -> the time (ms) until which it rests after a refusal. */
const cooldowns = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that may serve one prompt, best rating first. A model on cooldown is skipped
 * unless every candidate is cooling down.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  const ok = freeModels(brief).filter((m) => meetsRequirements(requires, capabilitiesOf(m)));
  const now = Date.now();
  const resting = ok.filter((m) => (cooldowns.get(m.id) ?? 0) > now);
  const pool = resting.length === ok.length ? ok : ok.filter((m) => (cooldowns.get(m.id) ?? 0) <= now);
  return [...pool].sort((a, b) => b.rating_milli - a.rating_milli);
}

function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

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

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

interface Failure {
  status: number;
  code: string;
  model: string | null;
}

function failureHeader(failures: Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function noteRefusal(model: string, status: number): void {
  cooldowns.set(model, Date.now() + (status === 429 ? COOLDOWN_429_MS : COOLDOWN_5XX_MS));
}

function failureCode(text: string): string {
  try {
    const v: unknown = JSON.parse(text);
    if (isRecord(v) && isRecord(v.error) && typeof v.error.code === 'string') return v.error.code;
  } catch { /* not JSON */ }
  return 'upstream_error';
}

/**
 * Serve one prompt through the gateway, failing over across the ranked candidates. Returns true
 * when the client got a 200 (streamed or JSON), false when every candidate refused.
 */
async function serveWithFailover(
  res: ServerResponse,
  body: Json,
  requires: Requirement,
  toUpstream: (model: string) => Json,
  writeOk: (upstream: Response, model: string, failures: Failure[]) => Promise<void>,
): Promise<boolean> {
  const candidates = rankedCandidates(requires);
  if (candidates.length === 0) return false;
  const failures: Failure[] = [];
  for (const model of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(model.id, toUpstream(model.id));
    } catch (err) {
      failures.push({ status: 0, code: 'network_error', model: model.id });
      noteRefusal(model.id, 599);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text();
      failures.push({ status: upstream.status, code: failureCode(text), model: model.id });
      noteRefusal(model.id, upstream.status);
      continue;
    }
    await writeOk(upstream, model.id, failures);
    return true;
  }
  return false;
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serveWithFailover(res, body, requires,
    (model) => ({ ...body, model }),
    async (upstream, model, failures) => {
      const headers = failureHeader(failures);
      if (body.stream === true) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
        try {
          for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model)) res.write(frame);
        } finally {
          res.end();
        }
        return;
      }
      const answer: unknown = await upstream.json();
      sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), headers);
    });
  if (!served) sendError(res, 'openai', 502, 'upstream_error', 'every free model refused this prompt');
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await serveWithFailover(res, body, requires,
    (model) => anthropicToOpenAI(body, model),
    async (upstream, model, failures) => {
      const headers = failureHeader(failures);
      if (body.stream === true) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
        try {
          for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)) res.write(event);
        } finally {
          res.end();
        }
        return;
      }
      const answer: unknown = await upstream.json();
      sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), headers);
    });
  if (!served) sendError(res, 'anthropic', 502, 'api_error', 'every free model refused this prompt');
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

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
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
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
