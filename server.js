// server.js - Resilient OpenAI-compatible OpenRouter Proxy for JanitorAI
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const OPENROUTER_API_BASE = process.env.OPENROUTER_API_BASE || 'https://openrouter.ai/api/v1';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || process.env.NIM_API_KEY;

// Show <think> blocks in Janitor only when the model actually returns reasoning
const SHOW_REASONING = true;
// Enable thinking on capable models. Ignored for models without reasoning support.
const ENABLE_THINKING_MODE = true;
const DEFAULT_REASONING_EFFORT = 'low';

const axiosInstance = axios.create({
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
});

const MODEL_MAPPING = {
  'deepseek-flash': 'deepseek/deepseek-v4-flash-0731:floor',
  'deepseek-chat': 'deepseek/deepseek-chat:floor',
  'deepseek-r1': 'deepseek/deepseek-r1:floor',
  'qwen-72b': 'qwen/qwen-2.5-72b-instruct:floor'
};

// Janitor / OpenAI generation fields to forward unchanged when present
const PASSTHROUGH_KEYS = [
  'temperature',
  'max_tokens',
  'max_completion_tokens',
  'top_p',
  'top_k',
  'min_p',
  'top_a',
  'presence_penalty',
  'frequency_penalty',
  'repetition_penalty',
  'stop',
  'seed',
  'n',
  'logit_bias'
];

// Fallback if OpenRouter model catalog has not loaded yet
const REASONING_MODEL_PATTERNS = [
  /deepseek-r1/i,
  /deepseek-reasoner/i,
  /deepseek-v3\.[12]/i,
  /deepseek-v4/i,
  /deepseek-flash/i,
  /\br1\b/i,
  /\bo1\b/i,
  /\bo3\b/i,
  /\bo4\b/i,
  /gpt-5/i,
  /gemini-2\.5/i,
  /gemini-3/i,
  /thinking/i,
  /reasoner/i,
  /qwen3/i,
  /qwq/i,
  /grok-3/i,
  /grok-4/i
];

const NON_REASONING_MODEL_PATTERNS = [
  /^deepseek\/deepseek-chat(?!-v3\.[12])(?!-v4)/i,
  /qwen-2\.5-72b-instruct/i,
  /qwen\/qwen-2\.5-72b-instruct/i
];

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

let modelReasoningCache = new Map();
let modelCacheLoadedAt = 0;
const MODEL_CACHE_TTL_MS = 30 * 60 * 1000;

function stripProviderSuffix(modelId = '') {
  return String(modelId).split(':')[0];
}

async function refreshModelReasoningCache() {
  try {
    const response = await axiosInstance.get(`${OPENROUTER_API_BASE}/models`, {
      timeout: 20000
    });
    const next = new Map();
    for (const model of response.data?.data || []) {
      const params = model.supported_parameters || [];
      const supports = !!(
        model.reasoning ||
        params.includes('reasoning') ||
        params.includes('reasoning_effort') ||
        params.includes('include_reasoning')
      );
      next.set(model.id, supports);
      next.set(stripProviderSuffix(model.id), supports);
    }
    modelReasoningCache = next;
    modelCacheLoadedAt = Date.now();
    console.log(`Loaded reasoning caps for ${next.size} model ids`);
  } catch (err) {
    console.error('Failed to refresh OpenRouter model catalog:', err.message);
  }
}

function fallbackSupportsReasoning(modelId) {
  const id = stripProviderSuffix(modelId);
  if (NON_REASONING_MODEL_PATTERNS.some((re) => re.test(id))) return false;
  return REASONING_MODEL_PATTERNS.some((re) => re.test(id));
}

function modelSupportsReasoning(modelId) {
  const id = stripProviderSuffix(modelId);
  if (modelReasoningCache.has(modelId)) return modelReasoningCache.get(modelId);
  if (modelReasoningCache.has(id)) return modelReasoningCache.get(id);
  return fallbackSupportsReasoning(id);
}

function copyDefined(target, source, keys) {
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) {
      target[key] = source[key];
    }
  }
  return target;
}

async function parseAxiosStreamError(error) {
  if (error.response?.data && typeof error.response.data.on === 'function') {
    try {
      const chunks = [];
      for await (const chunk of error.response.data) {
        chunks.push(chunk);
      }
      const raw = Buffer.concat(chunks).toString('utf-8');
      try {
        const parsed = JSON.parse(raw);
        return parsed.error?.message || parsed.detail || raw;
      } catch (e) {
        return raw || error.message;
      }
    } catch (e) {
      return error.message;
    }
  }
  return error.response?.data?.error?.message || error.response?.data?.detail || error.message;
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenRouter Proxy Active',
    thinkingDefault: ENABLE_THINKING_MODE,
    modelCacheSize: modelReasoningCache.size
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(m => ({
      id: m,
      object: 'model',
      created: Date.now(),
      owned_by: 'openrouter-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!modelCacheLoadedAt || Date.now() - modelCacheLoadedAt > MODEL_CACHE_TTL_MS) {
      refreshModelReasoningCache();
    }

    const body = req.body || {};
    let { model, messages, stream } = body;

    let cleanedMessages = (messages || []).map(msg => {
      let content = msg.content;
      if (msg.role === 'assistant' && typeof content === 'string') {
        content = content.replace(/<think>[\s\S]*?<\/think>\s*/g, '');
      }
      return {
        role: msg.role,
        content: content || ''
      };
    });

    const targetModel = MODEL_MAPPING[model] || model || 'deepseek/deepseek-v4-flash-0731:floor';
    const reasoningCapable = modelSupportsReasoning(targetModel);
    const useReasoning = ENABLE_THINKING_MODE && reasoningCapable;

    const openrouterRequest = {
      model: targetModel,
      messages: cleanedMessages,
      stream: !!stream,
      provider: {
        sort: 'price',
        allow_fallbacks: true
      }
    };

    copyDefined(openrouterRequest, body, PASSTHROUGH_KEYS);

    // Only attach reasoning for models that support it.
    // Non-reasoning models (deepseek-chat, qwen-2.5-72b, etc.) get no reasoning field.
    if (useReasoning) {
      openrouterRequest.reasoning = {
        effort: body.reasoning?.effort || body.reasoning_effort || DEFAULT_REASONING_EFFORT
      };
    }

    console.log(
      `→ \( {targetModel} | reasoning= \){useReasoning} | max_tokens=${openrouterRequest.max_tokens ?? 'unset'}`
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      if (res.flushHeaders) res.flushHeaders();

      const heartbeat = setInterval(() => {
        try {
          res.write(': keepalive\n\n');
        } catch (err) {
          clearInterval(heartbeat);
        }
      }, 15000);

      let response;
      const MAX_RETRIES = 5;
      let currentDelay = 1500;

      try {
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
          try {
            response = await axiosInstance.post(`${OPENROUTER_API_BASE}/chat/completions`, openrouterRequest, {
              headers: {
                'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://railway.app',
                'X-Title': 'JanitorAI Proxy'
              },
              responseType: 'stream',
              timeout: 300000
            });
            break;
          } catch (error) {
            const status = error.response?.status;
            const detailedErrorMsg = await parseAxiosStreamError(error);

            console.error(`🚨 OpenRouter Error [Status ${status}]:`, detailedErrorMsg);

            if (status === 400 || status === 401 || status === 404) {
              throw new Error(`[HTTP ${status}] ${detailedErrorMsg}`);
            }

            const isRetryable = status === 429 || status >= 500;
            if (isRetryable && attempt < MAX_RETRIES) {
              await sleep(currentDelay);
              currentDelay *= 1.5;
            } else {
              throw new Error(detailedErrorMsg);
            }
          }
        }

        clearInterval(heartbeat);

        let buffer = '';
        let reasoningStarted = false;
        let hasEmittedContent = false;

        response.data.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith(':')) continue;

            if (trimmed.startsWith('data: ')) {
              const payload = trimmed.slice(6);

              if (payload === '[DONE]') {
                if (reasoningStarted) {
                  res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n</think>\n\n' } }] })}\n\n`);
                  reasoningStarted = false;
                }
                res.write('data: [DONE]\n\n');
                continue;
              }

              try {
                const data = JSON.parse(payload);

                if (data.choices?.[0]?.delta) {
                  const delta = data.choices[0].delta;
                  const reasoning = useReasoning ? (delta.reasoning_content || delta.reasoning) : null;
                  const content = delta.content;

                  let combinedContent = '';

                  if (SHOW_REASONING && reasoning) {
                    if (!reasoningStarted) {
                      combinedContent += '<think>\n';
                      reasoningStarted = true;
                    }
                    combinedContent += reasoning;
                  }

                  if (content !== undefined && content !== null) {
                    if (reasoningStarted && content !== '') {
                      combinedContent += '\n</think>\n\n';
                      reasoningStarted = false;
                    }
                    combinedContent += content;
                  }

                  if (combinedContent !== '') {
                    hasEmittedContent = true;
                    delta.content = combinedContent;
                    delete delta.reasoning_content;
                    delete delta.reasoning;
                    res.write(`data: ${JSON.stringify(data)}\n\n`);
                  }
                } else {
                  res.write(`data: ${JSON.stringify(data)}\n\n`);
                }
              } catch (e) {
                // Ignore incomplete line chunks
              }
            }
          }
        });

        response.data.on('end', () => {
          if (reasoningStarted) {
            res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n</think>\n\n' } }] })}\n\n`);
          }
          if (!hasEmittedContent) {
            res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: '\n\n*(Model reached token budget before drafting narrative. Increase Max New Tokens in settings.)*' } }] })}\n\n`);
          }
          res.end();
        });

      } catch (streamError) {
        clearInterval(heartbeat);
        console.error('🚨 Stream Error:', streamError.message);

        res.write(`data: ${JSON.stringify({
          choices: [{ delta: { content: `\n\n**[Proxy Error]**: ${streamError.message}` } }]
        })}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }

    } else {
      const response = await axiosInstance.post(`${OPENROUTER_API_BASE}/chat/completions`, openrouterRequest, {
        headers: {
          'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://railway.app',
          'X-Title': 'JanitorAI Proxy'
        },
        responseType: 'json',
        timeout: 300000
      });
      res.json(response.data);
    }

  } catch (error) {
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.response?.data?.error?.message || error.message || 'Server error' }
      });
    }
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: 'Not found' } }));

app.listen(PORT, () => {
  console.log(`🚀 Proxy active on port ${PORT}`);
  refreshModelReasoningCache();
});
