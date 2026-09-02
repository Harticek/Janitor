// server.js - OpenAI to OpenRouter Proxy (Reasoning Disabled)
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

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

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
  res.json({ status: 'ok', service: 'OpenRouter Proxy Active (No Reasoning)' });
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
    let { model, messages, temperature, max_tokens, stream } = req.body;

    // Clean any legacy <think> blocks out of older chat history to save tokens
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

    const openrouterRequest = {
      model: targetModel,
      messages: cleanedMessages,
      temperature: temperature !== undefined ? temperature : 0.7,
      max_tokens: max_tokens || 2048,
      stream: stream || false,
      provider: {
        sort: 'price',
        allow_fallbacks: true
      },
      // 🚀 Explicitly disable thinking and drop reasoning tokens
      reasoning: {
        effort: 'none',
        exclude: true
      }
    };

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
                res.write('data: [DONE]\n\n');
                continue;
              }

              try {
                const data = JSON.parse(payload);

                if (data.choices?.[0]?.delta) {
                  const delta = data.choices[0].delta;

                  // Strip any reasoning fields if provider sends them
                  delete delta.reasoning_content;
                  delete delta.reasoning;
                  delete delta.reasoning_details;

                  // Pass through text directly
                  if (delta.content !== undefined && delta.content !== null) {
                    res.write(`data: ${JSON.stringify(data)}\n\n`);
                  }
                } else {
                  res.write(`data: ${JSON.stringify(data)}\n\n`);
                }
              } catch (e) {
                // Incomplete line chunk
              }
            }
          }
        });

        response.data.on('end', () => res.end());

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
app.listen(PORT, () => console.log(`🚀 Proxy active on port ${PORT}`));
