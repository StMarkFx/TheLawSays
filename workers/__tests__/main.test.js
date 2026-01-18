import test from 'node:test';
import assert from 'node:assert/strict';

import worker from '../main.js';

function createEnv(overrides = {}) {
  const aiCalls = [];
  const vectorizeCalls = [];
  const env = {
    CORS_ORIGINS: 'https://main.thelawsays-frontend.pages.dev',
    AI: {
      run: async (model, payload) => {
        aiCalls.push({ model, payload });
        if (model === '@cf/baai/bge-base-en-v1.5') {
          return { data: [[0.1, 0.2, 0.3]] };
        }
        if (model === '@cf/meta/llama-3-8b-instruct') {
          return { response: 'Test answer from AI.' };
        }
        throw new Error(`Unexpected model ${model}`);
      },
    },
    VECTORIZE_INDEX: {
      query: async (payload) => {
        vectorizeCalls.push(payload);
        return { matches: [] };
      },
    },
    USE_D1: 'false',
    ...overrides,
  };

  return { env, aiCalls, vectorizeCalls };
}

test('health endpoint returns ok', async () => {
  const { env } = createEnv();
  const request = new Request('https://example.com/health', { method: 'GET' });
  const response = await worker.fetch(request, env, {});
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.deepEqual(body, { status: 'ok' });
});

test('chat endpoint calls Cloudflare AI models and returns response', async () => {
  const { env, aiCalls } = createEnv();
  const request = new Request('https://example.com/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Origin: 'https://main.thelawsays-frontend.pages.dev',
    },
    body: JSON.stringify({ message: 'What is Nigerian contract law?' }),
  });

  const response = await worker.fetch(request, env, {});
  assert.equal(response.status, 200);
  assert.equal(
    response.headers.get('Access-Control-Allow-Origin'),
    'https://main.thelawsays-frontend.pages.dev'
  );

  const body = await response.json();
  assert.equal(body.answer, 'Test answer from AI.');
  assert.equal(body.retrieval_used, false);

  assert.equal(aiCalls.length, 2);
  assert.equal(aiCalls[0].model, '@cf/baai/bge-base-en-v1.5');
  assert.equal(aiCalls[1].model, '@cf/meta/llama-3-8b-instruct');
});

test('chat endpoint skips vectorize when top_k is 0', async () => {
  const { env, vectorizeCalls } = createEnv();
  const request = new Request('https://example.com/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Origin: 'https://main.thelawsays-frontend.pages.dev',
    },
    body: JSON.stringify({ message: 'What is Nigerian contract law?', top_k: 0 }),
  });

  const response = await worker.fetch(request, env, {});
  assert.equal(response.status, 200);
  assert.equal(vectorizeCalls.length, 0);
});

test('chat endpoint validates message payload', async () => {
  const { env } = createEnv();
  const request = new Request('https://example.com/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });

  const response = await worker.fetch(request, env, {});
  assert.equal(response.status, 400);
  const body = await response.json();
  assert.equal(body.error, 'Message is required');
});
