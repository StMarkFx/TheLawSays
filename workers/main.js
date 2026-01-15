/**
 * Cloudflare Workers entry point for TheLawSays backend.
 * Implements RAG functionality using Cloudflare AI and Vectorize.
 */

const DEFAULT_TOPK = 3;
const MAX_CONTEXT_CHUNKS = 4;
const KV_TTL_SECONDS = 60 * 30;
const AI_TIMEOUT_MS = 12000;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const corsOrigin = getCorsOrigin(request, env);

    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': corsOrigin,
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    try {
      // Route requests based on path
      if (url.pathname === '/health' && request.method === 'GET') {
        return await handleHealth();
      }

      if (url.pathname === '/v1/chat' && request.method === 'POST') {
        return await handleChat(request, env);
      }

      if (url.pathname === '/v1/feedback' && request.method === 'POST') {
        return await handleFeedback(request, env);
      }

      // 404 for unknown routes
      return new Response(JSON.stringify({ error: 'Not found' }), {
        status: 404,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': corsOrigin,
        },
      });

    } catch (error) {
      console.error('Request error:', error);
      return new Response(
        JSON.stringify({
          error: 'Internal server error',
          details: error.message
        }),
        {
          status: 500,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': corsOrigin,
          },
        }
      );
    }
  },
};

/**
 * Health check endpoint
 */
async function handleHealth() {
  return new Response(JSON.stringify({ status: 'ok' }), {
    headers: { 'Content-Type': 'application/json' },
  });
}

/**
 * Chat endpoint - main RAG functionality
 */
async function handleChat(request, env) {
  const corsOrigin = getCorsOrigin(request, env);
  const body = await request.json();

  // Validate request
  if (!body.message) {
    return new Response(JSON.stringify({ error: 'Message is required' }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': corsOrigin,
      },
    });
  }

  try {
    const jurisdiction = body.jurisdiction || 'auto';
    const topK = body.top_k || DEFAULT_TOPK;
    const cacheKey = await buildCacheKey(body.message, jurisdiction, topK);

    const cached = await getCache(env, cacheKey);
    if (cached) {
      return new Response(JSON.stringify(cached), {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': corsOrigin,
        },
      });
    }

    // Generate embedding for user query (cached)
    const queryEmbedding = await getEmbedding(body.message, env);

    // Search Vectorize for similar documents
    const vectorResults = await searchVectorize(queryEmbedding, env, {
      topK,
      jurisdiction,
    });

    let chunks = extractChunksFromVectorize(vectorResults).slice(0, MAX_CONTEXT_CHUNKS);

    if (env.USE_D1 === 'true' && env.DATABASE && vectorResults.length > 0) {
      try {
        const d1Chunks = await getChunksFromD1(vectorResults, env);
        if (d1Chunks.length > 0) {
          chunks = d1Chunks.slice(0, MAX_CONTEXT_CHUNKS);
        }
      } catch (error) {
        console.warn('D1 lookup failed, falling back to Vectorize metadata.', error);
      }
    }

    const retrievalUsed = chunks.length > 0;

    // Generate response using Cloudflare AI
    const answer = await generateAnswer(body.message, chunks, env);

    // Prepare response
    const response = {
      answer,
      chunks: chunks.map(chunk => ({
        id: chunk.id,
        source: chunk.source,
        jurisdiction: chunk.jurisdiction,
        text: chunk.text,
        score: chunk.score,
      })),
      retrieval_used: retrievalUsed,
      metadata: {
        jurisdiction,
        top_k: topK,
      },
    };

    await setCache(env, cacheKey, response);

    return new Response(JSON.stringify(response), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': corsOrigin,
      },
    });

  } catch (error) {
    console.error('Chat error:', error);
    return new Response(JSON.stringify({ error: 'Failed to process chat request' }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': corsOrigin,
      },
    });
  }
}

/**
 * Feedback endpoint
 */
async function handleFeedback(request, env) {
  const corsOrigin = getCorsOrigin(request, env);
  const body = await request.json();

  // Store feedback in D1 (simplified - just log for now)
  console.log('Feedback received:', body);

  return new Response(JSON.stringify({ status: 'received' }), {
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': corsOrigin,
    },
  });
}

/**
 * Generate embeddings using Cloudflare AI
 */
async function getEmbedding(text, env) {
  const cacheKey = await hashString(`embed:${text}`);
  const cached = await getCache(env, cacheKey);
  if (cached && cached.embedding) {
    return cached.embedding;
  }

  const response = await runWithTimeout(
    env.AI.run('@cf/baai/bge-base-en-v1.5', { text: [text] }),
    AI_TIMEOUT_MS
  );
  const embedding = response.data[0];
  await setCache(env, cacheKey, { embedding });
  return embedding;
}

/**
 * Search Vectorize index
 */
async function searchVectorize(embedding, env, options = {}) {
  const query = {
    vector: embedding,
    topK: options.topK || 5,
    returnValues: true,
    returnMetadata: true,
  };

  // Filter by jurisdiction if specified
  if (options.jurisdiction && options.jurisdiction !== 'auto') {
    query.filters = { jurisdiction: options.jurisdiction };
  }

  const results = await env.VECTORIZE_INDEX.query(query);
  return results.matches || [];
}

/**
 * Get chunk metadata from D1
 */
async function getChunksFromD1(vectorResults, env) {
  const chunkIds = vectorResults.map(result => result.id);

  // Query D1 for chunk metadata
  const placeholders = chunkIds.map((_, i) => `?${i + 1}`).join(',');
  const query = `SELECT * FROM chunks WHERE id IN (${placeholders})`;

  const result = await env.DATABASE.prepare(query).bind(...chunkIds).all();

  return result.results.map(row => ({
    id: row.id,
    source: row.source,
    jurisdiction: row.jurisdiction,
    text: row.text,
    score: vectorResults.find(r => r.id === row.id)?.score,
  }));
}

function extractChunksFromVectorize(vectorResults) {
  return vectorResults.map(result => ({
    id: result.id,
    source: result.metadata?.source || 'unknown',
    jurisdiction: result.metadata?.jurisdiction || 'federal',
    text: result.metadata?.text || '',
    score: result.score,
  }));
}

/**
 * Generate answer using Cloudflare AI
 */
async function generateAnswer(message, chunks, env) {
  let context = '';
  if (chunks.length > 0) {
    context = '\n\nRelevant legal information:\n' +
      chunks.map(chunk => `[${chunk.source}] ${chunk.text}`).join('\n\n');
  }

  const prompt = `You are a Nigerian legal assistant. Answer the following question accurately and cite relevant laws.

Question: ${message}

${context}

Instructions:
- Provide accurate, helpful legal information
- Cite specific laws, sections, and jurisdictions when possible
- If information is not available in the provided context, clearly state this
- Keep responses professional and objective
- Format citations properly (e.g., "Section 1 of the Criminal Code Act")

Answer:`;

  const response = await runWithTimeout(
    env.AI.run('@cf/meta/llama-3-8b-instruct', {
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.2,
      max_tokens: 800,
    }),
    AI_TIMEOUT_MS
  );

  return response.response;
}

async function buildCacheKey(message, jurisdiction, topK) {
  const payload = JSON.stringify({ message, jurisdiction, topK });
  return await hashString(`chat:${payload}`);
}

async function hashString(value) {
  const data = new TextEncoder().encode(value);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function getCache(env, key) {
  if (!env.LAWS_KV) return null;
  const cached = await env.LAWS_KV.get(key, 'json');
  return cached || null;
}

async function setCache(env, key, value) {
  if (!env.LAWS_KV) return;
  await env.LAWS_KV.put(key, JSON.stringify(value), { expirationTtl: KV_TTL_SECONDS });
}

async function runWithTimeout(promise, timeoutMs) {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error('AI request timed out')), timeoutMs);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function getCorsOrigin(request, env) {
  const origin = request.headers.get('Origin');
  const allowed = (env.CORS_ORIGINS || '*')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);

  if (!origin || allowed.includes('*')) {
    return '*';
  }

  if (allowed.includes(origin)) {
    return origin;
  }

  return allowed[0] || '*';
}
