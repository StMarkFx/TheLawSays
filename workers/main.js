/**
 * Cloudflare Workers entry point for TheLawSays backend.
 * Implements RAG functionality using Cloudflare AI and Vectorize.
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
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

      if (url.pathname === '/chat' && request.method === 'POST') {
        return await handleChat(request, env);
      }

      if (url.pathname === '/feedback' && request.method === 'POST') {
        return await handleFeedback(request, env);
      }

      // 404 for unknown routes
      return new Response(JSON.stringify({ error: 'Not found' }), {
        status: 404,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
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
            'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
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
  const body = await request.json();

  // Validate request
  if (!body.message) {
    return new Response(JSON.stringify({ error: 'Message is required' }), {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
      },
    });
  }

  try {
    // Generate embedding for user query
    const queryEmbedding = await generateEmbedding(body.message, env);

    // Search Vectorize for similar documents
    const vectorResults = await searchVectorize(queryEmbedding, env, {
      topK: body.top_k || 5,
      jurisdiction: body.jurisdiction,
    });

    // Classify intent to decide if retrieval is needed
    const intent = await classifyIntent(body.message, env);

    let chunks = [];
    let retrievalUsed = false;

    if (intent.retrieval_required && vectorResults.length > 0) {
      // Fetch chunk metadata from D1
      chunks = await getChunksFromD1(vectorResults, env);
      retrievalUsed = true;
    }

    // Generate response using Cloudflare AI
    const answer = await generateAnswer(body.message, chunks, intent, env);

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
        jurisdiction: body.jurisdiction,
        intent_reason: intent.reason,
        intent_label: intent.label,
      },
    };

    return new Response(JSON.stringify(response), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
      },
    });

  } catch (error) {
    console.error('Chat error:', error);
    return new Response(JSON.stringify({ error: 'Failed to process chat request' }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
      },
    });
  }
}

/**
 * Feedback endpoint
 */
async function handleFeedback(request, env) {
  const body = await request.json();

  // Store feedback in D1 (simplified - just log for now)
  console.log('Feedback received:', body);

  return new Response(JSON.stringify({ status: 'received' }), {
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': env.CORS_ORIGINS?.split(',')[0] || '*',
    },
  });
}

/**
 * Generate embeddings using Cloudflare AI
 */
async function generateEmbedding(text, env) {
  const response = await env.AI.run('@cf/baai/bge-base-en-v1.5', {
    text: [text],
  });
  return response.data[0];
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

/**
 * Classify user intent using Cloudflare AI
 */
async function classifyIntent(message, env) {
  const prompt = `
Classify the following legal question and decide if document retrieval is needed.

Question: "${message}"

Instructions:
- Determine if this question requires specific legal information from documents
- Return JSON with: {"retrieval_required": boolean, "reason": string, "label": string}

Examples:
- "What is the penalty for theft?" -> {"retrieval_required": true, "reason": "Question asks for specific legal information", "label": "legal_info"}
- "Hello, how are you?" -> {"retrieval_required": false, "reason": "Casual conversation", "label": "casual"}
- "Explain Nigerian law" -> {"retrieval_required": true, "reason": "Requests legal explanation", "label": "legal_explanation"}

Response must be valid JSON:
  `;

  const response = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.1,
  });

  try {
    const result = JSON.parse(response.response);
    return {
      retrieval_required: result.retrieval_required || false,
      reason: result.reason || 'Unknown',
      label: result.label || 'unknown',
    };
  } catch (error) {
    console.error('Intent classification parsing error:', error);
    return {
      retrieval_required: true, // Default to retrieval if parsing fails
      reason: 'Parsing error, defaulting to retrieval',
      label: 'default',
    };
  }
}

/**
 * Generate answer using Cloudflare AI
 */
async function generateAnswer(message, chunks, intent, env) {
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

  const response = await env.AI.run('@cf/meta/llama-3-8b-instruct', {
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.3,
    max_tokens: 1000,
  });

  return response.response;
}
