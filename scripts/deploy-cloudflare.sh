#!/bin/bash

# Cloudflare Migration Deployment Script
# This script helps deploy TheLawSays to Cloudflare Workers and Pages

set -e

echo "🚀 Starting Cloudflare Migration Deployment"

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "❌ Wrangler CLI not found. Please install it first:"
    echo "npm install -g wrangler"
    exit 1
fi

# Check if logged in to Cloudflare
if ! wrangler auth login --check; then
    echo "🔐 Please login to Cloudflare:"
    wrangler auth login
fi

echo "📦 Step 1: Deploying Backend to Cloudflare Workers"

# Deploy backend worker
wrangler deploy

echo "✅ Backend deployed successfully!"

echo "🌐 Step 2: Deploying Frontend to Cloudflare Pages"

# Change to web directory
cd web

# Build the Next.js app
echo "🔨 Building Next.js application..."
npm run build

# Deploy to Cloudflare Pages
echo "🚀 Deploying to Cloudflare Pages..."
npx wrangler pages deploy dist --compatibility-date 2024-01-01

echo "✅ Frontend deployed successfully!"

cd ..

echo "🎉 Migration completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Update DNS records to point to Cloudflare"
echo "2. Update environment variables in Cloudflare dashboard"
echo "3. Test the application"
echo "4. Gradually migrate FastAPI logic to Workers (if needed)"
echo ""
echo "🔗 Your application should be available at:"
echo "- Frontend: https://thelawsays-frontend.pages.dev"
echo "- Backend: https://thelawsays-backend.your-subdomain.workers.dev"