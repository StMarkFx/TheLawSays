/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Cloudflare Pages optimizations
  images: {
    loader: 'custom',
    loaderFile: './image-loader.js',
  },

  // Optimize for edge runtime
  experimental: {
    runtime: 'edge',
  },

  // Output configuration for Cloudflare Pages
  output: 'export',
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
  distDir: 'dist',

  // Environment variables
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL,
  },
};

export default nextConfig;
