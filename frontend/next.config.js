/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy /api/* → FastAPI backend so we avoid CORS in production
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
