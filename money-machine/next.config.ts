import type { NextConfig } from "next";

const nextConfig: NextConfig = {
    output: 'export',
    images: {
        unoptimized: true,
    },
    distDir: 'out',
    // Tauri wraps a static export, so trailingSlash keeps deep links resolvable
    // when loaded from the file:// protocol inside the bundled app.
    trailingSlash: true,
};

export default nextConfig;
