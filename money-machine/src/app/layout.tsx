import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Money Machine | AlphaAxiom",
  description: "AI-powered trading overlay for AlphaAxiom.",
};

/**
 * Top-level HTML layout that applies global font variables and renders the app content.
 *
 * @param children - The page or route content to render inside the document body.
 * @returns The root `<html>` element containing a `<body>` with global font classes and `children`.
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
