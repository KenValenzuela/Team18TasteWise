import type { Metadata } from "next";
import "./globals.css";
import LoadingScreen from "@/components/LoadingScreen";

export const metadata: Metadata = {
  title: "Tastewise — Restaurant Intelligence",
  description:
    "Yelp-powered restaurant recommender using sentiment analysis and BERTopic. Team 18 · CIS 509.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0908] overflow-x-hidden">
        <LoadingScreen>{children}</LoadingScreen>
      </body>
    </html>
  );
}
