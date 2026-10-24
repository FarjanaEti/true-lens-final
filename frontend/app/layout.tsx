import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TrueLens | Media verification",
  description: "Evidence-led image verification.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
