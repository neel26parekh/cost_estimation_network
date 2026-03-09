import { NextResponse } from "next/server";

import { backendFetchWithFallback } from "@/lib/backend";

export async function GET() {
  try {
    const response = await backendFetchWithFallback({ paths: ["/v1/monitoring/summary", "/monitoring/summary"], method: "GET" });
    const payload = await response.json();
    return NextResponse.json(payload, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the prediction backend." }, { status: 503 });
  }
}