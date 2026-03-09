import { NextRequest, NextResponse } from "next/server";

import { backendFetchWithFallback } from "@/lib/backend";

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const response = await backendFetchWithFallback({
      paths: ["/v1/predict", "/predict"],
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the prediction backend." }, { status: 503 });
  }
}