import { NextRequest, NextResponse } from "next/server";

import { backendFetchWithFallback } from "@/lib/backend";

export async function GET(request: NextRequest) {
  try {
    const limit = request.nextUrl.searchParams.get("limit") ?? "5";
    const response = await backendFetchWithFallback({
      paths: [`/v1/predictions/recent?limit=${limit}`, `/predictions/recent?limit=${limit}`],
      method: "GET",
    });
    const payload = await response.json();
    return NextResponse.json(payload, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the prediction backend." }, { status: 503 });
  }
}