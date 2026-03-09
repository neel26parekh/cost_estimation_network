const backendBaseUrl = process.env.API_BASE_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const backendApiKey = process.env.API_KEY ?? "";

type BackendRequestInit = RequestInit & {
  path: string;
};

type BackendFallbackRequestInit = RequestInit & {
  paths: string[];
};

export async function backendFetch({ path, headers, ...init }: BackendRequestInit) {
  return fetch(`${backendBaseUrl}${path}`, {
    ...init,
    headers: {
      ...(headers ?? {}),
      ...(backendApiKey ? { "x-api-key": backendApiKey } : {}),
    },
    cache: "no-store",
  });
}

export async function backendFetchWithFallback({ paths, ...init }: BackendFallbackRequestInit) {
  let lastResponse: Response | null = null;

  for (const path of paths) {
    const response = await backendFetch({ path, ...init });
    lastResponse = response;

    if (response.status !== 404) {
      return response;
    }
  }

  if (lastResponse) {
    return lastResponse;
  }

  throw new Error("No backend paths provided.");
}