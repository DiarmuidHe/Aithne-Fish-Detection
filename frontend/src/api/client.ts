/**
 * One fetch wrapper for the whole app.
 *
 * The API reports failures as `{"detail": "..."}`; those sentences are written
 * for operators, so they are surfaced verbatim rather than replaced with a
 * generic message.
 */

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export type QueryValue = string | number | boolean | null | undefined;

/** Build a query string, repeating a key for each value of an array. */
export function toQuery(params: Record<string, QueryValue | QueryValue[]>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    for (const entry of Array.isArray(value) ? value : [value]) {
      if (entry === undefined || entry === null || entry === '') continue;
      search.append(key, String(entry));
    }
  }
  const query = search.toString();
  return query ? `?${query}` : '';
}

async function failure(response: Response): Promise<never> {
  let detail: string | undefined;
  try {
    const payload = await response.json();
    if (typeof payload?.detail === 'string') detail = payload.detail;
    else if (Array.isArray(payload?.detail)) detail = payload.detail[0]?.msg;
  } catch {
    // A non-JSON body tells us nothing useful; fall through to the status.
  }
  throw new ApiError(detail ?? `Request failed (${response.status})`, response.status);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) return failure(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** Like `request`, but hands back the response so callers can read headers. */
export async function requestWithResponse<T>(
  path: string,
  init?: RequestInit,
): Promise<{ data: T; response: Response }> {
  const response = await fetch(path, init);
  if (!response.ok) return failure(response);
  return { data: (await response.json()) as T, response };
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function patchJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
