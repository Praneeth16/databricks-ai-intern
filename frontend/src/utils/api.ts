/**
 * Centralized API utilities.
 *
 * In production: HttpOnly cookie (hf_access_token) is sent automatically.
 * In development: auth is bypassed on the backend.
 */

import { triggerLogin } from '@/hooks/useAuth';

/** Wrapper around fetch with credentials and common headers. */
export async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  const response = await fetch(path, {
    ...options,
    headers,
    credentials: 'include', // Send cookies with every request
  });

  // Handle 401 — redirect to login
  if (response.status === 401) {
    try {
      const authStatus = await fetch('/auth/status', { credentials: 'include' });
      const data = await authStatus.json();
      if (data.auth_enabled) {
        triggerLogin();
        throw new Error('Authentication required — redirecting to login.');
      }
    } catch (e) {
      if (e instanceof Error && e.message.includes('redirecting')) throw e;
    }
  }

  return response;
}


export interface ApiUploadProgress {
  /** 0..100 when computable, null when the body length is unknown. */
  percent: number | null;
  loaded: number;
  total: number | null;
}


/** XHR-based multipart upload so we get an ``onProgress`` callback that
 *  ``fetch`` doesn't surface. Returns a ``Response``-shaped wrapper so
 *  callers using ``apiFetch`` can drop in with minimal change.
 */
export async function apiUpload(
  path: string,
  formData: FormData,
  options: {
    method?: 'POST' | 'PUT';
    onProgress?: (progress: ApiUploadProgress) => void;
    signal?: AbortSignal;
  } = {},
): Promise<Response> {
  const { method = 'POST', onProgress, signal } = options;

  return new Promise<Response>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, path, true);
    xhr.withCredentials = true;
    if (onProgress && xhr.upload) {
      xhr.upload.onprogress = (event) => {
        onProgress({
          percent: event.lengthComputable
            ? Math.round((event.loaded / event.total) * 100)
            : null,
          loaded: event.loaded,
          total: event.lengthComputable ? event.total : null,
        });
      };
    }
    xhr.onload = () => {
      const headers = new Headers();
      xhr.getAllResponseHeaders().split('\r\n').forEach((line) => {
        const [k, v] = line.split(': ');
        if (k && v) headers.set(k, v);
      });
      resolve(new Response(xhr.responseText, {
        status: xhr.status,
        statusText: xhr.statusText,
        headers,
      }));
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.onabort = () => reject(new Error('Upload aborted'));
    if (signal) {
      if (signal.aborted) { xhr.abort(); return; }
      signal.addEventListener('abort', () => xhr.abort(), { once: true });
    }
    xhr.send(formData);
  });
}