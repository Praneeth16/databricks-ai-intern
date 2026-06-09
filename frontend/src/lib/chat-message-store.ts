/**
 * Lightweight localStorage persistence for UIMessage arrays,
 * keyed by session ID.
 *
 * Uses the same storage namespace (`hf-agent-messages`) that the
 * old Zustand-based store used, so existing data is compatible.
 */
import type { UIMessage } from 'ai';
import { logger } from '@/utils/logger';

const STORAGE_KEY = 'hf-agent-messages';
const ACCESS_KEY = 'hf-agent-messages-access';
const MAX_SESSIONS = 50;

type MessagesMap = Record<string, UIMessage[]>;

// lastAccessedAt per session — sessions missing an entry (pre-LRU data)
// sort as oldest and get evicted first.
function readAccess(): Record<string, number> {
  try {
    const raw = localStorage.getItem(ACCESS_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeAccess(map: Record<string, number>): void {
  try {
    localStorage.setItem(ACCESS_KEY, JSON.stringify(map));
  } catch (e) {
    logger.warn('Failed to persist message access times:', e);
  }
}

function touchAccess(sessionId: string): void {
  const access = readAccess();
  access[sessionId] = Date.now();
  writeAccess(access);
}

function readAll(): MessagesMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    // Legacy format was { messagesBySession: {...} }
    if (parsed.messagesBySession) return parsed.messagesBySession;
    // New flat format
    if (typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    return {};
  } catch {
    return {};
  }
}

function writeAll(map: MessagesMap): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch (e) {
    logger.warn('Failed to persist messages:', e);
  }
}

export function loadMessages(sessionId: string): UIMessage[] {
  const map = readAll();
  const messages = map[sessionId] ?? [];
  if (map[sessionId]) touchAccess(sessionId);
  return messages;
}

export function saveMessages(sessionId: string, messages: UIMessage[]): void {
  const map = readAll();
  map[sessionId] = messages;

  const access = readAccess();
  access[sessionId] = Date.now();

  // Evict least-recently-accessed sessions if we exceed the cap
  const keys = Object.keys(map);
  if (keys.length > MAX_SESSIONS) {
    const byOldestAccess = keys.sort((a, b) => (access[a] ?? 0) - (access[b] ?? 0));
    const toRemove = byOldestAccess.slice(0, keys.length - MAX_SESSIONS);
    for (const k of toRemove) {
      delete map[k];
      delete access[k];
    }
  }

  writeAll(map);
  writeAccess(access);
}

export function deleteMessages(sessionId: string): void {
  const map = readAll();
  delete map[sessionId];
  writeAll(map);
  const access = readAccess();
  delete access[sessionId];
  writeAccess(access);
}

export function moveMessages(fromId: string, toId: string): void {
  const map = readAll();
  if (!map[fromId]) return;
  map[toId] = map[fromId];
  delete map[fromId];
  writeAll(map);
  const access = readAccess();
  access[toId] = access[fromId] ?? Date.now();
  delete access[fromId];
  writeAccess(access);
}
