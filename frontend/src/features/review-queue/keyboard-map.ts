/**
 * The review queue's keyboard map.
 *
 * Kept as data so the help dialog, the on-screen key hints and the handler can
 * never disagree — and so it can be tested without a DOM.
 */

export type QueueCommand =
  | 'next'
  | 'previous'
  | 'accept'
  | 'reject'
  | 'flag'
  | 'unreview'
  | 'toggle-play'
  | 'open-source'
  | 'back'
  | 'help';

export interface KeyBinding {
  key: string;
  display: string;
  command: QueueCommand;
  description: string;
}

export const KEY_BINDINGS: KeyBinding[] = [
  { key: 'j', display: 'j', command: 'next', description: 'Next track' },
  { key: 'k', display: 'k', command: 'previous', description: 'Previous track' },
  { key: 'a', display: 'a', command: 'accept', description: 'Accept — count this fish' },
  { key: 'r', display: 'r', command: 'reject', description: 'Reject — do not count it' },
  { key: 'f', display: 'f', command: 'flag', description: 'Flag as needing review' },
  { key: 'u', display: 'u', command: 'unreview', description: 'Return to unreviewed' },
  { key: ' ', display: 'Space', command: 'toggle-play', description: 'Play or pause the clip' },
  { key: 'enter', display: 'Enter', command: 'open-source', description: 'Open the source video' },
  { key: 'escape', display: 'Esc', command: 'back', description: 'Back to the library' },
  { key: '?', display: '?', command: 'help', description: 'Show this list' },
];

const BY_KEY = new Map(KEY_BINDINGS.map((binding) => [binding.key, binding.command]));

export interface KeyEventLike {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  target?: EventTarget | null;
}

/** True when the keystroke belongs to whatever the operator is typing into. */
export function isTypingTarget(target: EventTarget | null | undefined): boolean {
  if (!target || !(target as HTMLElement).tagName) return false;
  const element = target as HTMLElement;
  const tag = element.tagName.toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
  // A detached element reports `undefined` here rather than false.
  return element.isContentEditable === true;
}

/**
 * Resolve a keystroke to a command, or null when it is not ours. Modified keys
 * are always left to the browser so shortcuts like ctrl+F keep working.
 */
export function resolveCommand(event: KeyEventLike): QueueCommand | null {
  if (event.ctrlKey || event.metaKey || event.altKey) return null;
  if (isTypingTarget(event.target)) return null;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key.toLowerCase();
  return BY_KEY.get(key) ?? null;
}
