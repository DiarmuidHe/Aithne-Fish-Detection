import { describe, expect, it } from 'vitest';

import { KEY_BINDINGS, isTypingTarget, resolveCommand } from './keyboard-map';

describe('the review queue keyboard map', () => {
  it('binds every documented key to its command', () => {
    const resolved = Object.fromEntries(
      KEY_BINDINGS.map((binding) => [binding.command, resolveCommand({ key: binding.key })]),
    );

    expect(resolved).toEqual({
      next: 'next',
      previous: 'previous',
      accept: 'accept',
      reject: 'reject',
      flag: 'flag',
      unreview: 'unreview',
      'toggle-play': 'toggle-play',
      'open-source': 'open-source',
      back: 'back',
      help: 'help',
    });
  });

  it('matches the keys the prompt specified', () => {
    expect(resolveCommand({ key: 'j' })).toBe('next');
    expect(resolveCommand({ key: 'k' })).toBe('previous');
    expect(resolveCommand({ key: 'a' })).toBe('accept');
    expect(resolveCommand({ key: 'r' })).toBe('reject');
    expect(resolveCommand({ key: 'f' })).toBe('flag');
    expect(resolveCommand({ key: 'u' })).toBe('unreview');
    expect(resolveCommand({ key: ' ' })).toBe('toggle-play');
    expect(resolveCommand({ key: 'Enter' })).toBe('open-source');
    expect(resolveCommand({ key: 'Escape' })).toBe('back');
  });

  it('accepts a capital letter, so caps lock does not break the queue', () => {
    expect(resolveCommand({ key: 'A' })).toBe('accept');
    expect(resolveCommand({ key: 'J' })).toBe('next');
  });

  it('ignores keys it does not own', () => {
    expect(resolveCommand({ key: 'z' })).toBeNull();
    expect(resolveCommand({ key: 'Tab' })).toBeNull();
    expect(resolveCommand({ key: 'ArrowDown' })).toBeNull();
  });

  it('leaves modified keystrokes to the browser', () => {
    expect(resolveCommand({ key: 'a', ctrlKey: true })).toBeNull();
    expect(resolveCommand({ key: 'r', metaKey: true })).toBeNull();
    expect(resolveCommand({ key: 'f', altKey: true })).toBeNull();
  });

  it('does not steal keystrokes from whatever is being typed into', () => {
    const input = document.createElement('input');
    const textarea = document.createElement('textarea');
    const select = document.createElement('select');
    const editable = document.createElement('div');
    editable.contentEditable = 'true';

    for (const element of [input, textarea, select]) {
      expect(resolveCommand({ key: 'a', target: element })).toBeNull();
      expect(isTypingTarget(element)).toBe(true);
    }
    expect(isTypingTarget(document.createElement('div'))).toBe(false);
    expect(resolveCommand({ key: 'a', target: document.createElement('div') })).toBe('accept');
  });

  it('gives every binding a description for the help dialog', () => {
    for (const binding of KEY_BINDINGS) {
      expect(binding.description.length).toBeGreaterThan(0);
      expect(binding.display.length).toBeGreaterThan(0);
    }
  });

  it('binds each key exactly once', () => {
    const keys = KEY_BINDINGS.map((binding) => binding.key);

    expect(new Set(keys).size).toBe(keys.length);
  });
});
