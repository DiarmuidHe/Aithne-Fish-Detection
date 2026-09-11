/**
 * Light and dark, following the system by default with a manual override.
 * Operators run this in dim rooms, so the override is one control, not a
 * settings screen.
 */

import * as React from 'react';

import { MoonIcon, SunIcon } from '@/icons';
import { Segmented } from '@/components/base/controls';

export type ThemeChoice = 'system' | 'light' | 'dark';

const STORAGE_KEY = 'theme';

function readStored(): ThemeChoice {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === 'light' || raw === 'dark' || raw === 'system') return raw;
  } catch {
    // Storage can be refused; the system preference still applies.
  }
  return 'system';
}

export function useTheme(): [ThemeChoice, (choice: ThemeChoice) => void] {
  const [choice, setChoice] = React.useState<ThemeChoice>(readStored);

  React.useEffect(() => {
    const root = document.documentElement;
    if (choice === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', choice);
    try {
      window.localStorage.setItem(STORAGE_KEY, choice);
    } catch {
      // Nothing to do: the choice still applies for this session.
    }
  }, [choice]);

  return [choice, setChoice];
}

export function ThemeToggle() {
  const [choice, setChoice] = useTheme();
  return (
    <Segmented
      label="Colour theme"
      size="small"
      value={[choice]}
      onChange={(next) => setChoice((next[0] as ThemeChoice) ?? 'system')}
      options={[
        { value: 'system', label: 'Auto', title: 'Follow the system theme' },
        { value: 'light', label: <SunIcon />, title: 'Light theme' },
        { value: 'dark', label: <MoonIcon />, title: 'Dark theme' },
      ]}
    />
  );
}
