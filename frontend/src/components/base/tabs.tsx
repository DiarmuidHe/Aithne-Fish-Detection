import * as React from 'react';
import { Tabs } from '@base-ui/react/tabs';

import styles from './tabs.module.css';

export interface TabDefinition {
  value: string;
  label: string;
  /** A count beside the label, e.g. how many tracks the tab holds. */
  count?: number;
  disabled?: boolean;
}

export function TabBar({
  tabs,
  value,
  onValueChange,
  children,
  label,
}: {
  tabs: TabDefinition[];
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
  label: string;
}) {
  return (
    <Tabs.Root
      className={styles.root}
      value={value}
      onValueChange={(next) => onValueChange(next as string)}
    >
      <Tabs.List className={styles.list} aria-label={label}>
        {tabs.map((tab) => (
          <Tabs.Tab
            key={tab.value}
            value={tab.value}
            className={styles.tab}
            disabled={tab.disabled}
          >
            {tab.label}
            {tab.count === undefined ? null : (
              <span className={styles.count}>{tab.count}</span>
            )}
          </Tabs.Tab>
        ))}
        <Tabs.Indicator className={styles.indicator} />
      </Tabs.List>
      {children}
    </Tabs.Root>
  );
}

export function TabPanel({
  value,
  children,
}: {
  value: string;
  children: React.ReactNode;
}) {
  return (
    <Tabs.Panel value={value} className={styles.panel}>
      {children}
    </Tabs.Panel>
  );
}
