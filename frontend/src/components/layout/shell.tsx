import * as React from 'react';
import { NavLink, Outlet } from 'react-router';
import { useQuery } from '@tanstack/react-query';

import { fetchSystemStatus } from '@/api/system';
import { fetchFacets } from '@/api/videos';
import { StatusPill } from '@/components/base/feedback';
import { Hint } from '@/components/base/popups';
import {
  BroadcastIcon,
  FishMark,
  GaugeIcon,
  LibraryIcon,
  ReviewIcon,
} from '@/icons';
import { useConditionalPoll } from '@/hooks/use-poll';

import { ThemeToggle } from './theme';
import styles from './shell.module.css';

export function AppShell() {
  const system = useQuery({
    queryKey: ['system-status'],
    queryFn: fetchSystemStatus,
    // Only worth watching while the queue has something in it.
    refetchInterval: useConditionalPoll(true) && 15_000,
    refetchOnWindowFocus: true,
  });

  const facets = useQuery({ queryKey: ['facets'], queryFn: fetchFacets, staleTime: 30_000 });

  const outstanding =
    (facets.data?.review_status.awaiting ?? 0) + (facets.data?.review_status['in-progress'] ?? 0);

  return (
    <div className={styles.shell}>
      <nav className={styles.rail} aria-label="Sections">
        <NavLink to="/" className={styles.brand}>
          <FishMark style={{ color: 'var(--accent)' }} />
          <span className={styles.brandName}>Fish Monitor</span>
        </NavLink>

        <div className={styles.nav}>
          <NavLink to="/" end className={styles.navItem}>
            <GaugeIcon />
            Overview
          </NavLink>
          <NavLink to="/library" className={styles.navItem}>
            <LibraryIcon />
            Library
            {facets.data ? <span className={styles.navCount}>{facets.data.total}</span> : null}
          </NavLink>
          <NavLink to="/review" className={styles.navItem}>
            <ReviewIcon />
            Review
            {outstanding > 0 ? <span className={styles.navCount}>{outstanding}</span> : null}
          </NavLink>
          <NavLink to="/live" className={styles.navItem}>
            <BroadcastIcon />
            Live
          </NavLink>
        </div>

        <div className={styles.railFooter}>
          {system.data ? (
            <div className={styles.modeLine}>
              <span>Processing mode</span>
              <span className={styles.modeValue}>
                {system.data.processing_mode === 'gpu' ? 'VIAME GPU' : 'Mock'}
                <StatusPill status={system.data.ready ? 'completed' : 'failed'}>
                  {system.data.ready ? 'Worker ready' : 'No worker'}
                </StatusPill>
              </span>
              <Hint content={system.data.worker.message}>
                <span className="faint truncate">
                  {system.data.queue.queued} queued · {system.data.queue.processing} processing
                </span>
              </Hint>
            </div>
          ) : null}
          <ThemeToggle />
        </div>
      </nav>

      <main className={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}

export function Page({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className={styles.page}>
      <div className={styles.pageInner}>
        <header className={styles.pageHeader}>
          <div>
            <h1 className={styles.pageTitle}>{title}</h1>
            {subtitle ? <p className={styles.pageSubtitle}>{subtitle}</p> : null}
          </div>
          {actions}
        </header>
        {children}
      </div>
    </div>
  );
}

export function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <h2 className={styles.sectionTitle}>{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

export { styles as shellStyles };
