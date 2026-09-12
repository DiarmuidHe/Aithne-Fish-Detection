import * as React from 'react';
import { createPortal } from 'react-dom';
import { NavLink, Outlet, useLocation } from 'react-router';
import { useIsFetching, useQuery } from '@tanstack/react-query';

import { fetchSystemStatus } from '@/api/system';
import { fetchFacets } from '@/api/videos';
import { Skeleton } from '@/components/base/feedback';
import { Hint } from '@/components/base/popups';
import {
  BroadcastIcon,
  CollapseIcon,
  FilmIcon,
  FishMark,
  GaugeIcon,
  ReviewIcon,
  ServerIcon,
} from '@/icons';
import { useConditionalPoll } from '@/hooks/use-poll';
import { formatCount } from '@/lib/format';

import { ThemeToggle } from './theme';
import styles from './shell.module.css';

/*
 * The shell is a marine spine and a light workspace. The rail and the
 * application bar are fixed chrome that never scrolls; everything a route
 * renders lives in the region below the bar and owns its own scrolling.
 */

interface NavEntry {
  to: string;
  label: string;
  icon: React.ReactNode;
  end?: boolean;
  /** Shown as a count on the rail, and as the bar's page title elsewhere. */
  count?: number;
}

interface NavGroup {
  label: string;
  items: NavEntry[];
}

/** Titles for the application bar, longest path first so nesting resolves. */
const PAGE_TITLES: [string, string][] = [
  ['/library', 'Recordings'],
  ['/review', 'Review queue'],
  ['/live', 'Live monitoring'],
  ['/settings', 'System'],
  ['/', 'Overview'],
];

function titleFor(pathname: string): string {
  return PAGE_TITLES.find(([path]) => pathname === path || pathname.startsWith(`${path}/`))?.[1] ??
    'Overview';
}

/* --- Application bar slots -------------------------------------------------
 * Routes render their own context line and actions into the bar through
 * portals rather than through synchronised state, so a route that unmounts
 * takes its controls with it and nothing has to be cleaned up by hand.
 */

const SlotContext = React.createContext<{
  context: HTMLElement | null;
  actions: HTMLElement | null;
}>({ context: null, actions: null });

export function AppBarSlot({
  area,
  children,
}: {
  area: 'context' | 'actions';
  children: React.ReactNode;
}) {
  const slots = React.useContext(SlotContext);
  const target = area === 'context' ? slots.context : slots.actions;
  return target ? createPortal(children, target) : null;
}

export function AppShell() {
  const location = useLocation();
  const [collapsed, setCollapsed] = React.useState(readCollapsed);
  const [contextSlot, setContextSlot] = React.useState<HTMLElement | null>(null);
  const [actionsSlot, setActionsSlot] = React.useState<HTMLElement | null>(null);

  const system = useQuery({
    queryKey: ['system-status'],
    queryFn: fetchSystemStatus,
    refetchInterval: useConditionalPoll(true) && 15_000,
    refetchOnWindowFocus: true,
  });

  const facets = useQuery({ queryKey: ['facets'], queryFn: fetchFacets, staleTime: 30_000 });

  const outstanding =
    (facets.data?.review_status.awaiting ?? 0) + (facets.data?.review_status['in-progress'] ?? 0);

  const groups: NavGroup[] = [
    {
      label: 'Monitoring',
      items: [
        { to: '/', label: 'Overview', icon: <GaugeIcon />, end: true },
        { to: '/live', label: 'Live monitoring', icon: <BroadcastIcon /> },
      ],
    },
    {
      label: 'Records',
      items: [
        { to: '/library', label: 'Recordings', icon: <FilmIcon />, count: facets.data?.total },
        {
          to: '/review',
          label: 'Review queue',
          icon: <ReviewIcon />,
          count: outstanding > 0 ? outstanding : undefined,
        },
      ],
    },
    {
      label: 'Operations',
      items: [{ to: '/settings', label: 'System', icon: <ServerIcon /> }],
    },
  ];

  React.useEffect(() => {
    try {
      window.localStorage.setItem('rail-collapsed', collapsed ? '1' : '0');
    } catch {
      // The rail state is a convenience; losing it costs nothing.
    }
  }, [collapsed]);

  const slots = React.useMemo(
    () => ({ context: contextSlot, actions: actionsSlot }),
    [contextSlot, actionsSlot],
  );

  return (
    <SlotContext.Provider value={slots}>
    <div className={styles.shell} data-collapsed={collapsed || undefined}>
      <nav className={styles.rail} aria-label="Sections">
        <div className={styles.brand}>
          <NavLink to="/" className={styles.brandLink} title="Fish Monitor">
            <FishMark className={styles.brandMark} />
            <span className={styles.brandText}>
              <span className={styles.brandName}>Fish Monitor</span>
              <span className={styles.brandSub}>Detection operations</span>
            </span>
          </NavLink>
          <button
            type="button"
            className={styles.collapseButton}
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? 'Expand the navigation rail' : 'Collapse the navigation rail'}
            aria-pressed={collapsed}
          >
            <CollapseIcon />
          </button>
        </div>

        <div className={styles.nav}>
          {groups.map((group) => (
            <div className={styles.navGroup} key={group.label}>
              <span className={styles.navGroupLabel} aria-hidden={collapsed || undefined}>
                {group.label}
              </span>
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={styles.navItem}
                  title={collapsed ? item.label : undefined}
                >
                  <span className={styles.navIcon}>{item.icon}</span>
                  <span className={styles.navLabel}>{item.label}</span>
                  {item.count !== undefined ? (
                    <span className={styles.navCount}>{formatCount(item.count)}</span>
                  ) : null}
                </NavLink>
              ))}
            </div>
          ))}
        </div>

        <div className={styles.railFooter}>
          {system.data ? (
            <Hint content={`${system.data.worker.message}. ${system.data.database.message}.`}>
              <NavLink to="/settings" className={styles.health} data-ready={system.data.ready || undefined}>
                <span className={styles.healthDot} aria-hidden="true" />
                <span className={styles.healthText}>
                  <span className={styles.healthTitle}>
                    {system.data.ready ? 'Worker ready' : 'No worker'}
                  </span>
                  <span className={styles.healthMeta}>
                    {system.data.processing_mode === 'gpu' ? 'VIAME GPU' : 'Mock'} ·{' '}
                    {system.data.queue.queued + system.data.queue.processing} in flight
                  </span>
                </span>
              </NavLink>
            </Hint>
          ) : (
            <div className={styles.health} aria-hidden="true">
              <span className={styles.healthDot} />
              <span className={styles.healthText}>
                <span className={styles.healthTitle}>Checking…</span>
              </span>
            </div>
          )}
        </div>
      </nav>

      <div className={styles.workspace}>
        <header className={styles.appBar}>
          <div className={styles.appBarTitle}>
            <h1 className={styles.appBarHeading}>{titleFor(location.pathname)}</h1>
            <div className={styles.appBarContext} ref={setContextSlot} />
          </div>
          <div className={styles.appBarActions}>
            <div className={styles.appBarSlot} ref={setActionsSlot} />
            <ThemeToggle />
          </div>
          <RouteProgress />
        </header>

        <main className={styles.main}>
          <Outlet />
        </main>
      </div>
    </div>
    </SlotContext.Provider>
  );
}

/**
 * A hairline at the top of the workspace that fills while anything is in
 * flight. It waits 400 ms before showing so a cached response never produces a
 * flash, and it runs indeterminate because the API reports no progress.
 */
function RouteProgress() {
  const fetching = useIsFetching();
  const [visible, setVisible] = React.useState(false);

  React.useEffect(() => {
    if (fetching === 0) {
      setVisible(false);
      return;
    }
    const timer = window.setTimeout(() => setVisible(true), 400);
    return () => window.clearTimeout(timer);
  }, [fetching]);

  return (
    <div
      className={styles.routeProgress}
      data-active={visible || undefined}
      role="progressbar"
      aria-label="Loading"
      aria-busy={visible}
      aria-hidden={!visible}
    >
      <span className={styles.routeProgressBar} />
    </div>
  );
}

function readCollapsed(): boolean {
  try {
    return window.localStorage.getItem('rail-collapsed') === '1';
  } catch {
    return false;
  }
}

/* --- Page scaffolding ------------------------------------------------------ */

/**
 * A scrolling page region. The bar already carries the title, so `title` here
 * is the optional in-page heading for routes that want one restated.
 */
export function Page({
  title,
  subtitle,
  actions,
  children,
  width = 'default',
}: {
  title?: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  width?: 'default' | 'full';
}) {
  return (
    <div className={styles.page}>
      <div className={styles.pageInner} data-width={width}>
        {title || subtitle || actions ? (
          <header className={styles.pageHeader}>
            <div>
              {title ? <h2 className={styles.pageTitle}>{title}</h2> : null}
              {subtitle ? <p className={styles.pageSubtitle}>{subtitle}</p> : null}
            </div>
            {actions}
          </header>
        ) : null}
        {children}
      </div>
    </div>
  );
}

export function Section({
  title,
  note,
  action,
  children,
}: {
  title: string;
  note?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeading}>
          <h2 className={styles.sectionTitle}>{title}</h2>
          {note ? <p className={styles.sectionNote}>{note}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

/**
 * A bordered region. Panels separate content; they do not float. The only
 * elevation in the system belongs to portals.
 */
export function Panel({
  title,
  note,
  action,
  children,
  flush = false,
  className,
}: {
  title?: React.ReactNode;
  note?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  /** Drop the body padding, for tables and media that meet the border. */
  flush?: boolean;
  className?: string;
}) {
  return (
    <section className={[styles.panel, className].filter(Boolean).join(' ')}>
      {title ? (
        <header className={styles.panelHeader}>
          <div className={styles.panelHeading}>
            <h3 className={styles.panelTitle}>{title}</h3>
            {note ? <p className={styles.panelNote}>{note}</p> : null}
          </div>
          {action}
        </header>
      ) : null}
      <div className={styles.panelBody} data-flush={flush || undefined}>
        {children}
      </div>
    </section>
  );
}

/**
 * A full-bleed section band.
 *
 * A page is not one ground with cards floating on it. Bands run edge to edge
 * and change the material under a section — console dark for the operational
 * header, sunken for media, plain for everything else — so the rhythm of a
 * page comes from the ground rather than from multiplying bordered boxes.
 */
export function Band({
  tone = 'plain',
  label,
  note,
  action,
  children,
  className,
}: {
  tone?: 'plain' | 'sunken';
  /** The eyebrow. Bands label themselves with a rule, not with a panel header. */
  label?: string;
  note?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={[styles.band, className].filter(Boolean).join(' ')}
      data-tone={tone}
    >
      {label || action ? (
        <header className={styles.bandHeader}>
          <div className={styles.bandHeading}>
            {label ? <h2 className={styles.bandLabel}>{label}</h2> : null}
            {note ? <p className={styles.bandNote}>{note}</p> : null}
          </div>
          {action}
        </header>
      ) : null}
      <div className={styles.bandBody}>{children}</div>
    </section>
  );
}

/**
 * One operational figure inside a strip. The label reads first, the number
 * second, and the supporting line says what the number means — no gauge, no
 * ring, no card of its own. Cells are divided by a hairline, not by a gap.
 */
export function MetricCard({
  label,
  value,
  meta,
  tone,
  to,
}: {
  label: string;
  value: React.ReactNode;
  meta?: React.ReactNode;
  tone?: 'active' | 'failed' | 'flagged';
  to?: string;
}) {
  const body = (
    <>
      <span className={styles.metricLabel}>{label}</span>
      <span className={styles.metricValue} data-tone={tone}>
        {value}
      </span>
      {meta ? <span className={styles.metricMeta}>{meta}</span> : null}
    </>
  );

  return to ? (
    <NavLink to={to} className={styles.metric} data-interactive="">
      {body}
    </NavLink>
  ) : (
    <div className={styles.metric}>{body}</div>
  );
}

/**
 * A strip cell standing in for a figure that has not arrived. It carries the
 * cell's own ground so the hairline dividers stay where they will be once the
 * numbers land, and the strip does not change shape underneath the reader.
 */
export function MetricPlaceholder() {
  return (
    <div className={styles.metric} data-placeholder="" aria-hidden="true">
      <Skeleton width="52%" height={10} />
      <Skeleton width="34%" height={26} />
      <Skeleton width="74%" height={10} />
    </div>
  );
}

/**
 * The status strip: one surface carrying every headline figure, divided
 * internally by hairlines. Five figures on five cards is a dashboard; five
 * figures on one instrument is a console.
 */
export function MetricRow({
  children,
  tone = 'plain',
  aside,
}: {
  children: React.ReactNode;
  /** `console` puts the strip on the marine spine's own ground. */
  tone?: 'plain' | 'console';
  /** A trailing cell that reports the state of the system rather than a count. */
  aside?: React.ReactNode;
}) {
  return (
    <div className={styles.metricRow} data-tone={tone}>
      <div className={styles.metricCells}>{children}</div>
      {aside ? <div className={styles.metricAside}>{aside}</div> : null}
    </div>
  );
}

export { styles as shellStyles };
