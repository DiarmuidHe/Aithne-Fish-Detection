import { useQuery } from '@tanstack/react-query';

import { fetchSystemStatus } from '@/api/system';
import { Button } from '@/components/base/button';
import {
  MetricCard,
  MetricPlaceholder,
  MetricRow,
  Page,
  Panel,
  Section,
  AppBarSlot,
} from '@/components/layout/shell';
import { PanelError, Skeleton, StatusPill } from '@/components/base/feedback';
import { RefreshIcon } from '@/icons';
import { useConditionalPoll } from '@/hooks/use-poll';
import { formatClock, formatCount } from '@/lib/format';
import { ThemeToggle } from '@/components/layout/theme';

import styles from './system.module.css';

/*
 * Where the technical state of the stack lives, so the Overview can stay about
 * fish. Everything here comes from the one status endpoint the rail already
 * polls; this page just shows all of it instead of a summary line.
 */

export function SystemRoute() {
  const system = useQuery({
    queryKey: ['system-status'],
    queryFn: fetchSystemStatus,
    refetchInterval: useConditionalPoll(true) && 15_000,
  });

  if (system.isLoading) {
    return (
      <Page>
        <MetricRow>
          {Array.from({ length: 4 }, (_, index) => (
            <MetricPlaceholder key={index} />
          ))}
        </MetricRow>
        <Skeleton height={180} radius={6} />
      </Page>
    );
  }

  if (system.isError || !system.data) {
    return (
      <Page>
        <PanelError
          title="System status is unavailable"
          message={system.error instanceof Error ? system.error.message : 'Request failed'}
          onRetry={() => void system.refetch()}
        />
      </Page>
    );
  }

  const status = system.data;
  const gpu = status.processing_mode === 'gpu';

  return (
    <Page>
      <AppBarSlot area="context">
        <span>{gpu ? 'VIAME GPU' : 'Mock processing'}</span>
      </AppBarSlot>
      <AppBarSlot area="actions">
        <Button size="small" onClick={() => void system.refetch()} disabled={system.isFetching}>
          <RefreshIcon />
          {system.isFetching ? 'Refreshing' : 'Refresh'}
        </Button>
      </AppBarSlot>

      <MetricRow>
        <MetricCard
          label="Queued"
          value={formatCount(status.queue.queued)}
          meta="Waiting for a worker"
          tone={status.queue.queued > 0 ? 'accent' : undefined}
        />
        <MetricCard
          label="Processing"
          value={formatCount(status.queue.processing)}
          meta="Running now"
          tone={status.queue.processing > 0 ? 'accent' : undefined}
        />
        <MetricCard
          label="Failed jobs"
          value={formatCount(status.queue.failed)}
          meta="Ended without tracks"
          tone={status.queue.failed > 0 ? 'failed' : undefined}
        />
        <MetricCard
          label="Workers"
          value={formatCount(status.worker.active_workers)}
          meta={`${formatCount(status.worker.current_jobs)} in hand`}
        />
      </MetricRow>

      <Section title="Services">
        <div className={styles.services}>
          <ServiceRow
            name="Detection worker"
            ok={status.worker.available}
            message={status.worker.message}
            facts={[
              ['Mode', gpu ? 'VIAME GPU' : 'Mock'],
              ['Expected', status.worker.expected_mode],
              ['Attached', formatCount(status.worker.active_workers)],
              [
                'Other-mode workers',
                status.worker.other_mode_workers > 0
                  ? formatCount(status.worker.other_mode_workers)
                  : 'None',
              ],
              ['Last seen', formatClock(status.worker.last_seen_at)],
            ]}
          />
          <ServiceRow
            name="Database"
            ok={status.database.available}
            message={status.database.message}
            facts={[]}
          />
        </div>
      </Section>

      {status.ready ? null : (
        <Panel title="Bringing the stack back up">
          <p className="note">
            Nothing will process until a worker attaches. Start or restart the stack with:
          </p>
          <pre className={styles.command}>
            <code>{status.compose_command}</code>
          </pre>
        </Panel>
      )}

      {/* One setting does not need a card around it; a rule is enough. */}
      <Section title="Preferences">
        <div className={styles.preference}>
          <div>
            <p className={styles.preferenceName}>Colour theme</p>
            <p className="note">
              Dark suits long sessions in a dim room. Auto follows the operating system.
            </p>
          </div>
          <ThemeToggle />
        </div>
      </Section>
    </Page>
  );
}

function ServiceRow({
  name,
  ok,
  message,
  facts,
}: {
  name: string;
  ok: boolean;
  message: string;
  facts: [string, string][];
}) {
  return (
    <div className={styles.service} data-ok={ok || undefined}>
      <div className={styles.serviceHead}>
        <span className={styles.serviceName}>{name}</span>
        <StatusPill status={ok ? 'completed' : 'failed'}>{ok ? 'Healthy' : 'Down'}</StatusPill>
      </div>
      <p className="note">{message}</p>
      {facts.length > 0 ? (
        <dl className={styles.facts}>
          {facts.map(([term, value]) => (
            <div className={styles.fact} key={term}>
              <dt className={styles.factTerm}>{term}</dt>
              <dd className={styles.factValue}>{value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  );
}
