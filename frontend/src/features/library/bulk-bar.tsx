import * as React from 'react';

import { EXPORT_KINDS, batchExportUrl, runBatch, type ExportKind } from '@/api/videos';
import { Button } from '@/components/base/button';
import { ProgressBar, Separator, Toolbar } from '@/components/base/feedback';
import { ActionMenu, Confirm } from '@/components/base/popups';
import { useToast } from '@/components/base/toast';
import { formatCount, plural } from '@/lib/format';
import type { Selection } from '@/hooks/use-selection';

import styles from './library.module.css';

/**
 * The bulk bar only exists when something is selected. Four disabled buttons
 * parked above the list tell an operator nothing.
 */
export function BulkBar({
  selection,
  onDone,
}: {
  selection: Selection;
  onDone: () => void;
}) {
  const toast = useToast();
  const [busy, setBusy] = React.useState<null | { action: string; done: number; total: number }>(
    null,
  );
  const [confirmAnnotate, setConfirmAnnotate] = React.useState(false);

  if (selection.count === 0) return null;

  async function run(action: 'process' | 'annotate') {
    const ids = selection.ids;
    setBusy({ action, done: 0, total: ids.length });
    const results: { ok: boolean; error?: string }[] = [];
    // One request per chunk keeps the progress bar honest without 200 round trips.
    const CHUNK = 20;
    for (let index = 0; index < ids.length; index += CHUNK) {
      const chunk = ids.slice(index, index + CHUNK);
      try {
        const response = await runBatch(action, chunk);
        results.push(...response.results);
      } catch (error) {
        results.push(
          ...chunk.map(() => ({
            ok: false,
            error: error instanceof Error ? error.message : 'Request failed',
          })),
        );
      }
      setBusy({ action, done: Math.min(ids.length, index + CHUNK), total: ids.length });
    }
    setBusy(null);

    const failed = results.filter((result) => !result.ok);
    const succeeded = results.length - failed.length;
    toast.add({
      title:
        action === 'process'
          ? `${plural(succeeded, 'video')} queued`
          : `${plural(succeeded, 'annotation')} generated`,
      description:
        failed.length > 0
          ? `${plural(failed.length, 'video')} failed: ${failed[0].error ?? 'see server logs'}`
          : undefined,
      type: failed.length > 0 ? 'error' : 'success',
      timeout: failed.length > 0 ? 0 : 5_000,
    });
    onDone();
  }

  function exportSelected(kind: ExportKind) {
    // A CSV is a navigation, not a fetch: the browser handles the download.
    window.location.assign(batchExportUrl(kind, selection.ids));
  }

  return (
    <Toolbar.Root className={styles.bulkBar} aria-label="Actions for selected videos">
      <span className={styles.bulkCount}>
        {formatCount(selection.count)} selected
        {selection.wholeResultSelected ? (
          <span className={styles.bulkHidden}> — every video matching these filters</span>
        ) : null}
        {selection.hiddenCount > 0 ? (
          <span className={styles.bulkHidden}> ({selection.hiddenCount} hidden by filters)</span>
        ) : null}
      </span>

      <Separator orientation="vertical" />

      <Toolbar.Button
        render={<Button variant="primary" size="small" disabled={busy !== null} />}
        onClick={() => void run('process')}
      >
        Process
      </Toolbar.Button>
      <Toolbar.Button
        render={<Button size="small" disabled={busy !== null} />}
        onClick={() => setConfirmAnnotate(true)}
      >
        Annotate
      </Toolbar.Button>

      <ActionMenu
        trigger={<Button size="small">Export</Button>}
        items={EXPORT_KINDS.map(({ kind, label }) => ({
          label,
          onSelect: () => exportSelected(kind),
        }))}
      />

      <span className={styles.bulkSpacer} />

      <Toolbar.Button render={<Button variant="quiet" size="small" />} onClick={selection.clear}>
        Clear
      </Toolbar.Button>

      {busy ? (
        <div className={styles.bulkProgress}>
          <ProgressBar
            label={busy.action === 'process' ? 'Queueing' : 'Annotating'}
            value={busy.done}
            max={busy.total}
            valueLabel={`${busy.done} of ${busy.total}`}
          />
        </div>
      ) : null}

      <Confirm
        open={confirmAnnotate}
        onOpenChange={setConfirmAnnotate}
        title={`Generate annotations for ${plural(selection.count, 'video')}?`}
        description="Any existing annotated copy is replaced. Boxes are drawn only on frames containing accepted database detections, so current review decisions are baked in."
        confirmLabel="Generate"
        destructive={false}
        onConfirm={() => {
          setConfirmAnnotate(false);
          void run('annotate');
        }}
      />
    </Toolbar.Root>
  );
}
