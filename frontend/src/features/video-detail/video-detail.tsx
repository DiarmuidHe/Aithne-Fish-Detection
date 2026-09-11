import * as React from 'react';
import { useSearchParams } from 'react-router';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { annotateVideo, fetchVideo, generateFishClips, startProcessing } from '@/api/videos';
import { Button } from '@/components/base/button';
import { EmptyState, PanelError, Skeleton, StatusPill } from '@/components/base/feedback';
import { Confirm, Hint } from '@/components/base/popups';
import { TabBar, TabPanel } from '@/components/base/tabs';
import { useToast } from '@/components/base/toast';
import { CloseIcon } from '@/icons';
import { useConditionalPoll } from '@/hooks/use-poll';
import { formatBytes, formatDateTime, formatDuration } from '@/lib/format';

import { AnalyticsTab } from './analytics-tab';
import { AnnotatedTab } from './annotated-tab';
import { ClipsTab } from './clips-tab';
import { SummaryTab } from './summary-tab';
import { TracksTab } from './tracks-tab';
import styles from './detail.module.css';

const TABS = ['summary', 'tracks', 'analytics', 'annotated', 'clips'] as const;
type TabName = (typeof TABS)[number];

export function VideoDetail({
  videoId,
  onClose,
  filtersApplied,
}: {
  videoId?: string;
  onClose: () => void;
  filtersApplied: boolean;
}) {
  const [params, setParams] = useSearchParams();
  const queryClient = useQueryClient();
  const toast = useToast();
  const [confirmAnnotate, setConfirmAnnotate] = React.useState(false);

  const rawTab = params.get('tab');
  const tab: TabName = TABS.includes(rawTab as TabName) ? (rawTab as TabName) : 'summary';

  const video = useQuery({
    queryKey: ['video', videoId],
    queryFn: () => fetchVideo(videoId as string),
    enabled: Boolean(videoId),
  });

  const active =
    video.data !== undefined &&
    ['queued', 'processing'].includes(video.data.processing_status);
  const interval = useConditionalPoll(active);

  React.useEffect(() => {
    if (interval === false || !videoId) return;
    const timer = window.setInterval(() => {
      void queryClient.invalidateQueries({ queryKey: ['video', videoId] });
    }, interval);
    return () => window.clearInterval(timer);
  }, [interval, queryClient, videoId]);

  const process = useMutation({
    mutationFn: () => startProcessing(videoId as string),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['video', videoId] });
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
      toast.add({ title: 'Processing job queued', type: 'success' });
    },
    onError: (error) => toast.add({ title: 'Could not queue', description: text(error), type: 'error' }),
  });

  const annotate = useMutation({
    mutationFn: () => annotateVideo(videoId as string),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['video', videoId] });
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
      setParams((current) => {
        const next = new URLSearchParams(current);
        next.set('tab', 'annotated');
        return next;
      });
      toast.add({ title: 'Annotated video is ready', type: 'success' });
    },
    onError: (error) => toast.add({ title: 'Annotation failed', description: text(error), type: 'error' }),
  });

  const clips = useMutation({
    mutationFn: () => generateFishClips(videoId as string),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: ['fish-clips', videoId] });
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
      toast.add({
        title: created.length
          ? `${created.length} fish clips ready`
          : 'No accepted fish tracks to clip',
        type: 'success',
      });
    },
    onError: (error) => toast.add({ title: 'Clips failed', description: text(error), type: 'error' }),
  });

  if (!videoId) {
    return (
      <EmptyState
        title="Select a video"
        body={
          filtersApplied
            ? 'Choose a video from the filtered list to process it, review its tracks, or create an annotated copy.'
            : 'Choose a video from the library to process it, review its tracks, or create an annotated copy.'
        }
      />
    );
  }

  if (video.isLoading) {
    return (
      <div className={styles.panel}>
        <Skeleton width="45%" height={22} />
        <Skeleton width="70%" height={13} />
        <Skeleton height={140} />
      </div>
    );
  }

  if (video.isError || !video.data) {
    return (
      <div className={styles.panel}>
        <PanelError
          title="This video did not load"
          message={text(video.error)}
          onRetry={() => void video.refetch()}
        />
      </div>
    );
  }

  const data = video.data;
  const completed = data.processing_status === 'completed';
  const jobError =
    data.latest_job?.status === 'failed' ? data.latest_job.error_message : null;

  return (
    <div className={styles.detail}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <Hint content={data.original_filename}>
            <h2 className={styles.title}>{data.original_filename}</h2>
          </Hint>
          <StatusPill status={data.processing_status} />
          <Button variant="quiet" iconOnly aria-label="Close this video" onClick={onClose}>
            <CloseIcon />
          </Button>
        </div>

        <div className={styles.meta}>
          <span>{formatDateTime(data.created_at)}</span>
          <span>·</span>
          <span>{data.camera_id ?? 'No camera'}</span>
          <span>·</span>
          <span className={styles.metaValue}>{formatBytes(data.size_bytes)}</span>
          {data.duration_seconds ? (
            <>
              <span>·</span>
              <span className={styles.metaValue}>{formatDuration(data.duration_seconds)}</span>
            </>
          ) : null}
          {data.fps ? (
            <>
              <span>·</span>
              <span className={styles.metaValue}>{data.fps.toFixed(2)} fps</span>
            </>
          ) : null}
        </div>

        {active ? (
          <p className={styles.callout} aria-live="polite">
            {data.processing_status === 'queued'
              ? 'Waiting for a worker. This updates automatically; you can leave the page open.'
              : 'VIAME processing is in progress. This updates automatically.'}
          </p>
        ) : null}

        {jobError ? (
          <p className={`${styles.callout} ${styles.calloutError}`} role="alert">
            Processing did not complete. {jobError}
          </p>
        ) : null}

        <div className={styles.actions}>
          <Button
            variant="primary"
            size="small"
            disabled={active || process.isPending}
            onClick={() => process.mutate()}
          >
            {processLabel(data.processing_status)}
          </Button>
          <Button
            size="small"
            disabled={!completed || annotate.isPending}
            onClick={() => (data.annotated_at ? setConfirmAnnotate(true) : annotate.mutate())}
          >
            {annotate.isPending
              ? 'Generating…'
              : data.annotated_at
                ? 'Regenerate annotated video'
                : 'Generate annotated video'}
          </Button>
          <Button
            size="small"
            disabled={!completed || clips.isPending}
            onClick={() => clips.mutate()}
          >
            {clips.isPending ? 'Cutting clips…' : 'Generate fish clips'}
          </Button>
        </div>
      </div>

      <TabBar
        label="Video detail"
        value={tab}
        onValueChange={(next) =>
          setParams((current) => {
            const params = new URLSearchParams(current);
            params.set('tab', next);
            return params;
          })
        }
        tabs={[
          { value: 'summary', label: 'Summary' },
          { value: 'tracks', label: 'Tracks', disabled: !completed },
          { value: 'analytics', label: 'Analytics', disabled: !completed },
          { value: 'annotated', label: 'Annotated', disabled: !completed },
          { value: 'clips', label: 'Clips', disabled: !completed },
        ]}
      >
        <TabPanel value="summary">
          <SummaryTab video={data} />
        </TabPanel>
        <TabPanel value="tracks">
          {completed ? <TracksTab videoId={videoId} fps={data.fps} /> : null}
        </TabPanel>
        <TabPanel value="analytics">
          {completed ? <AnalyticsTab videoId={videoId} /> : null}
        </TabPanel>
        <TabPanel value="annotated">
          {completed ? <AnnotatedTab video={data} /> : null}
        </TabPanel>
        <TabPanel value="clips">
          {completed ? <ClipsTab videoId={videoId} /> : null}
        </TabPanel>
      </TabBar>

      <Confirm
        open={confirmAnnotate}
        onOpenChange={setConfirmAnnotate}
        title="Replace the annotated video?"
        description="The existing annotated copy is deleted and rebuilt from the current review decisions. The stored detections do not change."
        confirmLabel="Regenerate"
        onConfirm={() => {
          setConfirmAnnotate(false);
          annotate.mutate();
        }}
      />
    </div>
  );
}

function processLabel(status: string): string {
  if (status === 'queued') return 'Queued';
  if (status === 'processing') return 'Processing…';
  if (status === 'failed') return 'Retry processing';
  if (status === 'completed') return 'Process again';
  return 'Start processing';
}

function text(error: unknown): string {
  return error instanceof Error ? error.message : 'Request failed';
}
