import { useQuery } from '@tanstack/react-query';

import { fetchVideoAnalytics } from '@/api/videos';
import { ChartSkeleton, ColumnChart } from '@/components/charts/bar-chart';
import { PanelError } from '@/components/base/feedback';
import { formatCount } from '@/lib/format';

import styles from './detail.module.css';

export function AnalyticsTab({ videoId }: { videoId: string }) {
  const analytics = useQuery({
    queryKey: ['video-analytics', videoId],
    queryFn: () => fetchVideoAnalytics(videoId),
  });

  if (analytics.isLoading) {
    return (
      <div className={styles.panel}>
        <ChartSkeleton title="Detections over time" />
        <ChartSkeleton title="Confidence distribution" />
        <ChartSkeleton title="Track duration" />
      </div>
    );
  }

  if (analytics.isError || !analytics.data) {
    return (
      <div className={styles.panel}>
        <PanelError
          message={analytics.error instanceof Error ? analytics.error.message : 'Analytics failed'}
          onRetry={() => void analytics.refetch()}
        />
      </div>
    );
  }

  const data = analytics.data;
  const binWidth = data.time_bins[0] ? data.time_bins[0].end - data.time_bins[0].start : 1;
  const precision = binWidth < 0.1 ? 3 : 1;

  return (
    <div className={styles.panel}>
      <ColumnChart
        title="Detections over time"
        note="Accepted in colour, all observations behind. Empty windows contain no detections."
        valueLabel="accepted detections"
        columns={data.time_bins.map((bin) => ({
          label: `${bin.start.toFixed(precision)}–${bin.end.toFixed(precision)} s`,
          value: bin.accepted_detections,
          background: bin.all_detections,
        }))}
      />

      <ColumnChart
        title="Confidence distribution"
        note="All observations, including rejected tracks"
        valueLabel="observations"
        columns={data.confidence_distribution.map((bin) => ({
          label: `${Math.round(bin.start * 100)}–${Math.round(bin.end * 100)}%`,
          value: bin.count,
        }))}
      />

      <ColumnChart
        title="Track duration"
        note="All tracks · last minus first observation"
        valueLabel="tracks"
        columns={data.track_duration_distribution.map((bin) => ({
          label: `${bin.start.toFixed(2)}–${bin.end.toFixed(2)} s`,
          value: bin.count,
        }))}
      />

      <p className="note">
        {formatCount(data.unknown_timestamp_detections)} observations without timestamps ·{' '}
        {formatCount(data.unknown_duration_tracks)} tracks without duration ·{' '}
        {formatCount(data.accepted_fish_count)} accepted of{' '}
        {formatCount(data.all_track_count)} tracks. The final bin includes its upper boundary.
      </p>
    </div>
  );
}
