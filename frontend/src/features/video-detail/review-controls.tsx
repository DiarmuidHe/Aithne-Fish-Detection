/**
 * The review decision control, and the optimistic apply-with-undo behind it.
 *
 * A decision is made hundreds of times in a sitting, so it is a five-way toggle
 * group rather than a `<select>`: one click, no menu, and the current decision
 * is legible without opening anything.
 */

import * as React from 'react';
import { useQueryClient } from '@tanstack/react-query';

import type { ReviewState, TrackSummary } from '@/api/types';
import { reviewTrack, reviewTracks } from '@/api/tracks';
import { Segmented } from '@/components/base/controls';
import { UNDO_TIMEOUT, useToast } from '@/components/base/toast';
import { plural } from '@/lib/format';

export const REVIEW_DECISIONS: { value: ReviewState; label: string; title: string }[] = [
  { value: 'unreviewed', label: 'Unrev', title: 'Unreviewed — keep the threshold decision' },
  { value: 'reviewed', label: 'Rev', title: 'Reviewed — looked at, threshold decision stands' },
  { value: 'accepted', label: 'Accept', title: 'Accepted — count this fish whatever the threshold says' },
  { value: 'rejected', label: 'Reject', title: 'Rejected — do not count this fish' },
  { value: 'needs-review', label: 'Flag', title: 'Needs review — excluded until someone decides' },
];

export function ReviewToggle({
  value,
  onChange,
  trackLabel,
  disabled,
}: {
  value: ReviewState;
  onChange: (next: ReviewState) => void;
  trackLabel: string;
  disabled?: boolean;
}) {
  return (
    <Segmented
      size="small"
      label={`Review decision for ${trackLabel}`}
      value={[value]}
      onChange={(next) => {
        // A toggle group can deselect; a review decision always has a value.
        const chosen = (next[0] as ReviewState) ?? 'unreviewed';
        if (chosen !== value) onChange(chosen);
      }}
      options={REVIEW_DECISIONS.map((decision) => ({
        value: decision.value,
        label: decision.label,
        title: decision.title,
        disabled,
      }))}
    />
  );
}

export interface ReviewApplier {
  applyOne: (track: TrackSummary, next: ReviewState) => void;
  applyMany: (tracks: TrackSummary[], next: ReviewState) => void;
}

/**
 * Applies a decision immediately and offers an undo for eight seconds. On
 * failure the change is rolled back and the error is named, so the screen never
 * shows a decision the server did not accept.
 */
export function useReviewApplier(invalidateKeys: unknown[][]): ReviewApplier {
  const queryClient = useQueryClient();
  const toast = useToast();

  const invalidate = React.useCallback(() => {
    for (const key of invalidateKeys) void queryClient.invalidateQueries({ queryKey: key });
  }, [queryClient, invalidateKeys]);

  const applyOne = React.useCallback(
    (track: TrackSummary, next: ReviewState) => {
      const previous = track.review_state;
      void (async () => {
        try {
          await reviewTrack(track.id, next);
          invalidate();
          toast.add({
            title: `Track #${track.viame_track_id} ${wordFor(next)}`,
            description: 'Regenerate the annotated video to include this decision.',
            timeout: UNDO_TIMEOUT,
            actionProps: {
              children: 'Undo',
              onClick: () => {
                void reviewTrack(track.id, previous).then(invalidate);
              },
            },
          });
        } catch (error) {
          invalidate();
          toast.add({
            title: 'Review was not saved',
            description: error instanceof Error ? error.message : 'Request failed',
            type: 'error',
            timeout: 0,
          });
        }
      })();
    },
    [invalidate, toast],
  );

  const applyMany = React.useCallback(
    (tracks: TrackSummary[], next: ReviewState) => {
      if (tracks.length === 0) return;
      const before = tracks.map((track) => ({ id: track.id, state: track.review_state }));
      void (async () => {
        try {
          await reviewTracks(
            tracks.map((track) => track.id),
            next,
          );
          invalidate();
          toast.add({
            title: `${plural(tracks.length, 'track')} ${wordFor(next)}`,
            description: 'Regenerate the annotated video to include these decisions.',
            timeout: UNDO_TIMEOUT,
            actionProps: {
              children: 'Undo',
              onClick: () => {
                // Restoring means grouping by the state each track had before.
                const groups = new Map<ReviewState, string[]>();
                for (const entry of before) {
                  groups.set(entry.state, [...(groups.get(entry.state) ?? []), entry.id]);
                }
                void Promise.all(
                  [...groups].map(([state, ids]) => reviewTracks(ids, state)),
                ).then(invalidate);
              },
            },
          });
        } catch (error) {
          invalidate();
          toast.add({
            title: 'Bulk review was not saved',
            description: error instanceof Error ? error.message : 'Request failed',
            type: 'error',
            timeout: 0,
          });
        }
      })();
    },
    [invalidate, toast],
  );

  return { applyOne, applyMany };
}

function wordFor(state: ReviewState): string {
  if (state === 'needs-review') return 'flagged';
  if (state === 'unreviewed') return 'returned to unreviewed';
  return state;
}
