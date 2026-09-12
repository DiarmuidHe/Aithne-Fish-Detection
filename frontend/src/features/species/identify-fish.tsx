import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import type { LiveFishialState, TrackIdentification } from '@/api/types';
import { fetchSystemStatus } from '@/api/system';
import { identifyLiveTrack } from '@/api/live';
import { identifyTrack } from '@/api/tracks';
import { Button } from '@/components/base/button';
import { NumberInput } from '@/components/base/controls';
import { PopoverPanel } from '@/components/base/popups';
import { useToast } from '@/components/base/toast';
import { formatDateTime, formatPercent } from '@/lib/format';

import { AssignSpeciesButton } from './species-selector';
import { SpeciesNameplate, SpeciesReferenceCard, SpeciesThumb } from './species-reference';
import styles from './identify.module.css';

/*
 * Naming one fish, from wherever that fish is on screen.
 *
 * The same control appears on a live tile, on a clip tile, in a track row and in
 * the track inspector, because "which fish is that?" is asked in all four places
 * and an operator should not have to learn where the question lives. What the
 * operator chooses is the fish and how many frames to pay for; which frames those
 * are is the system's job, and the panel says so rather than implying a choice
 * that is not being offered.
 */

const FRAMES_STORAGE = 'species-identify-frames';

export type FishKind = 'track' | 'live';

// The automatic live pass has already claimed a fish in these states, so a
// request would be refused; the control says "identifying" rather than offering.
const RUNNING: LiveFishialState[] = ['submitted', 'pending', 'ready'];
const ANSWERED: LiveFishialState[] = ['identified', 'review_required', 'error'];

export interface IdentificationLike {
  state: LiveFishialState;
  species: string | null;
  confidence: number | null;
}

/** The deployment's answer to "may I ask, and for how many frames?". */
export function useIdentificationLimits() {
  const status = useQuery({ queryKey: ['system-status'], queryFn: fetchSystemStatus });
  return status.data?.species_identification ?? null;
}

function useStoredFrames(fallback: number): [number | null, (value: number | null) => void] {
  const [frames, setFrames] = React.useState<number | null>(() => {
    try {
      return Number(window.localStorage.getItem(FRAMES_STORAGE)) || null;
    } catch {
      // A browser can refuse storage entirely; the default still works.
      return null;
    }
  });
  const update = (value: number | null) => {
    setFrames(value);
    try {
      if (value) window.localStorage.setItem(FRAMES_STORAGE, String(value));
    } catch {
      // Remembering the last count is a convenience, not a requirement.
    }
  };
  return [frames ?? fallback, update];
}

export function IdentifyFishButton({
  kind,
  trackId,
  state,
  label = 'Identify species',
  size = 'small',
  variant = 'default',
}: {
  kind: FishKind;
  trackId: string;
  state: LiveFishialState;
  label?: string;
  size?: 'default' | 'small';
  variant?: 'default' | 'primary' | 'quiet';
}) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const limits = useIdentificationLimits();
  const [frames, setFrames] = useStoredFrames(limits?.default_frames ?? 5);

  const send = useMutation({
    // Both endpoints answer with the claim; the shapes differ only in what else
    // they carry, and this control reads neither - it invalidates instead.
    mutationFn: async () => {
      if (kind === 'live') await identifyLiveTrack(trackId, frames as number);
      else await identifyTrack(trackId, frames as number);
    },
    onSuccess: () => {
      toast.add({
        title: 'Sent to Fishial',
        description: `Choosing the clearest ${frames} ${frames === 1 ? 'frame' : 'frames'} of this fish.`,
        type: 'success',
      });
      void queryClient.invalidateQueries({ queryKey: ['track-identification', trackId] });
      void queryClient.invalidateQueries({ queryKey: ['track-summaries'] });
      void queryClient.invalidateQueries({ queryKey: ['fish-clips'] });
      void queryClient.invalidateQueries({ queryKey: ['track', trackId] });
      if (kind === 'live') {
        void queryClient.invalidateQueries({ queryKey: ['live-clips'] });
        void queryClient.invalidateQueries({ queryKey: ['live-latest'] });
      }
    },
    onError: (error) =>
      toast.add({
        title: 'Could not identify this fish',
        description: error instanceof Error ? error.message : 'Request failed',
        type: 'error',
        timeout: 0,
      }),
  });

  if (!limits?.available) return null;

  const running = RUNNING.includes(state) || send.isPending;
  const again = ANSWERED.includes(state);

  return (
    <PopoverPanel
      align="end"
      title={again ? 'Identify this fish again' : 'Identify this fish'}
      trigger={
        <Button variant={variant} size={size} disabled={running}>
          {running ? 'Identifying…' : again ? 'Identify again' : label}
        </Button>
      }
    >
      <div className={styles.form}>
        <NumberInput
          label="Frames to send"
          min={1}
          max={limits.max_frames}
          value={frames}
          onChange={setFrames}
          description={`1–${limits.max_frames}. Each frame is one Fishial API call.`}
        />
        <p className={styles.note}>
          The system picks which frames: the sharpest, largest, best-lit views of this
          fish, one per separate moment it was in view. A fish seen only briefly is sent
          fewer frames than you ask for rather than the same instant twice.
        </p>
        {again ? (
          <p className={styles.note}>
            This replaces the current result and pays for new frames.
          </p>
        ) : null}
        <Button variant="primary" onClick={() => send.mutate()} disabled={send.isPending}>
          {send.isPending ? 'Sending…' : `Send ${frames} ${frames === 1 ? 'frame' : 'frames'}`}
        </Button>
      </div>
    </PopoverPanel>
  );
}

/**
 * The one-line answer, for a table cell or under a tile.
 *
 * A name a person assigned wins the line outright. Three claims cannot fit in a
 * table cell, and of the three the human one is the only one a reader would be
 * wrong to overrule at a glance; the rest stay a click away in the inspector.
 */
export function IdentificationBadge({
  identification,
  manualSpecies,
  placeholder = 'Not identified',
}: {
  identification: IdentificationLike;
  manualSpecies?: string | null;
  placeholder?: string;
}) {
  const { state, species, confidence } = identification;
  if (manualSpecies) {
    return (
      <span className={styles.badge} data-state="manual">
        <SpeciesNameplate name={manualSpecies}>
          <span className={styles.badgeScore}>assigned</span>
        </SpeciesNameplate>
      </span>
    );
  }
  if (state === 'identified' && species) {
    return (
      <span className={styles.badge} data-state="identified">
        <SpeciesNameplate name={species}>
          <span className={styles.badgeScore}>{formatPercent(confidence ?? 0, 0)}</span>
        </SpeciesNameplate>
      </span>
    );
  }
  if (RUNNING.includes(state)) {
    return (
      <span className={styles.badge} data-state="submitted">
        Identifying…
      </span>
    );
  }
  if (state === 'review_required' || state === 'error') {
    return (
      <span className={styles.badge} data-state="review">
        Needs review
      </span>
    );
  }
  return <span className={styles.badgeEmpty}>{placeholder}</span>;
}

/**
 * The full account: what was asked for, what was sent, what came back, and — when
 * no name was agreed — which names were in play and why none of them won. An
 * abstention that cannot explain itself is indistinguishable from a broken
 * integration, so every reason is printed rather than summarised as "failed".
 */
export function IdentificationPanel({
  kind,
  trackId,
  identification,
  manualSpecies = null,
  manualSpeciesAt = null,
}: {
  kind: FishKind;
  trackId: string;
  identification: TrackIdentification;
  /** What a person decided, shown above the machine's account rather than instead. */
  manualSpecies?: string | null;
  manualSpeciesAt?: string | null;
}) {
  const limits = useIdentificationLimits();
  const { state, diagnostics } = identification;
  const candidates = diagnostics?.candidates ?? [];

  const assigned = manualSpecies ? (
    <p className={styles.assigned}>
      <SpeciesNameplate name={manualSpecies} size="row">
        <span className={styles.assignedWhen}>
          assigned by hand{manualSpeciesAt ? ` · ${formatDateTime(manualSpeciesAt)}` : ''}
        </span>
      </SpeciesNameplate>
    </p>
  ) : null;

  // A deployment with no classifier still has operators who can name a fish, so
  // the manual control is offered here whether or not Fishial is configured.
  if (!limits?.available) {
    return (
      <section className={styles.panel} aria-label="Species identification">
        {assigned}
        <div className={styles.panelHead}>
          <p className={styles.note}>
            Automatic identification is not configured on this deployment. Set{' '}
            <code>FISHIAL_ENABLED</code> with a client ID and secret to have fish named
            from the dashboard, or name this one yourself.
          </p>
          <AssignSpeciesButton
            kind={kind}
            trackId={trackId}
            current={manualSpecies}
            identification={identification}
            variant="primary"
          />
        </div>
      </section>
    );
  }

  return (
    <section className={styles.panel} aria-label="Species identification">
      {assigned}
      <div className={styles.panelHead}>
        <div>
          <p className="label">Fishial identification</p>
          <p className={styles.headline}>
            {state === 'identified' && identification.species ? (
              <>
                <span className={styles.species}>{identification.species}</span>
                <span className={styles.headlineNote}>
                  {formatPercent(identification.confidence ?? 0, 0)} mean score across{' '}
                  {identification.tally[identification.species] ?? 0} agreeing frames
                </span>
                <SpeciesReferenceCard name={identification.species} />
              </>
            ) : state === 'submitted' ? (
              'Choosing frames and asking Fishial…'
            ) : state === 'none' ? (
              'This fish has not been sent for identification.'
            ) : (
              (diagnostics?.reason ?? 'No species was agreed')
            )}
          </p>
        </div>
        {/* Ask the classifier, or decide it yourself. Both are offered here so a
            disagreement with the machine is settled where the machine's account
            is, rather than somewhere else on the page. */}
        <div className={styles.panelActions}>
          <IdentifyFishButton kind={kind} trackId={trackId} state={state} variant="primary" />
          <AssignSpeciesButton
            kind={kind}
            trackId={trackId}
            current={manualSpecies}
            identification={identification}
          />
        </div>
      </div>

      {state !== 'none' ? (
        <dl className={styles.figures}>
          <div>
            <dt>Frames requested</dt>
            <dd>{identification.frames_requested ?? '—'}</dd>
          </div>
          <div>
            <dt>Frames chosen</dt>
            <dd>{identification.frames_selected}</dd>
          </div>
          <div>
            <dt>Frames sent</dt>
            <dd>{identification.frames_submitted}</dd>
          </div>
          <div>
            <dt>Separate moments</dt>
            <dd>{identification.windows}</dd>
          </div>
          <div>
            <dt>Calls saved</dt>
            <dd>{identification.calls_saved}</dd>
          </div>
          <div>
            <dt>Crop quality</dt>
            <dd>
              {identification.quality_score === null
                ? '—'
                : formatPercent(identification.quality_score, 0)}
            </dd>
          </div>
        </dl>
      ) : null}

      {identification.implausible_for_region ? (
        <p className={styles.caution}>
          <strong>Not expected at this camera.</strong>{' '}
          <SpeciesNameplate name={identification.implausible_for_region} /> is not on
          this camera's regional species list. The name is shown rather than discarded so you can
          judge it — treat it as a suggestion, not a record.
        </p>
      ) : null}

      {identification.windows > 0 && identification.windows < identification.frames_submitted ? (
        <p className={styles.note}>
          {identification.frames_submitted} frames came from {identification.windows} separate{' '}
          {identification.windows === 1 ? 'moment' : 'moments'}, so some of them show the same
          instant. Frames of one moment repeat a pose rather than testing it, which makes their
          agreement weaker evidence than the count suggests.
        </p>
      ) : null}

      {identification.stopped_early ? (
        <p className={styles.note}>
          Stopped early: {identification.stopped_early}. The remaining frames could not have
          changed the answer, so they were not bought.
        </p>
      ) : null}

      {candidates.length > 0 ? (
        <div className={styles.candidates}>
          <p className="label">Names the classifier offered</p>
          <ul className={styles.candidateList}>
            {candidates.map((candidate) => (
              <li key={`${candidate.species}-${candidate.rejected_for_region}`}>
                <span className={styles.candidateName}>
                  <SpeciesThumb name={candidate.species} />
                  <span>
                    {candidate.common_name
                      ? `${candidate.common_name} (${candidate.species})`
                      : candidate.species}
                  </span>
                </span>
                <span className={styles.candidateScore}>
                  {formatPercent(candidate.max_score, 0)} best score
                  {candidate.frames ? ` · ${candidate.frames} frames` : ''}
                  {candidate.rejected_for_region ? ' · not plausible for this region' : ''}
                </span>
              </li>
            ))}
          </ul>
          <p className={styles.note}>
            These are individual frame answers, not a confirmed identification.
          </p>
        </div>
      ) : null}

      {diagnostics?.frame_reasons && Object.keys(diagnostics.frame_reasons).length > 0 ? (
        <p className={styles.note}>
          {Object.entries(diagnostics.frame_reasons)
            .map(([reason, count]) => `${count} × ${reason}`)
            .join(' · ')}
        </p>
      ) : null}
    </section>
  );
}
