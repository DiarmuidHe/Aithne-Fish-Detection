import * as React from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import type {
  FishialDiagnostics,
  LiveFishialState,
  SpeciesReference,
  SpeciesSearchResult,
} from '@/api/types';
import { MIN_SPECIES_QUERY, fetchSpeciesSearch } from '@/api/species';
import { assignLiveTrackSpecies } from '@/api/live';
import { assignTrackSpecies, fetchTrackIdentification } from '@/api/tracks';
import { Button } from '@/components/base/button';
import { SearchInput } from '@/components/base/controls';
import { EmptyState, PanelError, SkeletonRows } from '@/components/base/feedback';
import { Modal } from '@/components/base/popups';
import { TabBar, TabPanel } from '@/components/base/tabs';
import { useToast } from '@/components/base/toast';
import { useDebouncedValue } from '@/hooks/use-debounced';
import { CloseIcon } from '@/icons';
import { formatPercent } from '@/lib/format';

import { useSpeciesReference } from './species-reference';
import { SpeciesReferenceViewer } from './species-viewer';
import type { FishKind } from './identify-fish';
import styles from './species-selector.module.css';

/*
 * Putting a name on a fish yourself.
 *
 * A classifier abstains, or answers something an operator can see is wrong, and
 * until now that was the end of the conversation: the dashboard could report the
 * machine's opinion and nothing else. This is the other half - the operator's own
 * answer, recorded in its own field, beside the two the machines produced rather
 * than on top of them.
 *
 * The interface is built around the fact that a species name is only useful if you
 * can check it. Every name on offer is shown with a photograph of the animal, the
 * photograph can be opened to full size with the rest of the set beside it, and
 * nothing is committed until the operator presses Confirm. The two ways in are the
 * two questions actually being asked: "is one of the machine's guesses right?" and
 * "then what is it?".
 */

/**
 * What a classifier already said about this fish, if anything.
 *
 * Structurally a `TrackIdentification`, but named separately and asking for less:
 * a clip tile and a track row carry the three headline Fishial fields without the
 * full audit, and they should be able to offer the same shortlist as the inspector
 * rather than sending the operator to search for a name they were just shown.
 */
export interface SpeciesSuggestions {
  state: LiveFishialState;
  species: string | null;
  confidence?: number | null;
  diagnostics?: FishialDiagnostics | null;
  implausible_for_region?: string | null;
}

/** What has been picked, from either tab: enough to draw it and to record it. */
export interface SpeciesChoice {
  species: string;
  common_name: string | null;
  image_url: string | null;
  attribution: string | null;
  licence: string | null;
  source_url: string | null;
}

function choiceFrom(
  species: string,
  reference: SpeciesReference | SpeciesSearchResult | null,
  commonName?: string | null,
): SpeciesChoice {
  return {
    species,
    common_name: commonName ?? reference?.common_name ?? null,
    image_url: reference?.image_url ?? null,
    attribution: reference?.attribution ?? null,
    licence: reference?.licence ?? null,
    source_url: reference?.source_url ?? null,
  };
}

/* --- Recording the decision ----------------------------------------------- */

/**
 * Assign or clear the name, and tell every screen that shows this fish.
 *
 * The same query keys the identification button invalidates, because the two
 * write to the same rows from the operator's point of view: whichever way a fish
 * got its name, every table, tile and panel showing that fish is now stale.
 */
export function useAssignSpecies(kind: FishKind, trackId: string) {
  const toast = useToast();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (species: string | null) => {
      if (kind === 'live') await assignLiveTrackSpecies(trackId, species);
      else await assignTrackSpecies(trackId, species);
      return species;
    },
    onSuccess: (species) => {
      toast.add(
        species
          ? {
              title: 'Species applied to this fish',
              description: `Recorded as ${species}. The classifier's own answer is unchanged.`,
              type: 'success',
            }
          : { title: 'Assigned name removed', type: 'success' },
      );
      void queryClient.invalidateQueries({ queryKey: ['track', trackId] });
      void queryClient.invalidateQueries({ queryKey: ['track-summaries'] });
      void queryClient.invalidateQueries({ queryKey: ['fish-clips'] });
      void queryClient.invalidateQueries({ queryKey: ['track-identification', trackId] });
      if (kind === 'live') {
        void queryClient.invalidateQueries({ queryKey: ['live-clips'] });
        void queryClient.invalidateQueries({ queryKey: ['live-latest'] });
      }
    },
    onError: (error) =>
      toast.add({
        title: 'The name was not recorded',
        description: error instanceof Error ? error.message : 'Request failed',
        type: 'error',
        timeout: 0,
      }),
  });
}

/* --- The trigger ---------------------------------------------------------- */

/**
 * The button an operator reaches for, wherever the fish is on screen.
 *
 * It carries its own dialog rather than taking one from the page, for the same
 * reason the identify control does: the question is asked from a tile, a table
 * row and an inspector, and none of those should have to own a modal to offer it.
 */
export function AssignSpeciesButton({
  kind,
  trackId,
  current,
  identification,
  size = 'small',
  variant = 'default',
  label,
}: {
  kind: FishKind;
  trackId: string;
  /** The name already assigned by hand, if any. */
  current?: string | null;
  identification?: SpeciesSuggestions | null;
  size?: 'default' | 'small';
  variant?: 'default' | 'primary' | 'quiet';
  label?: string;
}) {
  const [open, setOpen] = React.useState(false);
  return (
    <>
      <Button variant={variant} size={size} onClick={() => setOpen(true)}>
        {label ?? (current ? 'Change species' : 'Assign species')}
      </Button>
      <SpeciesSelector
        open={open}
        onOpenChange={setOpen}
        kind={kind}
        trackId={trackId}
        current={current ?? null}
        identification={identification ?? null}
      />
    </>
  );
}

/* --- The dialog ----------------------------------------------------------- */

export function SpeciesSelector({
  open,
  onOpenChange,
  kind,
  trackId,
  current,
  identification,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  kind: FishKind;
  trackId: string;
  current: string | null;
  identification: SpeciesSuggestions | null;
}) {
  // A table row and a clip tile carry the headline Fishial fields but not the
  // audit the candidates are derived from. Rather than offering those two places a
  // worse shortlist than the inspector's, the dialog fetches the audit itself -
  // once, only while open, and under the key the inspector already fills.
  const audit = useQuery({
    queryKey: ['track-identification', trackId],
    queryFn: () => fetchTrackIdentification(trackId),
    enabled:
      open &&
      kind === 'track' &&
      !identification?.diagnostics &&
      Boolean(identification && identification.state !== 'none'),
    staleTime: 30_000,
  });

  const suggestions = identification?.diagnostics ? identification : (audit.data ?? identification);
  const recommended = React.useMemo(() => recommendations(suggestions), [suggestions]);
  const [tab, setTab] = React.useState(recommended.length ? 'recommended' : 'search');
  const [choice, setChoice] = React.useState<SpeciesChoice | null>(null);
  const [viewing, setViewing] = React.useState<SpeciesChoice | null>(null);
  const assign = useAssignSpecies(kind, trackId);

  // Opening is the only moment the dialog's state is meaningful; a stale choice
  // from the previous fish must never be sitting there ready to be confirmed.
  React.useEffect(() => {
    if (!open) return;
    setChoice(null);
    setViewing(null);
    setTab(recommended.length ? 'recommended' : 'search');
  }, [open, recommended.length]);

  const confirm = () => {
    if (!choice) return;
    assign.mutate(choice.species, { onSuccess: () => onOpenChange(false) });
  };

  const clear = () => assign.mutate(null, { onSuccess: () => onOpenChange(false) });

  return (
    <>
      <Modal
        open={open}
        onOpenChange={onOpenChange}
        title="Assign species to fish track"
        description={
          current
            ? `Currently recorded as ${current}, assigned by hand.`
            : 'Your answer is recorded beside the classifier’s, not in place of it.'
        }
        footer={
          <>
            {current ? (
              <Button variant="quiet" onClick={clear} disabled={assign.isPending}>
                Remove assigned name
              </Button>
            ) : null}
            <Button variant="quiet" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={confirm}
              disabled={!choice || choice.species === current || assign.isPending}
            >
              {assign.isPending ? 'Recording…' : 'Confirm'}
            </Button>
          </>
        }
      >
        <div className={styles.body}>
          <TabBar
            label="How to find the species"
            value={tab}
            onValueChange={setTab}
            tabs={[
              { value: 'recommended', label: 'Recommended', count: recommended.length },
              { value: 'search', label: 'Search all species' },
            ]}
          >
            <TabPanel value="recommended">
              <RecommendedTab
                entries={recommended}
                choice={choice}
                onChoose={setChoice}
                onView={setViewing}
              />
            </TabPanel>
            <TabPanel value="search">
              <SearchTab choice={choice} onChoose={setChoice} onView={setViewing} />
            </TabPanel>
          </TabBar>

          {choice ? (
            <ChosenSpecies
              choice={choice}
              onClear={() => setChoice(null)}
              onView={() => setViewing(choice)}
            />
          ) : null}
        </div>
      </Modal>

      <SpeciesReferenceViewer
        species={viewing?.species ?? null}
        commonName={viewing?.common_name}
        onClose={() => setViewing(null)}
      />
    </>
  );
}

/* --- Tab A: what the classifier offered ----------------------------------- */

/** One name the classifier put forward, and why it is on this list. */
interface Recommendation {
  species: string;
  common_name: string | null;
  note: string | null;
  footnote: string | null;
}

/**
 * Every name this fish's identification actually produced, best first.
 *
 * An agreed answer, then the per-frame candidates that lost, then a name the
 * region list rejected. The last is included on purpose: the system already shows
 * that name rather than discarding it, and an operator who agrees with it must be
 * able to say so - marked, so agreeing is a decision rather than an accident.
 */
function recommendations(identification: SpeciesSuggestions | null): Recommendation[] {
  if (!identification) return [];
  const entries: Recommendation[] = [];
  const seen = new Set<string>();
  const add = (entry: Recommendation) => {
    const key = entry.species.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    entries.push(entry);
  };

  if (identification.state === 'identified' && identification.species) {
    add({
      species: identification.species,
      common_name: null,
      note: `${formatPercent(identification.confidence ?? 0, 0)} mean score · agreed answer`,
      footnote: null,
    });
  }
  for (const candidate of identification.diagnostics?.candidates ?? []) {
    add({
      species: candidate.species,
      common_name: candidate.common_name ?? null,
      note: `${formatPercent(candidate.max_score, 0)} best score${
        candidate.frames ? ` · ${candidate.frames} frames` : ''
      }`,
      footnote: candidate.rejected_for_region ? 'Not expected at this camera' : null,
    });
  }
  if (identification.implausible_for_region) {
    add({
      species: identification.implausible_for_region,
      common_name: null,
      note: null,
      footnote: 'Not expected at this camera',
    });
  }
  return entries;
}

function RecommendedTab({
  entries,
  choice,
  onChoose,
  onView,
}: {
  entries: Recommendation[];
  choice: SpeciesChoice | null;
  onChoose: (choice: SpeciesChoice) => void;
  onView: (choice: SpeciesChoice) => void;
}) {
  if (entries.length === 0) {
    return (
      <EmptyState
        title="No names have been offered for this fish"
        body="Identify it with Fishial to see what the classifier proposes, or use the search tab to name it yourself."
      />
    );
  }
  return (
    <>
      <p className={styles.intro}>
        What the classifier put forward for this fish. These are its own frame-by-frame
        answers, not a confirmed identification — the photographs are of the named
        species, never of this fish.
      </p>
      <ul className={styles.grid}>
        {entries.map((entry) => (
          <li key={entry.species}>
            <RecommendedCard entry={entry} choice={choice} onChoose={onChoose} onView={onView} />
          </li>
        ))}
      </ul>
    </>
  );
}

/**
 * A recommendation, with the portrait looked up per name.
 *
 * Per name rather than in one batch on purpose: these are the same names the track
 * table has already drawn, so the photo is nearly always in the cache and the card
 * costs no request at all.
 */
function RecommendedCard({
  entry,
  choice,
  onChoose,
  onView,
}: {
  entry: Recommendation;
  choice: SpeciesChoice | null;
  onChoose: (choice: SpeciesChoice) => void;
  onView: (choice: SpeciesChoice) => void;
}) {
  const reference = useSpeciesReference(entry.species);
  const resolved = choiceFrom(entry.species, reference, entry.common_name);
  return (
    <SpeciesCard
      choice={resolved}
      note={entry.note}
      footnote={entry.footnote}
      selected={choice?.species === entry.species}
      onChoose={() => onChoose(resolved)}
      onView={() => onView(resolved)}
    />
  );
}

/* --- Tab B: the whole catalogue ------------------------------------------- */

function SearchTab({
  choice,
  onChoose,
  onView,
}: {
  choice: SpeciesChoice | null;
  onChoose: (choice: SpeciesChoice) => void;
  onView: (choice: SpeciesChoice) => void;
}) {
  const [query, setQuery] = React.useState('');
  const [draft, setDraft] = useDebouncedValue(query, setQuery, 200);
  const ready = query.trim().length >= MIN_SPECIES_QUERY;

  const results = useQuery({
    queryKey: ['species-search', query.trim()],
    queryFn: () => fetchSpeciesSearch(query.trim()),
    enabled: ready,
    staleTime: 5 * 60_000,
  });

  const rows = results.data ?? [];

  return (
    <>
      <div className={styles.searchRow}>
        <SearchInput
          label="Search species"
          placeholder="Search species name (common or scientific)"
          value={draft}
          onChange={setDraft}
        />
        {draft ? (
          <Button
            variant="quiet"
            iconOnly
            aria-label="Clear search"
            onClick={() => {
              setDraft('');
              setQuery('');
            }}
          >
            <CloseIcon />
          </Button>
        ) : null}
      </div>

      {!ready ? (
        <p className={styles.intro}>
          Type at least {MIN_SPECIES_QUERY} letters of a common or scientific name — “cod”,
          “wrasse”, “Zeus”. The list covers the species recorded for these waters plus any
          name this deployment has already seen.
        </p>
      ) : results.isLoading ? (
        <SkeletonRows rows={4} height={72} lines={2} />
      ) : results.isError ? (
        <PanelError
          title="Could not search species"
          message={results.error instanceof Error ? results.error.message : 'Search failed'}
          onRetry={() => void results.refetch()}
        />
      ) : rows.length === 0 ? (
        <EmptyState
          title={`Nothing matches “${query.trim()}”`}
          body="Try a shorter word, the scientific name, or a name you have seen on another fish."
        />
      ) : (
        <div className={styles.results}>
          <ul className={styles.grid}>
            {rows.map((row) => {
              const resolved = choiceFrom(row.species, row);
              return (
                <li key={row.species}>
                  <SpeciesCard
                    choice={resolved}
                    note={null}
                    footnote={row.origin === 'observed' ? 'Seen on this deployment' : null}
                    selected={choice?.species === row.species}
                    onChoose={() => onChoose(resolved)}
                    onView={() => onView(resolved)}
                  />
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </>
  );
}

/* --- The pieces both tabs draw -------------------------------------------- */

function SpeciesCard({
  choice,
  note,
  footnote,
  selected,
  onChoose,
  onView,
}: {
  choice: SpeciesChoice;
  note: string | null;
  footnote: string | null;
  selected: boolean;
  onChoose: () => void;
  onView: () => void;
}) {
  return (
    // A toggle rather than a radio: the card is the control, and pressing the same
    // one twice is a mistake an operator should be able to make without the list
    // silently keeping the selection.
    <button
      type="button"
      className={styles.card}
      aria-pressed={selected}
      onClick={onChoose}
    >
      <span className={styles.cardZoom}>
        {choice.image_url ? (
          <img
            className={styles.cardImage}
            src={choice.image_url}
            alt={`Reference photo of ${choice.species} — not this fish`}
            loading="lazy"
            decoding="async"
          />
        ) : (
          <span className={styles.cardImageEmpty}>No reference photo</span>
        )}
        {choice.image_url ? (
          // Nested inside the card button would be invalid markup, so this is a
          // sibling laid over the photo; stopping propagation keeps enlarging a
          // photo from also choosing the species behind it.
          <span
            role="button"
            tabIndex={0}
            className={styles.cardZoomButton}
            aria-label={`See more photos of ${choice.species}`}
            onClick={(event) => {
              event.stopPropagation();
              onView();
            }}
            onKeyDown={(event) => {
              if (event.key !== 'Enter' && event.key !== ' ') return;
              event.preventDefault();
              event.stopPropagation();
              onView();
            }}
          >
            Enlarge
          </span>
        ) : null}
      </span>
      <span className={styles.cardName}>{choice.common_name ?? choice.species}</span>
      {choice.common_name ? (
        <span className={styles.cardScientific}>{choice.species}</span>
      ) : null}
      {note ? <span className={styles.cardNote}>{note}</span> : null}
      {footnote ? <span className={styles.cardFootnote}>{footnote}</span> : null}
    </button>
  );
}

function ChosenSpecies({
  choice,
  onClear,
  onView,
}: {
  choice: SpeciesChoice;
  onClear: () => void;
  onView: () => void;
}) {
  return (
    <section className={styles.choice} aria-label="Chosen species">
      {choice.image_url ? (
        <img
          className={styles.choiceImage}
          src={choice.image_url}
          alt={`Reference photo of ${choice.species} — not this fish`}
          onClick={onView}
          decoding="async"
        />
      ) : null}
      <div className={styles.choiceText}>
        <p className={styles.choiceLabel}>This fish will be recorded as</p>
        <p className={styles.choiceName}>{choice.common_name ?? choice.species}</p>
        {choice.common_name ? (
          <p className={styles.choiceScientific}>{choice.species}</p>
        ) : null}
        {choice.attribution ? (
          <p className={styles.choiceCredit}>
            Reference photo:{' '}
            {choice.source_url ? (
              <a href={choice.source_url} target="_blank" rel="noreferrer noopener">
                {choice.attribution}
              </a>
            ) : (
              choice.attribution
            )}
            {choice.licence ? ` · ${choice.licence.toUpperCase()}` : ''}
          </p>
        ) : null}
        <div className={styles.choiceActions}>
          {choice.image_url ? (
            <Button size="small" variant="quiet" onClick={onView}>
              See more photos
            </Button>
          ) : null}
          <Button size="small" variant="quiet" onClick={onClear}>
            Choose a different species
          </Button>
        </div>
      </div>
    </section>
  );
}
