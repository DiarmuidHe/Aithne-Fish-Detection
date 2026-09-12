import { request, toQuery } from './client';
import type { SpeciesGallery, SpeciesReference, SpeciesSearchResult } from './types';

/** Reference photos for a set of species names. Names are looked up once, server-side. */
export function fetchSpeciesReferences(names: string[]): Promise<SpeciesReference[]> {
  return request<SpeciesReference[]>(`/species/reference${toQuery({ name: names })}`);
}

/** Below this the answer is most of the catalogue; the server holds the same floor. */
export const MIN_SPECIES_QUERY = 3;

/**
 * Species an operator may assign, best match first.
 *
 * Searched server-side against the curated list for these waters plus the names
 * this deployment has already recorded, so a classifier's answer stays choosable
 * even when no curated list holds it.
 */
export function fetchSpeciesSearch(query: string, limit = 24): Promise<SpeciesSearchResult[]> {
  return request<SpeciesSearchResult[]>(`/species/search${toQuery({ q: query, limit })}`);
}

/**
 * Every reference photo held for one species.
 *
 * Asked for only when the viewer is opened: this is the one call that may buy
 * several photos, and a table full of thumbnails must never trigger it.
 */
export function fetchSpeciesGallery(name: string): Promise<SpeciesGallery> {
  return request<SpeciesGallery>(`/species/gallery${toQuery({ name })}`);
}
