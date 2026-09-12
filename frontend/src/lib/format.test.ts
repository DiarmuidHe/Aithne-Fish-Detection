import { describe, expect, it } from 'vitest';

import { basename, formatBytes, formatPercent, formatSeconds, plural } from './format';

describe('counting things', () => {
  it('leaves nouns that are their own plural alone', () => {
    expect(plural(1, 'fish')).toBe('1 fish');
    expect(plural(3, 'fish')).toBe('3 fish');
    expect(plural(0, 'fish')).toBe('0 fish');
    expect(plural(2, 'species')).toBe('2 species');
  });

  it('still pluralises a phrase that only ends in an ordinary noun', () => {
    expect(plural(2, 'fish track')).toBe('2 fish tracks');
    expect(plural(1, 'fish track')).toBe('1 fish track');
    expect(plural(2, 'video')).toBe('2 videos');
    expect(plural(2, 'detection')).toBe('2 detections');
  });

  it('accepts an explicit plural for anything irregular', () => {
    expect(plural(2, 'analysis', 'analyses')).toBe('2 analyses');
  });
});

describe('formatting the numbers an operator reads', () => {
  it('writes a position in a video as m:ss.s', () => {
    expect(formatSeconds(0)).toBe('0:00.0');
    expect(formatSeconds(61.25)).toBe('1:01.3');
    expect(formatSeconds(125.4)).toBe('2:05.4');
    expect(formatSeconds(null)).toBe('Not detected');
  });

  it('names a missing size rather than showing a zero', () => {
    expect(formatBytes(null)).toBe('—');
    expect(formatBytes(2048)).toBe('2.0 KB');
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB');
  });

  it('shows confidences as percentages', () => {
    expect(formatPercent(0.945)).toBe('94.5%');
    expect(formatPercent(null)).toBe('—');
  });

  it('reduces a pipeline path to its leaf, on either separator', () => {
    expect(basename('/opt/noaa/viame/pipelines/fish.pipe')).toBe('fish.pipe');
    expect(basename('C:\\viame\\pipelines\\fish.pipe')).toBe('fish.pipe');
    expect(basename(null)).toBe('—');
  });
});
