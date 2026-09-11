/**
 * One 16px single-stroke icon set, drawn on a shared 16×16 grid.
 *
 * No icon font, no emoji, no second family. Icons appear on status, actions and
 * disclosure only — not beside every label.
 */

import * as React from 'react';

type IconProps = React.SVGProps<SVGSVGElement> & { title?: string };

function Icon({ title, children, ...props }: React.PropsWithChildren<IconProps>) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={title ? undefined : true}
      role={title ? 'img' : undefined}
      focusable="false"
      {...props}
    >
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  );
}

export const SearchIcon = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="7" cy="7" r="4.25" />
    <path d="M10.2 10.2 13.5 13.5" />
  </Icon>
);

export const ChevronDownIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 6.25 8 10.25 12 6.25" />
  </Icon>
);

export const ChevronRightIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M6.25 3.5 10.25 7.5 6.25 11.5" />
  </Icon>
);

export const ChevronUpDownIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M5 6.5 8 3.5l3 3" />
    <path d="M5 9.5 8 12.5l3-3" />
  </Icon>
);

export const CheckIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 8.5 6.25 11.75 13 5" />
  </Icon>
);

export const CloseIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 4 12 12M12 4 4 12" />
  </Icon>
);

export const PlusIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 3.5v9M3.5 8h9" />
  </Icon>
);

export const FilterIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 4h11M4.5 8h7M6.5 12h3" />
  </Icon>
);

export const SortAscIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 12.5V3.5M1.75 5.75 4 3.5l2.25 2.25" />
    <path d="M8.5 5h6M8.5 8.5h4M8.5 12h2" />
  </Icon>
);

export const SortDescIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 3.5v9M1.75 10.25 4 12.5l2.25-2.25" />
    <path d="M8.5 5h2M8.5 8.5h4M8.5 12h6" />
  </Icon>
);

export const PlayIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M5 3.5 12.5 8 5 12.5Z" />
  </Icon>
);

export const StopIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="4" y="4" width="8" height="8" rx="1" />
  </Icon>
);

export const DownloadIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 2.5v7.5M5 7.5 8 10.5l3-3" />
    <path d="M3 12.5h10" />
  </Icon>
);

export const UploadIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 12.5V5M5 8 8 5l3 3" />
    <path d="M3 2.5h10" />
  </Icon>
);

export const RefreshIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M13 8a5 5 0 1 1-1.6-3.65" />
    <path d="M13.5 2v3h-3" />
  </Icon>
);

export const MoreIcon = (props: IconProps) => (
  <Icon {...props} strokeWidth="2">
    <path d="M3.5 8h.01M8 8h.01M12.5 8h.01" />
  </Icon>
);

export const FlagIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 14V2.5" />
    <path d="M4 3h7.5l-1.5 2.5L11.5 8H4Z" />
  </Icon>
);

export const AlertIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M8 2.5 14.5 13.5h-13Z" />
    <path d="M8 6.5v3M8 11.5h.01" />
  </Icon>
);

export const ClockIcon = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="M8 4.75V8l2.25 1.5" />
  </Icon>
);

export const FilmIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2" y="3.5" width="12" height="9" rx="1" />
    <path d="M5 3.5v9M11 3.5v9M2 8h12" />
  </Icon>
);

export const GridIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2.5" y="2.5" width="4.5" height="4.5" rx="0.75" />
    <rect x="9" y="2.5" width="4.5" height="4.5" rx="0.75" />
    <rect x="2.5" y="9" width="4.5" height="4.5" rx="0.75" />
    <rect x="9" y="9" width="4.5" height="4.5" rx="0.75" />
  </Icon>
);

export const ChartIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 13.5h11" />
    <path d="M4.5 13.5V9M8 13.5V4M11.5 13.5V7" />
  </Icon>
);

export const LibraryIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 3.5h11M2.5 8h11M2.5 12.5h11" />
  </Icon>
);

export const ReviewIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 3.5h10v9H3z" />
    <path d="M5.5 7.5 7 9l3.5-3.5" />
  </Icon>
);

export const BroadcastIcon = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="1.75" />
    <path d="M4.6 4.6a4.8 4.8 0 0 0 0 6.8M11.4 4.6a4.8 4.8 0 0 1 0 6.8" />
  </Icon>
);

export const GaugeIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M2.5 11.5a5.5 5.5 0 1 1 11 0" />
    <path d="M8 11.5 10.5 6.5" />
  </Icon>
);

export const SunIcon = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="8" cy="8" r="3" />
    <path d="M8 1.5v1.5M8 13v1.5M14.5 8H13M3 8H1.5M12.6 3.4l-1 1M4.4 11.6l-1 1M12.6 12.6l-1-1M4.4 4.4l-1-1" />
  </Icon>
);

export const MoonIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5Z" />
  </Icon>
);

export const ServerIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2" y="3" width="12" height="4.5" rx="1" />
    <rect x="2" y="8.5" width="12" height="4.5" rx="1" />
    <path d="M4.5 5.25h.01M4.5 10.75h.01" />
  </Icon>
);

export const CollapseIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="2" y="3" width="12" height="10" rx="1" />
    <path d="M6.5 3v10" />
  </Icon>
);

export const CameraIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M1.5 5.5h8v5.5h-8z" />
    <path d="M9.5 8.25 14 5.75v4.5L9.5 7.75Z" />
  </Icon>
);

export const KeyboardIcon = (props: IconProps) => (
  <Icon {...props}>
    <rect x="1.5" y="4" width="13" height="8" rx="1" />
    <path d="M4 6.5h.01M6.5 6.5h.01M9 6.5h.01M11.5 6.5h.01M5 9.5h6" />
  </Icon>
);

/**
 * The brand mark, kept from the previous dashboard. Flat, one colour, no tile.
 */
export function FishMark(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 64 64"
      width="20"
      height="20"
      fill="currentColor"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      <path d="M10 33c10-15 27-18 40-5l8-8v26l-8-8C36 51 20 48 10 33Z" />
      {/* The eye is a hole punched in the mark, so it takes whatever is behind
          it — the dark rail or a light page — rather than a fixed colour. */}
      <circle cx="43" cy="30" r="2.5" fill="var(--mark-eye, var(--surface))" />
    </svg>
  );
}
