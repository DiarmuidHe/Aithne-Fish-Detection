import * as React from 'react';
import { Select } from '@base-ui/react/select';
import { Combobox } from '@base-ui/react/combobox';

import { CheckIcon, ChevronDownIcon, ChevronUpDownIcon, CloseIcon } from '@/icons';

import controls from './control.module.css';
import popup from './popup.module.css';

export interface Option {
  value: string;
  label: string;
  /** Shown right-aligned, e.g. how many videos a facet would match. */
  count?: number;
}

interface MultiSelectProps {
  label: string;
  options: Option[];
  value: string[];
  onChange: (value: string[]) => void;
  /** Shown in the popup when there is nothing to choose from. */
  emptyMessage?: string;
}

/**
 * A closed vocabulary chosen several at a time — status, review state.
 * The trigger always names the dimension, so the row reads as a filter bar
 * rather than a set of mystery values.
 */
export function MultiSelect({
  label,
  options,
  value,
  onChange,
  emptyMessage = 'Nothing to filter by yet.',
}: MultiSelectProps) {
  return (
    <Select.Root
      multiple
      value={value}
      onValueChange={(next) => onChange(next as string[])}
      items={options}
    >
      <Select.Trigger
        className={`${controls.control} ${controls.trigger}`}
        // A combobox takes its name from aria-label, never from its contents,
        // so the visible label has to be repeated here.
        aria-label={value.length > 0 ? `${label}, ${value.length} selected` : label}
      >
        <span className={controls.triggerLabel}>{label}</span>
        {value.length > 0 ? (
          <span className={controls.triggerCount}>{value.length}</span>
        ) : null}
        <Select.Icon className={controls.triggerIcon}>
          <ChevronDownIcon />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Positioner className={popup.positioner} sideOffset={4} alignItemWithTrigger={false}>
          <Select.Popup className={popup.popup}>
            {options.length === 0 ? (
              <p className={popup.groupLabel}>{emptyMessage}</p>
            ) : (
              <Select.List>
                {options.map((option) => (
                  <Select.Item
                    key={option.value}
                    value={option.value}
                    className={popup.item}
                  >
                    <Select.ItemIndicator className={popup.indicator}>
                      <CheckIcon />
                    </Select.ItemIndicator>
                    {value.includes(option.value) ? null : (
                      <span className={popup.indicator} aria-hidden="true" />
                    )}
                    <Select.ItemText>{option.label}</Select.ItemText>
                    {option.count === undefined ? null : (
                      <span className={popup.itemCount}>{option.count}</span>
                    )}
                  </Select.Item>
                ))}
              </Select.List>
            )}
          </Select.Popup>
        </Select.Positioner>
      </Select.Portal>
    </Select.Root>
  );
}

interface SingleSelectProps {
  label: string;
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  /** Show the chosen label on the trigger rather than only the dimension. */
  showValue?: boolean;
  id?: string;
}

export function SingleSelect({
  label,
  options,
  value,
  onChange,
  showValue = true,
  id,
}: SingleSelectProps) {
  const selected = options.find((option) => option.value === value);
  return (
    <Select.Root
      value={value}
      onValueChange={(next) => onChange(next as string)}
      items={options}
    >
      <Select.Trigger
        id={id}
        className={`${controls.control} ${controls.trigger}`}
        aria-label={selected ? `${label}, ${selected.label}` : label}
      >
        <span className={controls.triggerLabel}>{label}</span>
        {showValue ? (
          <span className={controls.triggerValue}>{selected?.label ?? value}</span>
        ) : null}
        <Select.Icon className={controls.triggerIcon}>
          <ChevronUpDownIcon />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Positioner className={popup.positioner} sideOffset={4} alignItemWithTrigger={false}>
          <Select.Popup className={popup.popup}>
            <Select.List>
              {options.map((option) => (
                <Select.Item key={option.value} value={option.value} className={popup.item}>
                  <Select.ItemIndicator className={popup.indicator}>
                    <CheckIcon />
                  </Select.ItemIndicator>
                  {option.value === value ? null : (
                    <span className={popup.indicator} aria-hidden="true" />
                  )}
                  <Select.ItemText>{option.label}</Select.ItemText>
                </Select.Item>
              ))}
            </Select.List>
          </Select.Popup>
        </Select.Positioner>
      </Select.Portal>
    </Select.Root>
  );
}

interface MultiComboboxProps {
  label: string;
  options: Option[];
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  emptyMessage?: string;
}

/**
 * The same job as MultiSelect, for a vocabulary that grows with the data —
 * camera IDs and species. Typing filters it, so a hundred species stays usable.
 */
export function MultiCombobox({
  label,
  options,
  value,
  onChange,
  placeholder = 'Type to filter',
  emptyMessage = 'No matches.',
}: MultiComboboxProps) {
  const id = React.useId();
  const labelOf = React.useCallback(
    (entry: string) => options.find((option) => option.value === entry)?.label ?? entry,
    [options],
  );

  return (
    <Combobox.Root
      multiple
      items={options.map((option) => option.value)}
      value={value}
      onValueChange={(next) => onChange(next as string[])}
      itemToStringLabel={labelOf}
    >
      <Combobox.InputGroup className={controls.control}>
        <label className={controls.triggerLabel} htmlFor={id}>
          {label}
        </label>
        <Combobox.Value>
          {(selected: string[]) => (
            <Combobox.Chips className={controls.chips}>
              {selected.length > 0 ? (
                <span className={controls.triggerCount}>{selected.length}</span>
              ) : null}
              <Combobox.Input
                id={id}
                className={controls.input}
                placeholder={selected.length > 0 ? '' : placeholder}
                aria-description={
                  selected.length > 0
                    ? `${selected.map(labelOf).join(', ')} selected`
                    : undefined
                }
              />
            </Combobox.Chips>
          )}
        </Combobox.Value>
        {value.length > 0 ? (
          <Combobox.Clear className={controls.triggerIcon} aria-label={`Clear ${label}`}>
            <CloseIcon />
          </Combobox.Clear>
        ) : null}
        <Combobox.Icon className={controls.triggerIcon}>
          <ChevronDownIcon />
        </Combobox.Icon>
      </Combobox.InputGroup>
      <Combobox.Portal>
        <Combobox.Positioner className={popup.positioner} sideOffset={4}>
          <Combobox.Popup className={popup.popup}>
            <Combobox.Empty>
              <p className={popup.groupLabel}>{emptyMessage}</p>
            </Combobox.Empty>
            <Combobox.List>
              {(entry: string) => (
                <Combobox.Item key={entry} value={entry} className={popup.item}>
                  <Combobox.ItemIndicator className={popup.indicator}>
                    <CheckIcon />
                  </Combobox.ItemIndicator>
                  {value.includes(entry) ? null : (
                    <span className={popup.indicator} aria-hidden="true" />
                  )}
                  <span>{labelOf(entry)}</span>
                  {options.find((option) => option.value === entry)?.count === undefined ? null : (
                    <span className={popup.itemCount}>
                      {options.find((option) => option.value === entry)?.count}
                    </span>
                  )}
                </Combobox.Item>
              )}
            </Combobox.List>
          </Combobox.Popup>
        </Combobox.Positioner>
      </Combobox.Portal>
    </Combobox.Root>
  );
}
