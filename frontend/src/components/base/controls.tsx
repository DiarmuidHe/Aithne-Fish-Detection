import * as React from 'react';
import { Checkbox } from '@base-ui/react/checkbox';
import { Field } from '@base-ui/react/field';
import { Fieldset } from '@base-ui/react/fieldset';
import { Form } from '@base-ui/react/form';
import { Input } from '@base-ui/react/input';
import { NumberField } from '@base-ui/react/number-field';
import { Slider } from '@base-ui/react/slider';
import { Switch } from '@base-ui/react/switch';
import { Toggle } from '@base-ui/react/toggle';
import { ToggleGroup } from '@base-ui/react/toggle-group';

import { CheckIcon, SearchIcon } from '@/icons';

import styles from './control.module.css';
import toggles from './toggle.module.css';

export { Field, Fieldset, Form };

/* --- Text ----------------------------------------------------------------- */

export interface TextFieldProps {
  label: string;
  name?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  description?: string;
  maxLength?: number;
  type?: string;
  required?: boolean;
}

export function TextField({
  label,
  name,
  value,
  onChange,
  placeholder,
  description,
  maxLength,
  type = 'text',
  required,
}: TextFieldProps) {
  return (
    <Field.Root name={name} className={styles.fieldRoot}>
      <Field.Label className={styles.fieldLabel}>{label}</Field.Label>
      <span className={styles.control}>
        <Input
          className={styles.input}
          type={type}
          value={value}
          placeholder={placeholder}
          maxLength={maxLength}
          required={required}
          onChange={(event) => onChange(event.target.value)}
        />
      </span>
      {description ? (
        <Field.Description className={styles.fieldDescription}>{description}</Field.Description>
      ) : null}
      <Field.Error className={styles.fieldError} />
    </Field.Root>
  );
}

export interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  label: string;
  placeholder?: string;
}

export function SearchInput({ value, onChange, label, placeholder }: SearchInputProps) {
  const id = React.useId();
  return (
    <span className={`${styles.control} ${styles.searchField}`}>
      <SearchIcon className={styles.triggerIcon} />
      <label className="visually-hidden" htmlFor={id}>
        {label}
      </label>
      <Input
        id={id}
        className={styles.input}
        type="search"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </span>
  );
}

/* --- Number --------------------------------------------------------------- */

export interface NumberInputProps {
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  description?: string;
  placeholder?: string;
}

export function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  description,
  placeholder,
}: NumberInputProps) {
  return (
    <NumberField.Root
      value={value}
      onValueChange={(next) => onChange(next ?? null)}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      className={styles.fieldRoot}
    >
      {/* A plain label, not a ScrubArea: dragging a number that spends API
          calls is not an affordance an operator should discover by accident. */}
      <span className={styles.fieldLabel}>{label}</span>
      <NumberField.Group className={styles.numberGroup}>
        <NumberField.Decrement className={styles.numberStep} aria-label={`Decrease ${label}`}>
          −
        </NumberField.Decrement>
        <NumberField.Input
          className={styles.numberInput}
          placeholder={placeholder}
          aria-label={label}
        />
        <NumberField.Increment className={styles.numberStep} aria-label={`Increase ${label}`}>
          +
        </NumberField.Increment>
      </NumberField.Group>
      {description ? (
        <span className={styles.fieldDescription}>{description}</span>
      ) : null}
    </NumberField.Root>
  );
}

/* --- Checkbox and switch -------------------------------------------------- */

export interface CheckProps {
  checked: boolean;
  indeterminate?: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: React.ReactNode;
  /** Hide the label visually but keep it for assistive technology. */
  hideLabel?: boolean;
  disabled?: boolean;
}

export function Check({
  checked,
  indeterminate,
  onCheckedChange,
  label,
  hideLabel,
  disabled,
}: CheckProps) {
  const control = (
    <Checkbox.Root
      className={styles.checkbox}
      checked={checked}
      indeterminate={indeterminate}
      disabled={disabled}
      onCheckedChange={onCheckedChange}
      aria-label={hideLabel && typeof label === 'string' ? label : undefined}
    >
      <Checkbox.Indicator className={styles.checkboxIndicator}>
        {indeterminate ? <MinusGlyph /> : <CheckIcon />}
      </Checkbox.Indicator>
    </Checkbox.Root>
  );
  if (hideLabel) return control;
  return (
    <label className={styles.checkboxLabel}>
      {control}
      <span>{label}</span>
    </label>
  );
}

function MinusGlyph() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M4 8h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export interface ToggleSwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: React.ReactNode;
  disabled?: boolean;
}

export function ToggleSwitch({ checked, onCheckedChange, label, disabled }: ToggleSwitchProps) {
  return (
    <label className={styles.checkboxLabel}>
      <Switch.Root
        className={styles.switch}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
      >
        <Switch.Thumb className={styles.switchThumb} />
      </Switch.Root>
      <span>{label}</span>
    </label>
  );
}

/* --- Toggles -------------------------------------------------------------- */

export interface ToggleOption {
  value: string;
  label: React.ReactNode;
  /** Read out instead of the visible label when the label is a glyph. */
  title?: string;
  disabled?: boolean;
}

export interface SegmentedProps {
  options: ToggleOption[];
  value: string[];
  onChange: (value: string[]) => void;
  multiple?: boolean;
  label: string;
  size?: 'default' | 'small';
}

/** A row of pressed/unpressed buttons: date presets, "has", review decisions. */
export function Segmented({
  options,
  value,
  onChange,
  multiple = false,
  label,
  size = 'default',
}: SegmentedProps) {
  return (
    <ToggleGroup
      className={`${toggles.group} ${size === 'small' ? toggles.small : ''}`}
      value={value}
      multiple={multiple}
      onValueChange={(next) => onChange(next)}
      aria-label={label}
    >
      {options.map((option) => (
        <Toggle
          key={option.value}
          value={option.value}
          className={toggles.item}
          disabled={option.disabled}
          title={option.title}
          aria-label={option.title}
        >
          {option.label}
        </Toggle>
      ))}
    </ToggleGroup>
  );
}

/* --- Slider --------------------------------------------------------------- */

export interface RangeSliderProps {
  label: string;
  value: [number, number];
  onChange: (value: [number, number]) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (value: number) => string;
}

export function RangeSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  format = (entry) => entry.toFixed(2),
}: RangeSliderProps) {
  return (
    <Slider.Root
      className={styles.sliderRoot}
      value={value}
      min={min}
      max={max}
      step={step}
      onValueChange={(next) => onChange(next as [number, number])}
    >
      <div className={styles.sliderHeader}>
        <Slider.Label>{label}</Slider.Label>
        <span className={styles.sliderValue}>
          {format(value[0])}–{format(value[1])}
        </span>
      </div>
      <Slider.Control className={styles.sliderControl}>
        <Slider.Track className={styles.sliderTrack}>
          <Slider.Indicator className={styles.sliderIndicator} />
          <Slider.Thumb className={styles.sliderThumb} index={0} />
          <Slider.Thumb className={styles.sliderThumb} index={1} />
        </Slider.Track>
      </Slider.Control>
    </Slider.Root>
  );
}
