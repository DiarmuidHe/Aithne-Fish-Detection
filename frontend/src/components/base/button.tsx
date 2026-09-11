import * as React from 'react';
import { Link } from 'react-router';

import styles from './button.module.css';

export type ButtonVariant = 'default' | 'primary' | 'quiet' | 'danger' | 'link';

export interface ButtonOwnProps {
  variant?: ButtonVariant;
  size?: 'default' | 'small';
  iconOnly?: boolean;
}

export function buttonClass({
  variant = 'default',
  size = 'default',
  iconOnly = false,
  className,
}: ButtonOwnProps & { className?: string }): string {
  return [
    styles.button,
    variant !== 'default' ? styles[variant] : null,
    size === 'small' ? styles.small : null,
    iconOnly ? styles.iconOnly : null,
    className,
  ]
    .filter(Boolean)
    .join(' ');
}

export type ButtonProps = ButtonOwnProps & React.ButtonHTMLAttributes<HTMLButtonElement>;

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant, size, iconOnly, className, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={buttonClass({ variant, size, iconOnly, className })}
      {...props}
    />
  );
});

export type LinkButtonProps = ButtonOwnProps &
  React.ComponentProps<typeof Link>;

export function LinkButton({ variant, size, iconOnly, className, ...props }: LinkButtonProps) {
  return <Link className={buttonClass({ variant, size, iconOnly, className })} {...props} />;
}

export type AnchorButtonProps = ButtonOwnProps &
  React.AnchorHTMLAttributes<HTMLAnchorElement>;

/** For downloads and other real URLs the router must not intercept. */
export function AnchorButton({
  variant,
  size,
  iconOnly,
  className,
  ...props
}: AnchorButtonProps) {
  return <a className={buttonClass({ variant, size, iconOnly, className })} {...props} />;
}
