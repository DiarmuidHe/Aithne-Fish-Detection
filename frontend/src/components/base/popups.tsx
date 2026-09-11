import * as React from 'react';
import { AlertDialog } from '@base-ui/react/alert-dialog';
import { Collapsible } from '@base-ui/react/collapsible';
import { ContextMenu } from '@base-ui/react/context-menu';
import { Dialog } from '@base-ui/react/dialog';
import { Menu } from '@base-ui/react/menu';
import { Popover } from '@base-ui/react/popover';
import { PreviewCard } from '@base-ui/react/preview-card';
import { Tooltip } from '@base-ui/react/tooltip';

import { CheckIcon, CloseIcon } from '@/icons';
import { Button } from './button';

import styles from './popup.module.css';

export { Collapsible, ContextMenu, Menu, Popover, PreviewCard, Tooltip };
export { styles as popupStyles };

/* --- Dialog --------------------------------------------------------------- */

export interface ModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: React.ReactNode;
  description?: React.ReactNode;
  narrow?: boolean;
  footer?: React.ReactNode;
  children: React.ReactNode;
}

export function Modal({
  open,
  onOpenChange,
  title,
  description,
  narrow,
  footer,
  children,
}: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Backdrop className={styles.backdrop} />
        <Dialog.Viewport className={styles.viewport}>
          <Dialog.Popup className={`${styles.dialog} ${narrow ? styles.dialogNarrow : ''}`}>
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{title}</Dialog.Title>
                {description ? (
                  <Dialog.Description className={styles.dialogDescription}>
                    {description}
                  </Dialog.Description>
                ) : null}
              </div>
              <Dialog.Close
                render={<Button variant="quiet" iconOnly aria-label="Close" />}
              >
                <CloseIcon />
              </Dialog.Close>
            </div>
            <div className={styles.dialogBody}>{children}</div>
            {footer ? <div className={styles.dialogFooter}>{footer}</div> : null}
          </Dialog.Popup>
        </Dialog.Viewport>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/* --- Alert dialog --------------------------------------------------------- */

export interface ConfirmProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: React.ReactNode;
  /** Say what will happen, in the operator's terms. */
  description: React.ReactNode;
  confirmLabel: string;
  onConfirm: () => void;
  destructive?: boolean;
}

export function Confirm({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
  destructive = true,
}: ConfirmProps) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className={styles.backdrop} />
        <AlertDialog.Viewport className={styles.viewport}>
          <AlertDialog.Popup className={`${styles.dialog} ${styles.dialogNarrow}`}>
            <div className={styles.dialogHeader}>
              <div>
                <AlertDialog.Title className={styles.dialogTitle}>{title}</AlertDialog.Title>
                <AlertDialog.Description className={styles.dialogDescription}>
                  {description}
                </AlertDialog.Description>
              </div>
            </div>
            <div className={styles.dialogFooter}>
              <AlertDialog.Close render={<Button variant="quiet" />}>Cancel</AlertDialog.Close>
              <Button variant={destructive ? 'danger' : 'primary'} onClick={onConfirm}>
                {confirmLabel}
              </Button>
            </div>
          </AlertDialog.Popup>
        </AlertDialog.Viewport>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}

/* --- Menu ----------------------------------------------------------------- */

export interface MenuAction {
  label: string;
  onSelect: () => void;
  disabled?: boolean;
}

export function ActionMenu({
  trigger,
  items,
}: {
  trigger: React.ReactElement;
  items: MenuAction[];
}) {
  return (
    <Menu.Root>
      <Menu.Trigger render={trigger} />
      <Menu.Portal>
        <Menu.Positioner className={styles.positioner} sideOffset={4} align="end">
          <Menu.Popup className={styles.popup}>
            {items.map((item) => (
              <Menu.Item
                key={item.label}
                className={styles.item}
                disabled={item.disabled}
                onClick={item.onSelect}
              >
                {item.label}
              </Menu.Item>
            ))}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>
  );
}

export interface CheckedMenuItem {
  value: string;
  label: string;
}

/** Column visibility and density: a menu of checkboxes, not a settings page. */
export function CheckedMenu({
  trigger,
  items,
  checked,
  onCheckedChange,
  label,
}: {
  trigger: React.ReactElement;
  items: CheckedMenuItem[];
  checked: string[];
  onCheckedChange: (value: string, next: boolean) => void;
  label: string;
}) {
  return (
    <Menu.Root>
      <Menu.Trigger render={trigger} />
      <Menu.Portal>
        <Menu.Positioner className={styles.positioner} sideOffset={4} align="end">
          <Menu.Popup className={styles.popup}>
            <Menu.GroupLabel className={styles.groupLabel}>{label}</Menu.GroupLabel>
            {items.map((item) => (
              <Menu.CheckboxItem
                key={item.value}
                className={styles.item}
                checked={checked.includes(item.value)}
                onCheckedChange={(next) => onCheckedChange(item.value, next)}
                closeOnClick={false}
              >
                <Menu.CheckboxItemIndicator className={styles.indicator}>
                  <CheckIcon />
                </Menu.CheckboxItemIndicator>
                {checked.includes(item.value) ? null : (
                  <span className={styles.indicator} aria-hidden="true" />
                )}
                {item.label}
              </Menu.CheckboxItem>
            ))}
          </Menu.Popup>
        </Menu.Positioner>
      </Menu.Portal>
    </Menu.Root>
  );
}

/* --- Tooltip -------------------------------------------------------------- */

export function Hint({
  content,
  children,
}: {
  content: React.ReactNode;
  children: React.ReactElement;
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger render={children} />
      <Tooltip.Portal>
        <Tooltip.Positioner className={styles.positioner} sideOffset={6}>
          <Tooltip.Popup className={styles.tooltip}>{content}</Tooltip.Popup>
        </Tooltip.Positioner>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

/* --- Popover -------------------------------------------------------------- */

export function PopoverPanel({
  trigger,
  title,
  children,
  align = 'start',
}: {
  trigger: React.ReactElement;
  title?: string;
  children: React.ReactNode;
  align?: 'start' | 'center' | 'end';
}) {
  return (
    <Popover.Root>
      <Popover.Trigger render={trigger} />
      <Popover.Portal>
        <Popover.Positioner className={styles.positioner} sideOffset={4} align={align}>
          <Popover.Popup className={styles.popup} style={{ padding: 'var(--space-4)' }}>
            {title ? (
              <Popover.Title className={styles.groupLabel}>{title}</Popover.Title>
            ) : null}
            {children}
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  );
}
