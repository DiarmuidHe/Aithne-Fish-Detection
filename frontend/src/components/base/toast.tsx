import { Toast } from '@base-ui/react/toast';

import { CloseIcon } from '@/icons';
import { Button } from './button';

import styles from './toast.module.css';

export const useToast = Toast.useToastManager;

/** Optimistic actions get 8 seconds to be undone before they settle. */
export const UNDO_TIMEOUT = 8_000;

function ToastList() {
  const { toasts } = Toast.useToastManager();
  return (
    <>
      {toasts.map((toast) => (
        <Toast.Root
          key={toast.id}
          toast={toast}
          className={styles.toast}
          data-tone={toast.type ?? 'info'}
        >
          <Toast.Content className={styles.content}>
            <div className={styles.text}>
              <Toast.Title className={styles.title} />
              <Toast.Description className={styles.description} />
            </div>
            {toast.actionProps ? (
              <Toast.Action render={<Button size="small" />} />
            ) : null}
            <Toast.Close
              render={<Button variant="quiet" size="small" iconOnly aria-label="Dismiss" />}
            >
              <CloseIcon />
            </Toast.Close>
          </Toast.Content>
        </Toast.Root>
      ))}
    </>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  return (
    <Toast.Provider>
      {children}
      <Toast.Portal>
        <Toast.Viewport className={styles.viewport}>
          <ToastList />
        </Toast.Viewport>
      </Toast.Portal>
    </Toast.Provider>
  );
}
