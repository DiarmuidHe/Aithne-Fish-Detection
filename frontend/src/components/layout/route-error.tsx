import { useQueryClient } from '@tanstack/react-query';
import { useRouteError } from 'react-router';

import { Button } from '@/components/base/button';
import { EmptyState } from '@/components/base/feedback';
import { AlertIcon } from '@/icons';

import styles from './shell.module.css';

/**
 * A route-level failure offers a refetch, not a page reload: reloading throws
 * away every other panel that loaded fine, and the filters the operator set.
 */
export function RouteError() {
  const error = useRouteError();
  const queryClient = useQueryClient();
  const message =
    error instanceof Error ? error.message : 'The screen could not be built from the API response.';

  return (
    <div className={styles.page}>
      <div className={styles.pageInner}>
        <EmptyState
          title="This screen did not load"
          body={
            <>
              <AlertIcon /> {message}
            </>
          }
          action={
            <Button
              variant="primary"
              onClick={() => {
                void queryClient.refetchQueries();
              }}
            >
              Retry
            </Button>
          }
        />
      </div>
    </div>
  );
}
