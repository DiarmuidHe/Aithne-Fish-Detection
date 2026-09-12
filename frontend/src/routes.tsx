import { createBrowserRouter, Navigate } from 'react-router';

import { AppShell } from '@/components/layout/shell';
import { RouteError } from '@/components/layout/route-error';

// Each route is fetched when it is first opened. An operator working through
// the library should not pay for the live monitoring screen's code.
export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      {
        index: true,
        errorElement: <RouteError />,
        lazy: async () => ({
          Component: (await import('@/features/overview/overview-route')).OverviewRoute,
        }),
      },
      {
        path: 'library',
        errorElement: <RouteError />,
        lazy: async () => ({
          Component: (await import('@/features/library/library-route')).LibraryRoute,
        }),
      },
      {
        path: 'library/:videoId',
        errorElement: <RouteError />,
        lazy: async () => ({
          Component: (await import('@/features/library/library-route')).LibraryRoute,
        }),
      },
      {
        path: 'review',
        errorElement: <RouteError />,
        lazy: async () => ({
          Component: (await import('@/features/review-queue/review-queue-route'))
            .ReviewQueueRoute,
        }),
      },
      {
        path: 'live',
        errorElement: <RouteError />,
        lazy: async () => ({
          Component: (await import('@/features/live/live-route')).LiveRoute,
        }),
      },
      {
        path: 'settings',
        errorElement: <RouteError />,
        lazy: async () => ({
          Component: (await import('@/features/system/system-route')).SystemRoute,
        }),
      },
      // The old dashboard URL still points at something useful.
      { path: 'dashboard', element: <Navigate to="/" replace /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);
