import * as React from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';

import { uploadVideo } from '@/api/videos';
import { Button } from '@/components/base/button';
import { Modal } from '@/components/base/popups';
import { TextField } from '@/components/base/controls';
import { PanelError } from '@/components/base/feedback';
import { useToast } from '@/components/base/toast';

const ACCEPT =
  '.mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska';

export function UploadDialog({
  open,
  onOpenChange,
  onUploaded,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUploaded: (videoId: string) => void;
}) {
  const toast = useToast();
  const queryClient = useQueryClient();
  const [file, setFile] = React.useState<File | null>(null);
  const [cameraId, setCameraId] = React.useState('');

  const upload = useMutation({
    mutationFn: () => uploadVideo(file as File, cameraId),
    onSuccess: (video) => {
      void queryClient.invalidateQueries({ queryKey: ['videos'] });
      void queryClient.invalidateQueries({ queryKey: ['facets'] });
      toast.add({ title: `${video.original_filename} uploaded`, type: 'success' });
      setFile(null);
      setCameraId('');
      onOpenChange(false);
      onUploaded(video.id);
    },
  });

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      narrow
      title="Upload an underwater video"
      description="MP4, MOV, AVI or MKV. Uploading stores the file; you choose when processing starts."
      footer={
        <>
          <Button variant="quiet" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={!file || upload.isPending}
            onClick={() => upload.mutate()}
          >
            {upload.isPending ? 'Uploading…' : 'Upload'}
          </Button>
        </>
      }
    >
      <div style={{ display: 'grid', gap: 'var(--space-4)' }}>
        <label style={{ display: 'grid', gap: 'var(--space-1)' }}>
          <span style={{ fontWeight: 'var(--weight-medium)', fontSize: 'var(--text-sm)' }}>
            Video file
          </span>
          <input
            type="file"
            accept={ACCEPT}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </label>

        <TextField
          label="Camera ID"
          value={cameraId}
          onChange={setCameraId}
          maxLength={128}
          placeholder="e.g. River camera 2"
          description="Optional. Used to group and filter the library later."
        />

        {upload.isError ? (
          <PanelError
            title="Upload did not complete"
            message={upload.error instanceof Error ? upload.error.message : 'Upload failed'}
            onRetry={() => upload.mutate()}
          />
        ) : null}
      </div>
    </Modal>
  );
}
