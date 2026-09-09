const state = {
  videos: [],
  selectedVideoId: null,
  system: null,
  showAllTracks: false,
  refreshing: false,
};

const elements = Object.fromEntries(
  [
    "refresh-button", "mode-banner", "mode-badge", "mode-title", "mode-copy",
    "compose-command", "upload-form", "video-file", "file-name", "camera-id",
    "upload-button", "upload-message", "video-count", "video-list", "detail-empty",
    "detail-content", "detail-meta", "detail-title", "detail-status", "job-error",
    "job-error-copy", "processing-note", "processing-title", "processing-copy",
    "process-button", "annotate-button", "summary-section", "summary-grid",
    "summary-pipeline", "summary-model", "summary-viame", "tracks-section",
    "show-all-tracks", "tracks-body", "tracks-empty", "annotation-section",
    "annotation-player", "download-link", "toast",
  ].map((id) => [id, document.getElementById(id)])
);

async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  if (!value) return "Unknown time";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatSeconds(value) {
  if (value === null || value === undefined) return "Not detected";
  const total = Math.max(0, Number(value));
  const minutes = Math.floor(total / 60);
  const seconds = total - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function formatBytes(value) {
  if (value === null || value === undefined) return "Size unavailable";
  if (!Number.isFinite(Number(value))) return "Size unavailable";
  const bytes = Number(value);
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function shortPath(value) {
  const parts = String(value || "Unknown").split(/[\\/]/);
  return parts.at(-1) || value;
}

function statusPill(status) {
  const safe = escapeHtml(status || "unknown");
  return `<span class="status-pill ${safe}">${safe}</span>`;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 4200);
}

function renderSystemStatus() {
  const system = state.system;
  if (!system) return;
  const worker = system.worker;
  elements["mode-badge"].textContent = `${system.processing_mode} mode`;
  elements["compose-command"].textContent = system.compose_command;
  elements["mode-banner"].classList.remove("warning", "error");
  if (!system.database.available) {
    elements["mode-banner"].classList.add("error");
    elements["mode-title"].textContent = "Database unavailable";
    elements["mode-copy"].textContent = system.database.message;
    return;
  }
  if (!worker.available) {
    elements["mode-banner"].classList.add("warning");
    elements["mode-title"].textContent = `No ${system.processing_mode.toUpperCase()} worker is available`;
    elements["mode-copy"].textContent = `${worker.message}. Start or restart the stack with the command shown here.`;
    return;
  }
  elements["mode-title"].textContent = `${system.processing_mode === "gpu" ? "Real VIAME GPU" : "Mock processing"} is ready`;
  elements["mode-copy"].textContent = `${worker.message}. Database is available. ${system.queue.queued} queued, ${system.queue.processing} processing.`;
}

function renderVideoList() {
  elements["video-count"].textContent = state.videos.length;
  if (!state.videos.length) {
    elements["video-list"].innerHTML = '<div class="list-empty">No videos yet. Upload your first file above.</div>';
    return;
  }
  elements["video-list"].innerHTML = state.videos.map((video) => {
    const selected = video.id === state.selectedVideoId ? " selected" : "";
    return `
      <div class="video-select-row"><input class="batch-select" type="checkbox" value="${escapeHtml(video.id)}" aria-label="Select ${escapeHtml(video.original_filename)} for batch">
      <button class="video-item${selected}" type="button" data-video-id="${escapeHtml(video.id)}">
        <span class="video-item-top">
          <span class="video-name" title="${escapeHtml(video.original_filename)}">${escapeHtml(video.original_filename)}</span>
          ${statusPill(video.processing_status)}
        </span>
        <span class="video-subtitle">${escapeHtml(formatDate(video.created_at))} · ${escapeHtml(formatBytes(video.size_bytes))}</span>
      </button></div>`;
  }).join("");
  elements["video-list"].querySelectorAll("[data-video-id]").forEach((button) => {
    button.addEventListener("click", () => selectVideo(button.dataset.videoId));
  });
  window.productUI?.syncSelection();
}

function currentVideo() {
  return state.videos.find((video) => video.id === state.selectedVideoId) || null;
}

async function selectVideo(videoId) {
  state.selectedVideoId = videoId;
  state.showAllTracks = false;
  elements["show-all-tracks"].checked = false;
  renderVideoList();
  await refreshDetail();
}

function renderDetailShell(video) {
  if (!video) {
    elements["detail-empty"].hidden = false;
    elements["detail-content"].hidden = true;
    return;
  }
  elements["detail-empty"].hidden = true;
  elements["detail-content"].hidden = false;
  elements["detail-title"].textContent = video.original_filename;
  elements["detail-meta"].textContent = `${formatDate(video.created_at)} · ${video.camera_id || "No camera ID"} · ${formatBytes(video.size_bytes)}`;
  elements["detail-status"].textContent = video.processing_status;
  elements["detail-status"].className = `status-pill ${video.processing_status}`;

  const active = ["queued", "processing"].includes(video.processing_status);
  elements["processing-note"].hidden = !active;
  if (active) {
    elements["processing-title"].textContent = video.processing_status === "queued" ? "Waiting for a worker" : "VIAME processing is in progress";
    elements["processing-copy"].textContent = "Status updates automatically. You can leave this page open.";
  }

  const error = video.latest_job?.status === "failed" ? video.latest_job.error_message : null;
  elements["job-error"].hidden = !error;
  elements["job-error-copy"].textContent = error || "";
  elements["process-button"].disabled = active;
  elements["process-button"].textContent = active
    ? (video.processing_status === "queued" ? "Queued" : "Processing…")
    : video.processing_status === "failed"
      ? "Retry processing"
      : video.processing_status === "completed"
        ? "Process again"
        : "Start processing";
  const completed = video.processing_status === "completed";
  elements["annotate-button"].hidden = !completed;
  elements["summary-section"].hidden = !completed;
  elements["tracks-section"].hidden = !completed;
  document.getElementById("video-analytics").hidden = !completed;
  document.getElementById("clips-button").hidden = !completed;
  document.getElementById("clips-section").hidden = !completed;
  renderAnnotation(video);
}

function renderSummary(summary) {
  const metrics = [
    [summary.fish_tracks, "Accepted fish tracks"],
    [summary.total_detections, "Accepted detections"],
    [formatSeconds(summary.first_fish_timestamp_seconds), "First fish"],
    [formatSeconds(summary.last_fish_timestamp_seconds), "Last fish"],
    [Number(summary.confidence_threshold).toFixed(2), "Confidence threshold"],
  ];
  elements["summary-grid"].innerHTML = metrics.map(([value, label]) => `
    <div class="metric-card">
      <span class="metric-value">${escapeHtml(value)}</span>
      <span class="metric-label">${escapeHtml(label)}</span>
    </div>`).join("");
  elements["summary-pipeline"].textContent = shortPath(summary.pipeline_name);
  elements["summary-pipeline"].title = summary.pipeline_name;
  elements["summary-model"].textContent = [summary.model_name, summary.model_version].filter(Boolean).join(" · ") || "Unknown";
  elements["summary-viame"].textContent = summary.viame_version || (state.system?.processing_mode === "mock" ? "mock" : "Not reported");
}

function renderTracks(tracks) {
  elements["tracks-body"].innerHTML = tracks.map((track) => `
    <tr>
      <td class="track-id"><button class="inspect-track text-link" data-track-id="${escapeHtml(track.id)}">#${escapeHtml(track.viame_track_id)}</button></td>
      <td><span class="track-result ${track.accepted ? "accepted" : "low"}">${track.machine_accepted ? "Above threshold" : "Low confidence"}<br>${track.accepted ? "Accepted" : "Excluded"}</span></td>
      <td><select class="review-select" data-track-id="${escapeHtml(track.id)}" aria-label="Review track ${escapeHtml(track.viame_track_id)}">${["unreviewed", "reviewed", "accepted", "rejected", "needs-review"].map(value => `<option value="${value}" ${track.review_state === value ? "selected" : ""}>${value}</option>`).join("")}</select></td>
      <td>${escapeHtml(track.detection_count)}</td>
      <td>${escapeHtml(formatSeconds(track.first_timestamp_seconds))}</td>
      <td>${escapeHtml(formatSeconds(track.last_timestamp_seconds))}</td>
      <td>${escapeHtml((Number(track.max_confidence) * 100).toFixed(1))}%</td>
      <td>${escapeHtml(track.species || "Unclassified")}</td>
    </tr>`).join("");
  elements["tracks-empty"].hidden = tracks.length > 0;
}

function renderAnnotation(video) {
  const available = video.processing_status === "completed" && Boolean(video.annotated_at);
  elements["annotation-section"].hidden = !available;
  if (!available) {
    elements["annotation-player"].removeAttribute("src");
    elements["annotation-player"].load();
    return;
  }
  const baseUrl = `/videos/${encodeURIComponent(video.id)}/annotated-video`;
  const src = `${baseUrl}?v=${encodeURIComponent(video.annotated_at)}`;
  if (elements["annotation-player"].getAttribute("src") !== src) {
    elements["annotation-player"].src = src;
  }
  elements["download-link"].href = `${baseUrl}?download=true`;
}

async function refreshDetail() {
  const video = currentVideo();
  const requestId = state.detailRequestId = (state.detailRequestId || 0) + 1;
  renderDetailShell(video);
  if (!video || video.processing_status !== "completed") return;
  try {
    const [summary, tracks] = await Promise.all([
      apiRequest(`/videos/${encodeURIComponent(video.id)}/summary`),
      apiRequest(`/videos/${encodeURIComponent(video.id)}/track-summaries?accepted_only=${state.showAllTracks ? "false" : "true"}`),
    ]);
    if (requestId !== state.detailRequestId || video.id !== state.selectedVideoId) return;
    renderSummary(summary);
    renderTracks(tracks);
    await window.productUI?.renderDetail(video);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function refreshAll() {
  if (document.activeElement?.matches("select") || document.getElementById("track-inspector")?.open) return;
  if (state.refreshing) return;
  state.refreshing = true;
  elements["refresh-button"].disabled = true;
  try {
    const [system, videos] = await Promise.all([
      apiRequest("/system/status"),
      apiRequest("/videos"),
    ]);
    state.system = system;
    state.videos = videos;
    if (state.selectedVideoId && !state.videos.some((video) => video.id === state.selectedVideoId)) {
      state.selectedVideoId = null;
    }
    renderSystemStatus();
    renderVideoList();
    await refreshDetail();
    await window.productUI?.renderOverview();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.refreshing = false;
    elements["refresh-button"].disabled = false;
  }
}

elements["video-file"].addEventListener("change", () => {
  elements["file-name"].textContent = elements["video-file"].files[0]?.name || "No file selected";
});

elements["upload-form"].addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = elements["video-file"].files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  const cameraId = elements["camera-id"].value.trim();
  if (cameraId) formData.append("camera_id", cameraId);
  elements["upload-button"].disabled = true;
  elements["upload-button"].textContent = "Uploading…";
  elements["upload-message"].textContent = "";
  try {
    const video = await apiRequest("/videos", { method: "POST", body: formData });
    elements["upload-form"].reset();
    elements["file-name"].textContent = "No file selected";
    state.selectedVideoId = video.id;
    showToast(`${video.original_filename} uploaded`);
    await refreshAll();
  } catch (error) {
    elements["upload-message"].textContent = error.message;
  } finally {
    elements["upload-button"].disabled = false;
    elements["upload-button"].textContent = "Upload video";
  }
});

elements["process-button"].addEventListener("click", async () => {
  const video = currentVideo();
  if (!video) return;
  elements["process-button"].disabled = true;
  try {
    await apiRequest(`/videos/${encodeURIComponent(video.id)}/process`, { method: "POST" });
    showToast("Processing job queued");
    await refreshAll();
  } catch (error) {
    showToast(error.message, true);
    elements["process-button"].disabled = false;
  }
});

elements["annotate-button"].addEventListener("click", async () => {
  const video = currentVideo();
  if (!video) return;
  elements["annotate-button"].disabled = true;
  elements["annotate-button"].textContent = "Generating…";
  try {
    await apiRequest(`/videos/${encodeURIComponent(video.id)}/annotate`, { method: "POST" });
    showToast("Annotated video is ready");
    await refreshAll();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements["annotate-button"].disabled = false;
    elements["annotate-button"].textContent = "Generate annotated video";
  }
});

elements["show-all-tracks"].addEventListener("change", async (event) => {
  state.showAllTracks = event.target.checked;
  await refreshDetail();
});

elements["refresh-button"].addEventListener("click", refreshAll);
window.setInterval(() => {
  if (state.videos.some((video) => ["queued", "processing"].includes(video.processing_status))) {
    refreshAll();
  }
}, 3000);
window.setInterval(() => {
  if (!state.videos.some((video) => ["queued", "processing"].includes(video.processing_status))) {
    refreshAll();
  }
}, 10000);

refreshAll();
