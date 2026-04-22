'use client';

import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Loader2, CheckCircle, AlertTriangle, ScanLine, RefreshCw } from 'lucide-react';

type OutputArtifact = {
  mimeType: string;
  encoding: 'base64' | 'utf8';
  data: string;
};

type OutputFiles = Record<string, OutputArtifact>;

type Segment = {
  id: string;
  layerCount: number;
  hasMask: boolean;
  dimensionsHint?: { width: number; height: number };
};

type HealthModel = {
  key: string;
  label: string;
  description: string;
  available: boolean;
};

type HealthResponse = {
  ok: boolean;
  sourceDir: string | null;
  sourceDirExists: boolean;
  defaultModel: string;
  models: HealthModel[];
};

type SegmentsResponse = {
  sourceDir: string;
  segments: Segment[];
};

type InkStartResponse = {
  success: boolean;
  jobId: string;
  segments: string[];
  model: string;
};

type InkStatusResponse = {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  message: string;
  logLines?: string[];
  files?: OutputFiles;
  segments: string[];
  model: string;
  error?: string;
  timings?: Record<string, number>;
};

const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:2632').replace(/\/$/, '');

function toDataUri(file: OutputArtifact): string {
  if (file.encoding === 'base64') {
    return `data:${file.mimeType};base64,${file.data}`;
  }
  return `data:${file.mimeType};charset=utf-8,${encodeURIComponent(file.data)}`;
}

type VariantKey = 'prediction' | 'enhanced' | 'binary' | 'other';

function classifyVariant(filename: string): VariantKey {
  const name = filename.toLowerCase();
  if (name.endsWith('_binary.png')) return 'binary';
  if (name.endsWith('_enhanced.png')) return 'enhanced';
  if (name.includes('prediction')) return 'prediction';
  return 'other';
}

const VARIANT_META: Record<VariantKey, { label: string; accent: string; blurb: string }> = {
  prediction: {
    label: 'Raw prediction',
    accent: 'text-cyan-700 dark:text-cyan-300',
    blurb: 'Direct TimeSformer probability map (rotated) from the Winner checkpoint.',
  },
  enhanced: {
    label: 'Percentile-stretched',
    accent: 'text-violet-700 dark:text-violet-300',
    blurb: 'Post-processed contrast stretch (5–99th percentile) for visual readability.',
  },
  binary: {
    label: 'Binary mask',
    accent: 'text-fuchsia-700 dark:text-fuchsia-300',
    blurb: 'Thresholded at the 85th percentile — pixels the model is most confident contain ink.',
  },
  other: {
    label: 'Other artifacts',
    accent: 'text-slate-700 dark:text-slate-300',
    blurb: 'Additional files emitted by the Winner inference script.',
  },
};

function groupByVariant(files: OutputFiles): Record<VariantKey, OutputFiles> {
  const grouped: Record<VariantKey, OutputFiles> = {
    prediction: {},
    enhanced: {},
    binary: {},
    other: {},
  };
  for (const [filename, artifact] of Object.entries(files)) {
    if (!artifact.mimeType.startsWith('image/')) continue;
    grouped[classifyVariant(filename)][filename] = artifact;
  }
  return grouped;
}

function VariantGallery({
  variant,
  files,
  onImageClick,
}: {
  variant: VariantKey;
  files: OutputFiles;
  onImageClick: (image: { src: string; name: string }) => void;
}) {
  const entries = Object.entries(files);
  if (!entries.length) return null;
  const meta = VARIANT_META[variant];
  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 p-5 shadow-lg shadow-slate-200/40 dark:shadow-black/40 backdrop-blur">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className={`text-lg font-semibold ${meta.accent}`}>{meta.label}</h3>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{meta.blurb}</p>
        </div>
        <span className="rounded-full bg-slate-100 dark:bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-700 dark:text-slate-300">
          {entries.length} file{entries.length === 1 ? '' : 's'}
        </span>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {entries.map(([filename, file]) => (
          <div key={filename} className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
            <button
              type="button"
              onClick={() => onImageClick({ src: toDataUri(file), name: filename })}
              className="block w-full"
            >
              <img
                src={toDataUri(file)}
                alt={filename}
                className="h-56 w-full cursor-zoom-in object-contain bg-black/90 transition duration-200 hover:scale-[1.02]"
              />
            </button>
            <div className="border-t border-slate-200 dark:border-slate-700 px-3 py-2">
              <p className="truncate text-sm font-medium text-slate-700 dark:text-slate-300">{filename}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function formatSeconds(ms: number | undefined) {
  if (!ms || !Number.isFinite(ms)) return null;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function InkPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [sourceDir, setSourceDir] = useState<string>('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [model, setModel] = useState<string>('timesformer_64');
  const [loadingSegments, setLoadingSegments] = useState<boolean>(true);
  const [segmentsError, setSegmentsError] = useState<string | null>(null);

  const [submitting, setSubmitting] = useState<boolean>(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<InkStatusResponse | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<{ src: string; name: string } | null>(null);

  const fetchEverything = async () => {
    setLoadingSegments(true);
    setSegmentsError(null);
    try {
      const [healthRes, segRes] = await Promise.all([
        axios.get<HealthResponse>(`${backendBaseUrl}/api/ink/health`, { timeout: 10_000 }),
        axios.get<SegmentsResponse>(`${backendBaseUrl}/api/ink/segments`, { timeout: 10_000 }),
      ]);
      setHealth(healthRes.data);
      setSegments(segRes.data.segments ?? []);
      setSourceDir(segRes.data.sourceDir ?? '');
      const preferred = healthRes.data.defaultModel;
      if (preferred && healthRes.data.models.some((m) => m.key === preferred)) {
        setModel(preferred);
      } else if (healthRes.data.models.length > 0) {
        setModel(healthRes.data.models[0].key);
      }
    } catch (err: any) {
      setSegmentsError(err?.response?.data?.error || err?.message || 'Failed to load ink dataset');
    } finally {
      setLoadingSegments(false);
    }
  };

  useEffect(() => {
    void fetchEverything();
  }, []);

  const toggleSegment = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => setSelected(new Set(segments.map((s) => s.id)));
  const clearAll = () => setSelected(new Set());

  const handleRun = async () => {
    if (selected.size === 0) {
      setRunError('Pick at least one segment to run ink detection on.');
      return;
    }
    setSubmitting(true);
    setRunError(null);
    setStatus(null);
    setJobId(null);
    try {
      const startRes = await axios.post<InkStartResponse>(`${backendBaseUrl}/api/ink/process`, {
        segments: Array.from(selected),
        model,
      }, { timeout: 30_000 });

      const startedJobId = startRes.data.jobId;
      setJobId(startedJobId);
      setStatus({
        jobId: startedJobId,
        status: 'queued',
        stage: 'queued',
        message: 'Queued',
        segments: startRes.data.segments,
        model: startRes.data.model,
      });

      let finished = false;
      let consecutivePollErrors = 0;
      while (!finished) {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        try {
          const statusRes = await axios.get<InkStatusResponse>(
            `${backendBaseUrl}/api/ink/process/${startedJobId}`,
            { timeout: 20_000 },
          );
          consecutivePollErrors = 0;
          setStatus(statusRes.data);
          if (statusRes.data.status === 'completed') {
            finished = true;
          } else if (statusRes.data.status === 'failed') {
            throw new Error(statusRes.data.error || statusRes.data.message || 'Ink detection failed');
          }
        } catch (pollError: any) {
          consecutivePollErrors += 1;
          if (consecutivePollErrors >= 10) {
            throw new Error(pollError?.response?.data?.error || pollError?.message || 'Repeated polling failures');
          }
        }
      }
    } catch (err: any) {
      setRunError(err?.response?.data?.error || err?.message || 'Ink detection failed');
    } finally {
      setSubmitting(false);
    }
  };

  const variantGroups = useMemo(() => (status?.files ? groupByVariant(status.files) : null), [status?.files]);

  const modelOptions = health?.models ?? [];
  const anyModelAvailable = modelOptions.some((m) => m.available);

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      <header className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 p-6 shadow-xl shadow-slate-200/50 dark:shadow-black/50 backdrop-blur-md">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-700 dark:text-violet-400">Ink Detection</p>
        <h1 className="mt-2 text-4xl font-black tracking-tight text-slate-900 dark:text-white md:text-5xl flex items-center gap-3">
          <ScanLine className="h-9 w-9 text-violet-600 dark:text-violet-400" />
          TimeSformer (Winner)
        </h1>
        <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
          Runs the Vesuvius Grand Prize Winner checkpoint on the pre-staged ink-detection test set
          {' '}(<span className="font-mono text-slate-800 dark:text-slate-200">{sourceDir || 'INK_TEST_DIR'}</span>).
          Pick a segment and a checkpoint, then kick off inference — outputs are raw probability maps plus stretched & binarized variants.
        </p>
      </header>

      <section className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white/85 dark:bg-slate-900/85 p-6 shadow-lg shadow-slate-200/60 dark:shadow-black/60 backdrop-blur-md">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Available segments</h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={fetchEverything}
              className="flex items-center gap-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700"
            >
              <RefreshCw size={14} /> Refresh
            </button>
            <button
              type="button"
              onClick={selectAll}
              disabled={!segments.length}
              className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50"
            >
              Select all
            </button>
            <button
              type="button"
              onClick={clearAll}
              disabled={!selected.size}
              className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50"
            >
              Clear
            </button>
          </div>
        </div>

        {loadingSegments && (
          <p className="mt-4 flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading segments from backend…
          </p>
        )}

        {segmentsError && (
          <div className="mt-4 flex items-start gap-3 rounded-xl border border-rose-300 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 p-4 text-rose-800 dark:text-rose-200">
            <AlertTriangle className="h-5 w-5 shrink-0" />
            <div>
              <p className="font-semibold">Could not load segments</p>
              <p className="mt-1 text-sm">{segmentsError}</p>
            </div>
          </div>
        )}

        {!loadingSegments && !segmentsError && segments.length === 0 && (
          <p className="mt-4 rounded-xl border border-amber-300 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-4 text-sm text-amber-900 dark:text-amber-200">
            No segments found in <span className="font-mono">{sourceDir || 'INK_TEST_DIR'}</span>. Each segment needs a
            {' '}<span className="font-mono">surface_volume/</span> folder with layered TIFFs.
          </p>
        )}

        {segments.length > 0 && (
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {segments.map((segment) => {
              const isSelected = selected.has(segment.id);
              return (
                <label
                  key={segment.id}
                  className={`cursor-pointer rounded-2xl border p-4 transition ${
                    isSelected
                      ? 'border-violet-400 dark:border-violet-500 bg-violet-50 dark:bg-violet-900/20 shadow-md'
                      : 'border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/60 hover:border-slate-300 dark:hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-mono text-base font-semibold text-slate-900 dark:text-white">{segment.id}</p>
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        {segment.layerCount} layers · {segment.hasMask ? 'mask ✓' : 'no mask'}
                      </p>
                      {segment.dimensionsHint && (
                        <p className="mt-1 text-xs font-mono text-slate-500 dark:text-slate-400">
                          {segment.dimensionsHint.width} × {segment.dimensionsHint.height}
                        </p>
                      )}
                    </div>
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleSegment(segment.id)}
                      className="mt-1 h-5 w-5 cursor-pointer accent-violet-600"
                    />
                  </div>
                </label>
              );
            })}
          </div>
        )}
      </section>

      <section className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white/85 dark:bg-slate-900/85 p-6 shadow-lg shadow-slate-200/60 dark:shadow-black/60 backdrop-blur-md">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Checkpoint</h2>

        {!modelOptions.length && (
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">Loading model list…</p>
        )}

        {modelOptions.length > 0 && !anyModelAvailable && (
          <div className="mt-3 flex items-start gap-3 rounded-xl border border-amber-300 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-4 text-amber-900 dark:text-amber-200">
            <AlertTriangle className="h-5 w-5 shrink-0" />
            <div>
              <p className="font-semibold">No Winner checkpoints found on disk</p>
              <p className="mt-1 text-sm">
                Set <span className="font-mono">INK_CHECKPOINT_TIMESFORMER_64</span> (and optionally{' '}
                <span className="font-mono">INK_CHECKPOINT_TIMESFORMER_256</span>) in the backend{' '}
                <span className="font-mono">.env</span> so they point at existing <span className="font-mono">.ckpt</span> files.
              </p>
            </div>
          </div>
        )}

        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {modelOptions.map((m) => {
            const isSelected = model === m.key;
            const disabled = !m.available;
            return (
              <label
                key={m.key}
                className={`cursor-pointer rounded-2xl border p-4 transition ${
                  disabled
                    ? 'border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 opacity-60 cursor-not-allowed'
                    : isSelected
                      ? 'border-violet-400 dark:border-violet-500 bg-violet-50 dark:bg-violet-900/20 shadow-md'
                      : 'border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/60 hover:border-slate-300 dark:hover:border-slate-700'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-semibold text-slate-900 dark:text-white">{m.label}</p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{m.description}</p>
                    {!m.available && (
                      <p className="mt-2 text-xs font-semibold text-rose-600 dark:text-rose-400">Checkpoint file not found</p>
                    )}
                  </div>
                  <input
                    type="radio"
                    name="ink-model"
                    checked={isSelected}
                    onChange={() => setModel(m.key)}
                    disabled={disabled}
                    className="mt-1 h-5 w-5 cursor-pointer accent-violet-600 disabled:cursor-not-allowed"
                  />
                </div>
              </label>
            );
          })}
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 dark:border-slate-800 pt-4">
          <p className="text-sm text-slate-600 dark:text-slate-400">
            {selected.size} segment{selected.size === 1 ? '' : 's'} selected · model <span className="font-mono">{model}</span>
          </p>
          <button
            type="button"
            onClick={handleRun}
            disabled={submitting || selected.size === 0 || !anyModelAvailable}
            className="h-11 min-w-56 rounded-xl bg-violet-600 px-6 text-sm font-semibold text-white shadow-lg shadow-violet-500/30 transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:bg-slate-400 dark:disabled:bg-slate-700"
          >
            {submitting ? 'Running ink detection…' : 'Run ink detection'}
          </button>
        </div>

        {runError && (
          <div className="mt-4 flex items-start gap-3 rounded-xl border border-rose-300 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 p-4 text-rose-800 dark:text-rose-200">
            <AlertTriangle className="h-5 w-5 shrink-0" />
            <p className="text-sm">{runError}</p>
          </div>
        )}
      </section>

      {submitting && status && (
        <section className="rounded-2xl border border-violet-200 dark:border-violet-800 bg-violet-50 dark:bg-violet-900/20 p-5 text-violet-900 dark:text-violet-100 shadow-sm">
          <p className="flex items-center gap-2 font-semibold"><Loader2 className="h-5 w-5 animate-spin" />Ink detection is running</p>
          <p className="mt-1 text-sm">Job ID: {status.jobId}</p>
          <p className="mt-1 text-sm">Stage: {status.stage}</p>
          <p className="mt-1 text-sm">{status.message}</p>
          {status.logLines && status.logLines.length > 0 && (
            <div className="mt-3 border-t border-violet-200/50 dark:border-violet-800/50 pt-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-violet-800 dark:text-violet-300">Live Logs</p>
              <pre className="mt-2 max-h-56 overflow-auto rounded-lg bg-slate-900 text-violet-200 p-3 text-xs leading-relaxed whitespace-pre-wrap">
                {status.logLines.slice(-40).join('\n')}
              </pre>
            </div>
          )}
          <p className="mt-4 text-xs opacity-80">Large fragments can take 10–30 minutes per segment on 4 GB VRAM. Keep this tab open.</p>
        </section>
      )}

      {!submitting && status?.status === 'completed' && (
        <section className="rounded-3xl border border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 p-6 shadow-lg shadow-emerald-200/50 dark:shadow-black/30 backdrop-blur-md">
          <h2 className="text-xl font-bold text-emerald-800 dark:text-emerald-100 flex items-center gap-2"><CheckCircle className="h-6 w-6"/>Ink detection complete</h2>
          <p className="mt-1 text-emerald-700 dark:text-emerald-300">{status.message}</p>
          <p className="mt-1 text-sm text-emerald-800 dark:text-emerald-200">
            Ran <span className="font-mono">{status.model}</span> on [{status.segments.join(', ')}]
            {status.timings?.inference ? ` — ${formatSeconds(status.timings.inference)}` : ''}.
          </p>
        </section>
      )}

      {variantGroups && status?.status === 'completed' && (
        <div className="space-y-6">
          <VariantGallery variant="prediction" files={variantGroups.prediction} onImageClick={setSelectedImage} />
          <VariantGallery variant="enhanced" files={variantGroups.enhanced} onImageClick={setSelectedImage} />
          <VariantGallery variant="binary" files={variantGroups.binary} onImageClick={setSelectedImage} />
          <VariantGallery variant="other" files={variantGroups.other} onImageClick={setSelectedImage} />
        </div>
      )}

      {selectedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/90 dark:bg-black/90 backdrop-blur-sm p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-h-[95vh] max-w-[95vw]" onClick={(event) => event.stopPropagation()}>
            <button
              type="button"
              onClick={() => setSelectedImage(null)}
              className="absolute -right-3 -top-3 rounded-full bg-white dark:bg-slate-800 p-2 text-slate-900 dark:text-slate-100 shadow-xl border border-slate-200 dark:border-slate-700 hover:scale-110 hover:bg-rose-50 dark:hover:bg-rose-900 opacity-90 hover:opacity-100"
            >
              X
            </button>
            <img
              src={selectedImage.src}
              alt={selectedImage.name}
              className="max-h-[85vh] max-w-[92vw] rounded-xl border border-slate-300 dark:border-slate-700 object-contain shadow-2xl bg-white/50 dark:bg-black/50"
            />
            <p className="mt-2 text-center text-sm text-slate-200">{selectedImage.name}</p>
          </div>
        </div>
      )}
    </div>
  );
}
