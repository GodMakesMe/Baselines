'use client';

import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Sun, Moon, Loader2, UploadCloud, CheckCircle, AlertTriangle, AlertCircle, XCircle } from 'lucide-react';

type OutputArtifact = {
  mimeType: string;
  encoding: 'base64' | 'utf8';
  data: string;
};

type OutputFiles = Record<string, OutputArtifact>;

type GroupedOutputs = {
  vis: OutputFiles;
  unet: OutputFiles;
  nnunet: OutputFiles;
  finalModel: OutputFiles;
  kaggleSeg: OutputFiles;
  textRecovery: OutputFiles;
  other: OutputFiles;
};

type ProcessStartResponse = {
  success: boolean;
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;cd 
  message: string;
};

type ProcessStatusResponse = {
  success: boolean;
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  message: string;
  logLines?: string[];
  files?: OutputFiles;
  error?: string;
  warnings?: string[];
  timings?: Record<string, number>;
};

const backendBaseUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:2632').replace(/\/$/, '');
const maxUploadSizeMb = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB || '95');
const maxUploadSizeBytes = maxUploadSizeMb * 1024 * 1024;

function groupOutputFiles(files: OutputFiles): GroupedOutputs {
  const grouped: GroupedOutputs = {
    vis: {},
    unet: {},
    nnunet: {},
    finalModel: {},
    kaggleSeg: {},
    textRecovery: {},
    other: {},
  };

  Object.entries(files).forEach(([filename, artifact]) => {
    if (filename.startsWith('vis/')) {
      grouped.vis[filename] = artifact;
      return;
    }

    if (filename.startsWith('unet/')) {
      grouped.unet[filename] = artifact;
      return;
    }

    if (filename.startsWith('nnunet/')) {
      grouped.nnunet[filename] = artifact;
      return;
    }

    if (filename.startsWith('final/')) {
      grouped.finalModel[filename] = artifact;
      return;
    }

    if (filename.startsWith('kaggle_seg/')) {
      grouped.kaggleSeg[filename] = artifact;
      return;
    }

    if (filename.startsWith('text_recovery/')) {
      grouped.textRecovery[filename] = artifact;
      return;
    }

    grouped.other[filename] = artifact;
  });

  return grouped;
}

function toDataUri(file: OutputArtifact): string {
  if (file.encoding === 'base64') {
    return `data:${file.mimeType};base64,${file.data}`;
  }

  return `data:${file.mimeType};charset=utf-8,${encodeURIComponent(file.data)}`;
}

function openArtifactInNewTab(file: OutputArtifact): void {
  const blob =
    file.encoding === 'utf8'
      ? new Blob([file.data], { type: file.mimeType })
      : new Blob([
          Uint8Array.from(window.atob(file.data), (character) => character.charCodeAt(0)),
        ], { type: file.mimeType });

  const url = URL.createObjectURL(blob);
  window.open(url, '_blank', 'noopener,noreferrer');
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

function getVolumeIdFromVisPath(filePath: string): string | null {
  const fileName = filePath.split('/').pop() || '';
  const match = fileName.match(/^([^_]+)_/);
  return match ? match[1] : null;
}

function OutputGroup({
  title,
  files,
  accentClassName,
  onImageClick,
}: {
  title: string;
  files: OutputFiles;
  accentClassName: string;
  onImageClick: (image: { src: string; name: string }) => void;
}) {
  const [openHtmlPreviews, setOpenHtmlPreviews] = useState<Record<string, boolean>>({});
  const images = Object.entries(files).filter(([, file]) => file.mimeType.startsWith('image/'));
  const htmlFiles = Object.entries(files).filter(([, file]) => file.mimeType === 'text/html');

  if (!images.length && !htmlFiles.length) {
    return null;
  }

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 p-5 shadow-lg shadow-slate-200/40 dark:shadow-black/40 backdrop-blur">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className={`text-lg font-semibold ${accentClassName}`}>{title}</h3>
        <span className="rounded-full bg-slate-100 dark:bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-700 dark:text-slate-300">
          {images.length + htmlFiles.length} file{images.length + htmlFiles.length === 1 ? '' : 's'}
        </span>
      </div>

      {images.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {images.map(([filename, file]) => (
            <div key={filename} className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
              <button
                type="button"
                onClick={() => onImageClick({ src: toDataUri(file), name: filename })}
                className="block w-full"
              >
                <img
                  src={toDataUri(file)}
                  alt={filename}
                  className="h-56 w-full cursor-zoom-in object-cover transition duration-200 hover:scale-[1.02]"
                />
              </button>
              <div className="border-t border-slate-200 px-3 py-2">
                <p className="truncate text-sm font-medium text-slate-700 dark:text-slate-300">{filename}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {htmlFiles.length > 0 && (
        <div className="mt-4 grid gap-4">
          {htmlFiles.map(([filename, file]) => (
            <div key={filename} className="overflow-hidden rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900">
              <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-700 px-4 py-3 gap-3">
                <p className="truncate text-sm font-medium text-slate-700 dark:text-slate-300">{filename}</p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setOpenHtmlPreviews((prev) => ({
                        ...prev,
                        [filename]: !prev[filename],
                      }))
                    }
                    className="rounded-md bg-slate-900 dark:bg-cyan-600 px-3 py-1.5 text-xs font-semibold text-cyan-100 dark:text-white hover:bg-slate-700 dark:hover:bg-cyan-500 shadow-sm"
                  >
                    {openHtmlPreviews[filename] ? 'Hide 3D' : 'Show 3D'}
                  </button>
                  <button
                    type="button"
                    onClick={() => openArtifactInNewTab(file)}
                    className="text-xs font-semibold text-cyan-700 dark:text-cyan-400 hover:text-cyan-900 dark:hover:text-cyan-300"
                  >
                    Open in New Tab
                  </button>
                </div>
              </div>
              {openHtmlPreviews[filename] ? (
                <iframe
                  title={filename}
                  srcDoc={file.encoding === 'utf8' ? file.data : ''}
                  src={file.encoding === 'base64' ? toDataUri(file) : undefined}
                  className="h-[520px] w-full bg-white"
                />
              ) : (
                <div className="flex h-40 items-center justify-center bg-slate-50 dark:bg-slate-800/30 text-sm text-slate-600 dark:text-slate-400">
                  3D preview is paused to save memory. Click Show 3D to load it.
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function VisualizationSection({
  files,
  onImageClick,
}: {
  files: OutputFiles;
  onImageClick: (image: { src: string; name: string }) => void;
}) {
  const [selectedVolume, setSelectedVolume] = useState<string>('all');

  const volumeIds = useMemo(
    () =>
      Array.from(
        new Set(
          Object.keys(files)
            .map((filePath) => getVolumeIdFromVisPath(filePath))
            .filter((value): value is string => Boolean(value)),
        ),
      ).sort(),
    [files],
  );

  const filteredFiles = useMemo(() => {
    if (selectedVolume === 'all') {
      return files;
    }

    return Object.fromEntries(
      Object.entries(files).filter(([filePath]) => getVolumeIdFromVisPath(filePath) === selectedVolume),
    );
  }, [files, selectedVolume]);

  if (!Object.keys(files).length) {
    return null;
  }

  return (
    <section className="rounded-2xl border border-cyan-200 dark:border-cyan-800 bg-white/90 dark:bg-slate-900/90 p-5 shadow-lg shadow-cyan-100/40 dark:shadow-cyan-900/20 backdrop-blur">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-cyan-800 dark:text-cyan-400">Visualization Output</h3>
        {volumeIds.length > 0 && (
          <label className="flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-300">
            Show TIFF:
            <select
              value={selectedVolume}
              onChange={(event) => setSelectedVolume(event.target.value)}
              className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-sm dark:text-slate-100"
            >
              <option value="all">All</option>
              {volumeIds.map((volumeId) => (
                <option key={volumeId} value={volumeId}>
                  {volumeId}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      <OutputGroup title="Visualization" files={filteredFiles} accentClassName="text-cyan-700" onImageClick={onImageClick} />
    </section>
  );
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<ProcessStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStage, setJobStage] = useState<string | null>(null);
  const [jobMessage, setJobMessage] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<{ src: string; name: string } | null>(null);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(false);
  const [activeResultTab, setActiveResultTab] = useState<'visualization' | 'segmentation' | 'text-recovery' | 'other'>('visualization');

  useEffect(() => {
    const root = window.document.documentElement;
    setIsDarkMode(root.classList.contains('dark'));
  }, []);

  const toggleDarkMode = () => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.remove('dark');
      setIsDarkMode(false);
    } else {
      root.classList.add('dark');
      setIsDarkMode(true);
    }
  };

  const groupedOutputs = response?.files ? groupOutputFiles(response.files) : null;

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files));
    }
  };

  const handleUpload = async () => {
    if (!files.length) {
      setError('Please select at least one file');
      return;
    }

    const totalSizeBytes = files.reduce((sum, file) => sum + file.size, 0);
    if (totalSizeBytes > maxUploadSizeBytes) {
      setError(
        `Selected files are too large (${(totalSizeBytes / (1024 * 1024)).toFixed(1)} MB). ` +
        `Current upload limit is ${maxUploadSizeMb} MB for this hosted endpoint. ` +
        'Please upload a smaller ZIP/TIFF batch.',
      );
      return;
    }

    setLoading(true);
    setError(null);
    setResponse(null);
    setJobId(null);
    setJobStage(null);
    setJobMessage(null);

    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      const tUploadStart = Date.now();
      const startRes = await axios.post<ProcessStartResponse>(`${backendBaseUrl}/api/scripts/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 0,
      });
      const uploadTime = Date.now() - tUploadStart;

      const startedJobId = startRes.data.jobId;
      setJobId(startedJobId);
      setJobStage(startRes.data.stage);
      setJobMessage(startRes.data.message);
      
      setResponse({
        success: true,
        jobId: startedJobId,
        status: startRes.data.status,
        stage: startRes.data.stage,
        message: startRes.data.message,
        timings: { uploadAndContact: uploadTime }
      } as ProcessStatusResponse);
      setActiveResultTab('visualization');

      let finished = false;
      let consecutivePollErrors = 0;
      while (!finished) {
        await new Promise((resolve) => setTimeout(resolve, 2500));
        let statusRes;
        try {
          statusRes = await axios.get<ProcessStatusResponse>(`${backendBaseUrl}/api/scripts/process/${startedJobId}`, {
            timeout: 0,
          });
          consecutivePollErrors = 0;
        } catch (pollError: any) {
          consecutivePollErrors += 1;
          setJobMessage(`Connection retry ${consecutivePollErrors}/10 while job is still running...`);
          if (consecutivePollErrors >= 10) {
            throw new Error(pollError?.response?.data?.error || pollError?.message || 'Repeated polling failures');
          }
          continue;
        }

        setJobStage(statusRes.data.stage);
        setJobMessage(statusRes.data.message);

        setResponse((prev) => {
          if (!prev) return statusRes.data;
          return {
            ...statusRes.data,
            timings: { ...prev.timings, ...(statusRes.data.timings || {}) }
          };
        });

        if (statusRes.data.status === 'completed') {
          finished = true;
          continue;
        }

        if (statusRes.data.status === 'failed') {
          throw new Error(statusRes.data.error || 'Backend processing failed');
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-300 px-4 py-8 md:px-8">
      <div className="fixed inset-0 z-0 pointer-events-none opacity-40 dark:opacity-20 [background-image:linear-gradient(to_right,#0f172a10_1px,transparent_1px),linear-gradient(to_bottom,#0f172a10_1px,transparent_1px)] dark:[background-image:linear-gradient(to_right,#e2e8f010_1px,transparent_1px),linear-gradient(to_bottom,#e2e8f010_1px,transparent_1px)] [background-size:40px_40px]" />
      <div className="fixed inset-0 z-0 pointer-events-none bg-[radial-gradient(circle_at_8%_0%,#ecfeff_0%,transparent_35%),radial-gradient(circle_at_100%_12%,#cffafe_0%,transparent_35%)] dark:bg-[radial-gradient(circle_at_8%_0%,#0891b210_0%,transparent_40%),radial-gradient(circle_at_100%_12%,#06b6d410_0%,transparent_40%)]" />

      <div className="relative z-10 mx-auto max-w-7xl space-y-8">
<header className="flex flex-wrap items-start justify-between gap-4 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 p-6 shadow-xl shadow-slate-200/50 dark:shadow-black/50 backdrop-blur-md">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700 dark:text-cyan-400">Vesuvius Pipeline</p>
            <h1 className="mt-2 text-4xl font-black tracking-tight text-slate-900 dark:text-white md:text-5xl">CV Project</h1>
            <p className="mt-3 max-w-3xl text-slate-600 dark:text-slate-400">
              Upload one ZIP file or multiple TIFF files. The backend runs visualization and model inference as a long-running background job.  
            </p>
          </div>
          <button onClick={toggleDarkMode} className="flex items-center justify-center rounded-full bg-slate-200 dark:bg-slate-800 p-3 text-slate-700 dark:text-cyan-400 transition-all hover:scale-105 hover:bg-slate-300 dark:hover:bg-slate-700 shadow-sm" aria-label="Toggle dark mode">
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </header>

        <section className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white/85 dark:bg-slate-900/85 p-6 shadow-lg shadow-slate-200/60 dark:shadow-black/60 backdrop-blur-md">
          <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
            <div>
              <label className="mb-2 block text-sm font-semibold text-slate-700 dark:text-slate-300">Input Files</label>
              <input
                type="file"
                multiple
                onChange={handleFileChange}
                accept=".tif,.tiff,.zip"
                className="block w-full cursor-pointer rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 px-4 py-3 text-sm text-slate-700 dark:text-slate-300 focus:outline-none transition-colors file:mr-4 file:rounded-xl file:border-0 file:bg-cyan-50 file:px-4 file:py-2.5 file:text-sm file:font-semibold file:text-cyan-700 hover:file:bg-cyan-100 dark:file:bg-cyan-900/40 dark:file:text-cyan-300 dark:hover:file:bg-cyan-900/70"
              />
              {files.length > 0 && (
                <div className="mt-3 grid max-h-36 gap-1 overflow-auto rounded-xl border border-emerald-200 dark:border-emerald-800/50 bg-emerald-50 dark:bg-emerald-900/20 p-3 text-sm text-emerald-700 dark:text-emerald-300">
                  {files.map((file) => (
                    <p key={file.name} className="truncate"> {file.name}</p>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={loading || !files.length}
              className="h-12 min-w-56 rounded-xl bg-slate-900 dark:bg-cyan-600 px-6 text-sm font-semibold text-white transition-all hover:bg-slate-800 dark:hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-slate-400 dark:disabled:bg-slate-700 shadow-lg shadow-slate-300/50 dark:shadow-black/50"
            >
              {loading ? 'Running Pipeline...' : 'Upload & Process'}
            </button>
          </div>

          {loading && (
            <div className="mt-4 rounded-xl border border-cyan-200 dark:border-cyan-800 bg-cyan-50 dark:bg-cyan-900/20 p-5 text-cyan-900 dark:text-cyan-100 shadow-sm transition-all">
              <p className="font-semibold flex items-center gap-2"><Loader2 className="animate-spin h-5 w-5"/>Processing job is running</p>
              {jobId && <p className="mt-1 text-sm">Job ID: {jobId}</p>}
              {jobStage && <p className="mt-1 text-sm">Stage: {jobStage}</p>}
              {jobMessage && <p className="mt-1 text-sm">{jobMessage}</p>}
              {response?.timings && (
                <div className="mt-3 grid gap-2 text-sm border-t border-cyan-200/50 dark:border-cyan-800/50 pt-3">
                  {response?.timings?.uploadAndContact && <div className="flex items-center gap-2"><CheckCircle className="h-4 w-4 text-emerald-500" /><span>Upload & Init: {(response.timings.uploadAndContact / 1000).toFixed(1)}s</span></div>}
                  {response?.timings?.visualization ? <div className="flex items-center gap-2"><CheckCircle className="h-4 w-4 text-emerald-500" /><span>Visualization: {(response.timings.visualization / 1000).toFixed(1)}s</span></div> : <div className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin text-cyan-500/70" /><span>Waiting for Visualization</span></div>}
                  {response?.timings?.modelInference ? <div className="flex items-center gap-2"><CheckCircle className="h-4 w-4 text-emerald-500" /><span>Model Inference: {(response.timings.modelInference / 1000).toFixed(1)}s</span></div> : (response?.timings?.visualization ? <div className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin text-cyan-500/70" /><span>Waiting for Model Inference</span></div> : null)}
                  {response?.timings?.kaggleSegmentation ? <div className="flex items-center gap-2"><CheckCircle className="h-4 w-4 text-emerald-500" /><span>Kaggle Segmentation: {(response.timings.kaggleSegmentation / 1000).toFixed(1)}s</span></div> : null}
                  {response?.timings?.inkDetection ? <div className="flex items-center gap-2"><CheckCircle className="h-4 w-4 text-emerald-500" /><span>Ink Detection: {(response.timings.inkDetection / 1000).toFixed(1)}s</span></div> : null}
                </div>
              )}
              {response?.logLines && response.logLines.length > 0 && (
                <div className="mt-3 border-t border-cyan-200/50 dark:border-cyan-800/50 pt-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-cyan-800 dark:text-cyan-300">Live Logs</p>
                  <pre className="mt-2 max-h-44 overflow-auto rounded-lg bg-slate-900 text-cyan-200 p-3 text-xs leading-relaxed whitespace-pre-wrap">
                    {response.logLines.slice(-30).join('\n')}
                  </pre>
                </div>
              )}
              <p className="mt-4 text-sm opacity-80 text-cyan-800 dark:text-cyan-300">Large TIFF sets can take time. You can keep the tab open while status updates continue.</p>
            </div>
          )}

          {error && <div className="mt-4 rounded-xl border border-rose-300 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 p-4 text-rose-800 dark:text-rose-200 flex items-start gap-3">{error}</div>}

          {response?.warnings && response.warnings.length > 0 && (
            <div className="mt-4 rounded-xl border border-amber-300 dark:border-amber-700/50 bg-amber-50 dark:bg-amber-900/20 p-5 text-amber-900 dark:text-amber-200">
              <p className="font-semibold">Completed with warnings</p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
                {response.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {response && !loading && (
          <section className="rounded-3xl border border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 p-6 shadow-lg shadow-emerald-200/50 dark:shadow-black/30 backdrop-blur-md">
            <h2 className="text-xl font-bold text-emerald-800 dark:text-emerald-100 flex items-center gap-2"><CheckCircle className="h-6 w-6"/>Processing Complete</h2>
            <p className="mt-1 text-emerald-700 dark:text-emerald-300">{response.message}</p>
            {response?.timings && (
              <div className="mt-4 grid gap-2 text-sm border-t border-emerald-200/60 dark:border-emerald-800/60 pt-4 text-emerald-800 dark:text-emerald-200">
                {response?.timings?.uploadAndContact && <p>Upload & Init: {(response.timings.uploadAndContact / 1000).toFixed(1)}s</p>}
                {response?.timings?.visualization && <p>Visualization: {(response.timings.visualization / 1000).toFixed(1)}s</p>}
                {response?.timings?.modelInference && <p>Model Inference: {(response.timings.modelInference / 1000).toFixed(1)}s</p>}
                {response?.timings?.kaggleSegmentation && <p>Kaggle Segmentation: {(response.timings.kaggleSegmentation / 1000).toFixed(1)}s</p>}
                {response?.timings?.inkDetection && <p>Ink Detection: {(response.timings.inkDetection / 1000).toFixed(1)}s</p>}
              </div>
            )}
          </section>
        )}

        {groupedOutputs && (
          <div className="space-y-6">
            <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 p-4 shadow-sm">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setActiveResultTab('visualization')}
                  className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${activeResultTab === 'visualization' ? 'bg-cyan-600 text-white' : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200'}`}
                >
                  Visualization
                </button>
                <button
                  type="button"
                  onClick={() => setActiveResultTab('segmentation')}
                  className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${activeResultTab === 'segmentation' ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200'}`}
                >
                  Segmentation
                </button>
                <button
                  type="button"
                  onClick={() => setActiveResultTab('text-recovery')}
                  className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${activeResultTab === 'text-recovery' ? 'bg-amber-600 text-white' : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200'}`}
                >
                  Text Recovery
                </button>
                <button
                  type="button"
                  onClick={() => setActiveResultTab('other')}
                  className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${activeResultTab === 'other' ? 'bg-slate-700 text-white' : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200'}`}
                >
                  Other
                </button>
              </div>
            </section>

            {activeResultTab === 'visualization' && (
              <VisualizationSection files={groupedOutputs.vis} onImageClick={setSelectedImage} />
            )}

            {activeResultTab === 'segmentation' && (
              <>
                <OutputGroup title="UNet Output" files={groupedOutputs.unet} accentClassName="text-emerald-700 dark:text-emerald-300" onImageClick={setSelectedImage} />
                <OutputGroup title="nnUNet Output" files={groupedOutputs.nnunet} accentClassName="text-violet-700" onImageClick={setSelectedImage} />
                <OutputGroup title="Final Model Output" files={groupedOutputs.finalModel} accentClassName="text-fuchsia-700 dark:text-fuchsia-300" onImageClick={setSelectedImage} />
                <OutputGroup title="Kaggle 1st-Place Segmentation" files={groupedOutputs.kaggleSeg} accentClassName="text-lime-700 dark:text-lime-300" onImageClick={setSelectedImage} />
              </>
            )}

            {activeResultTab === 'text-recovery' && (
              <OutputGroup title="Recovered Text (Ink Detection)" files={groupedOutputs.textRecovery} accentClassName="text-amber-700 dark:text-amber-300" onImageClick={setSelectedImage} />
            )}

            {activeResultTab === 'other' && (
              <OutputGroup title="Model Inference" files={groupedOutputs.other} accentClassName="text-slate-700" onImageClick={setSelectedImage} />
            )}
          </div>
        )}
      </div>

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
    </main>
  );
}
