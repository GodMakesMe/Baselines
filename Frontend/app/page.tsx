'use client';

import { useMemo, useState } from 'react';
import axios from 'axios';

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
  other: OutputFiles;
};

type ProcessStartResponse = {
  success: boolean;
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  message: string;
};

type ProcessStatusResponse = {
  success: boolean;
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  message: string;
  files?: OutputFiles;
  error?: string;
  warnings?: string[];
};

function groupOutputFiles(files: OutputFiles): GroupedOutputs {
  const grouped: GroupedOutputs = {
    vis: {},
    unet: {},
    nnunet: {},
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
    <section className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-lg shadow-slate-200/40 backdrop-blur">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className={`text-lg font-semibold ${accentClassName}`}>{title}</h3>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
          {images.length + htmlFiles.length} file{images.length + htmlFiles.length === 1 ? '' : 's'}
        </span>
      </div>

      {images.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {images.map(([filename, file]) => (
            <div key={filename} className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
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
                <p className="truncate text-sm font-medium text-slate-700">{filename}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {htmlFiles.length > 0 && (
        <div className="mt-4 grid gap-4">
          {htmlFiles.map(([filename, file]) => (
            <div key={filename} className="overflow-hidden rounded-xl border border-slate-300 bg-white">
              <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
                <p className="truncate text-sm font-medium text-slate-700">{filename}</p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setOpenHtmlPreviews((prev) => ({
                        ...prev,
                        [filename]: !prev[filename],
                      }))
                    }
                    className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-semibold text-cyan-100 hover:bg-slate-700"
                  >
                    {openHtmlPreviews[filename] ? 'Hide 3D' : 'Show 3D'}
                  </button>
                  <button
                    type="button"
                    onClick={() => openArtifactInNewTab(file)}
                    className="text-xs font-semibold text-cyan-700 hover:text-cyan-900"
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
                <div className="flex h-40 items-center justify-center bg-slate-50 text-sm text-slate-600">
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
    <section className="rounded-2xl border border-cyan-200 bg-white/90 p-5 shadow-lg shadow-cyan-100/40 backdrop-blur">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-cyan-800">Visualization Output</h3>
        {volumeIds.length > 0 && (
          <label className="flex items-center gap-2 text-sm font-semibold text-slate-700">
            Show TIFF:
            <select
              value={selectedVolume}
              onChange={(event) => setSelectedVolume(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm"
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

      const startRes = await axios.post<ProcessStartResponse>('http://localhost:2632/api/scripts/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 0,
      });

      const startedJobId = startRes.data.jobId;
      setJobId(startedJobId);
      setJobStage(startRes.data.stage);
      setJobMessage(startRes.data.message);

      let finished = false;
      while (!finished) {
        await new Promise((resolve) => setTimeout(resolve, 2500));
        const statusRes = await axios.get<ProcessStatusResponse>(`http://localhost:2632/api/scripts/process/${startedJobId}`, {
          timeout: 0,
        });

        setJobStage(statusRes.data.stage);
        setJobMessage(statusRes.data.message);

        if (statusRes.data.status === 'completed') {
          setResponse(statusRes.data);
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
    <main className="relative min-h-screen overflow-hidden bg-[radial-gradient(circle_at_8%_0%,#ecfeff_0%,#ecfeff_25%,transparent_70%),radial-gradient(circle_at_100%_12%,#cffafe_0%,transparent_45%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_100%)] px-4 py-8 md:px-8">
      <div className="pointer-events-none absolute inset-0 opacity-40 [background-image:linear-gradient(to_right,#0f172a10_1px,transparent_1px),linear-gradient(to_bottom,#0f172a10_1px,transparent_1px)] [background-size:40px_40px]" />

      <div className="relative mx-auto max-w-7xl space-y-6">
        <header className="rounded-3xl border border-slate-200 bg-white/80 p-6 shadow-xl shadow-slate-200/50 backdrop-blur">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">Vesuvius Pipeline</p>
          <h1 className="mt-2 text-4xl font-black tracking-tight text-slate-900 md:text-5xl">CV Project</h1>
          <p className="mt-3 max-w-3xl text-slate-600">
            Upload one ZIP file or multiple TIFF files. The backend runs visualization and model inference as a long-running background job.
          </p>
        </header>

        <section className="rounded-3xl border border-slate-200 bg-white/85 p-6 shadow-lg shadow-slate-200/60 backdrop-blur">
          <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
            <div>
              <label className="mb-2 block text-sm font-semibold text-slate-700">Input Files</label>
              <input
                type="file"
                multiple
                onChange={handleFileChange}
                accept=".tif,.tiff,.zip"
                className="block w-full cursor-pointer rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-3 text-sm text-slate-700"
              />
              {files.length > 0 && (
                <div className="mt-3 grid max-h-36 gap-1 overflow-auto rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-700">
                  {files.map((file) => (
                    <p key={file.name}> {file.name}</p>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={loading || !files.length}
              className="h-12 min-w-56 rounded-xl bg-slate-900 px-5 text-sm font-semibold text-cyan-100 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {loading ? 'Running Pipeline...' : 'Upload & Process'}
            </button>
          </div>

          {loading && (
            <div className="mt-4 rounded-xl border border-cyan-200 bg-cyan-50 p-4 text-cyan-900">
              <p className="font-semibold">Processing job is running</p>
              {jobId && <p className="mt-1 text-sm">Job ID: {jobId}</p>}
              {jobStage && <p className="mt-1 text-sm">Stage: {jobStage}</p>}
              {jobMessage && <p className="mt-1 text-sm">{jobMessage}</p>}
              <p className="mt-2 text-sm">Large TIFF sets can take time. You can keep the tab open while status updates continue.</p>
            </div>
          )}

          {error && <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-700">{error}</div>}

          {response?.warnings && response.warnings.length > 0 && (
            <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
              <p className="font-semibold">Completed with warnings</p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
                {response.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {response && (
          <section className="rounded-3xl border border-emerald-200 bg-emerald-50/80 p-6 shadow-lg shadow-emerald-100/70">
            <h2 className="text-xl font-bold text-emerald-800">Processing Complete</h2>
            <p className="mt-1 text-emerald-700">{response.message}</p>
          </section>
        )}

        {groupedOutputs && (
          <div className="space-y-6">
            <VisualizationSection files={groupedOutputs.vis} onImageClick={setSelectedImage} />
            <OutputGroup title="UNet Output" files={groupedOutputs.unet} accentClassName="text-emerald-700" onImageClick={setSelectedImage} />
            <OutputGroup title="nnUNet Output" files={groupedOutputs.nnunet} accentClassName="text-violet-700" onImageClick={setSelectedImage} />
            <OutputGroup title="Other Output" files={groupedOutputs.other} accentClassName="text-slate-700" onImageClick={setSelectedImage} />
          </div>
        )}
      </div>

      {selectedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-h-[95vh] max-w-[95vw]" onClick={(event) => event.stopPropagation()}>
            <button
              type="button"
              onClick={() => setSelectedImage(null)}
              className="absolute -right-3 -top-3 rounded-full bg-white px-3 py-1 text-sm font-bold text-slate-900 shadow"
            >
              X
            </button>
            <img
              src={selectedImage.src}
              alt={selectedImage.name}
              className="max-h-[85vh] max-w-[92vw] rounded-xl border border-slate-300 object-contain shadow-2xl"
            />
            <p className="mt-2 text-center text-sm text-slate-200">{selectedImage.name}</p>
          </div>
        </div>
      )}
    </main>
  );
}
