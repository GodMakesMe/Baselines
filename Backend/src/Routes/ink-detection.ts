import { Router, Request, Response } from 'express';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

const router = Router();

const baseDir = path.join(__dirname, '..', '..');
const tempDir = path.join(baseDir, 'temp');
const inkJobsDir = path.join(tempDir, 'ink_jobs');
const envFilePath = path.join(baseDir, '.env');
const fallbackPythonExecutable = path.join(baseDir, '..', '..', '.venv', 'Scripts', 'python.exe');
const inkScript = path.join(baseDir, 'src', 'Scripts', 'ink_winner.py');

const JOB_TIMEOUT_MS = 60 * 60 * 1000; // 1h per ink job — fragments can be big
const JOB_RETENTION_MS = 60 * 60 * 1000;

type JobState = 'queued' | 'running' | 'completed' | 'failed';

type OutputArtifact = {
  mimeType: string;
  encoding: 'base64' | 'utf8';
  data: string;
};

type InkJob = {
  jobId: string;
  status: JobState;
  stage: string;
  message: string;
  createdAt: number;
  updatedAt: number;
  logLines?: string[];
  files?: Record<string, OutputArtifact>;
  segments: string[];
  model: string;
  error?: string;
  timings?: Record<string, number>;
};

const jobs = new Map<string, InkJob>();

fs.mkdirSync(inkJobsDir, { recursive: true });

setInterval(() => {
  const now = Date.now();
  for (const [jobId, job] of jobs.entries()) {
    if ((job.status === 'completed' || job.status === 'failed') && now - job.updatedAt > JOB_RETENTION_MS) {
      jobs.delete(jobId);
    }
  }
}, 10 * 60 * 1000).unref();

function parseEnvFile(filePath: string): Record<string, string> {
  if (!fs.existsSync(filePath)) return {};
  const parsed: Record<string, string> = {};
  const text = fs.readFileSync(filePath, 'utf8');
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (key) parsed[key] = value;
  }
  return parsed;
}

function resolvePython(): string {
  const envFile = parseEnvFile(envFilePath);
  const configured = process.env.Python_ENV ?? envFile.Python_ENV;
  if (configured) {
    return path.isAbsolute(configured) ? configured : path.resolve(baseDir, configured);
  }
  return fallbackPythonExecutable;
}

function resolveConfigured(p: string | undefined): string | null {
  if (!p) return null;
  return path.isAbsolute(p) ? p : path.resolve(baseDir, p);
}

function readEnv(...keys: string[]): string | undefined {
  const envFile = parseEnvFile(envFilePath);
  for (const key of keys) {
    const v = process.env[key] ?? envFile[key];
    if (v !== undefined) return v;
  }
  return undefined;
}

const pythonExecutable = resolvePython();

const MODELS: Record<string, { envKey: string; label: string; description: string }> = {
  timesformer_64: {
    envKey: 'INK_CHECKPOINT_TIMESFORMER_64',
    label: 'TimeSformer — tile 64 (default, 4 GB VRAM safe)',
    description: 'Canonical Winner checkpoint trained on segment 20230702185753. Fastest, lowest VRAM.',
  },
  timesformer_256: {
    envKey: 'INK_CHECKPOINT_TIMESFORMER_256',
    label: 'TimeSformer — tile 256 (needs 6 GB+ VRAM)',
    description: 'Deeper wild14-deduped checkpoint with tile size 256. Higher quality but larger memory footprint.',
  },
};

function resolveCheckpoint(modelKey: string): string | null {
  const m = MODELS[modelKey];
  if (!m) return null;
  return resolveConfigured(readEnv(m.envKey));
}

function resolveInkSourceDir(): string | null {
  return resolveConfigured(readEnv('INK_TEST_DIR'));
}

function inferMime(ext: string): string {
  const e = ext.toLowerCase();
  if (e === '.png') return 'image/png';
  if (e === '.jpg' || e === '.jpeg') return 'image/jpeg';
  if (e === '.txt') return 'text/plain';
  if (e === '.json') return 'application/json';
  return 'application/octet-stream';
}

function collectImages(dir: string): Record<string, OutputArtifact> {
  const out: Record<string, OutputArtifact> = {};
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    const ext = path.extname(entry.name).toLowerCase();
    if (!['.png', '.jpg', '.jpeg'].includes(ext)) continue;
    const data = fs.readFileSync(path.join(dir, entry.name));
    out[entry.name] = {
      mimeType: inferMime(ext),
      encoding: 'base64',
      data: data.toString('base64'),
    };
  }
  return out;
}

function updateJob(jobId: string, updates: Partial<InkJob>): void {
  const existing = jobs.get(jobId);
  if (!existing) return;
  jobs.set(jobId, { ...existing, ...updates, updatedAt: Date.now() });
}

function appendLog(jobId: string, line: string): void {
  const existing = jobs.get(jobId);
  if (!existing) return;
  const trimmed = line.trim();
  if (!trimmed) return;
  const nextLines = [...(existing.logLines ?? []), trimmed];
  const logLines = nextLines.length > 300 ? nextLines.slice(nextLines.length - 300) : nextLines;
  jobs.set(jobId, { ...existing, logLines, message: trimmed, updatedAt: Date.now() });
}

function runPython(
  scriptPath: string,
  args: string[],
  logPrefix: string,
  onLine: (line: string) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable, [scriptPath, ...args], {
      cwd: baseDir,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
        PYTHONUNBUFFERED: '1',
        PYTORCH_CUDA_ALLOC_CONF: process.env.PYTORCH_CUDA_ALLOC_CONF ?? 'expandable_segments:True',
        WANDB_MODE: process.env.WANDB_MODE ?? 'disabled',
      },
      windowsHide: true,
    });

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, JOB_TIMEOUT_MS);

    const startLine = `[${logPrefix}] START ${scriptPath} ${args.join(' ')}`;
    console.log(startLine);
    onLine(startLine);

    const pump = (prefix: string, chunk: Buffer, sink: NodeJS.WritableStream) => {
      const text = chunk.toString('utf8');
      sink.write(`[${logPrefix}] ${text}`);
      text.split(/\r?\n/).forEach((line) => {
        if (line.trim()) onLine(`[${logPrefix}] ${line}`);
      });
    };

    child.stdout.on('data', (chunk: Buffer) => pump('stdout', chunk, process.stdout));
    child.stderr.on('data', (chunk: Buffer) => pump('stderr', chunk, process.stderr));

    child.on('error', (error) => {
      clearTimeout(timer);
      reject(new Error(`${logPrefix} launch error: ${error.message}`));
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(new Error(`${logPrefix} timed out after ${JOB_TIMEOUT_MS / 1000}s`));
        return;
      }
      if (code !== 0) {
        reject(new Error(`${logPrefix} exited with code ${code}`));
        return;
      }
      resolve();
    });
  });
}

function listSegments(sourceDir: string): Array<{
  id: string;
  layerCount: number;
  hasMask: boolean;
  dimensionsHint?: { width: number; height: number };
}> {
  if (!fs.existsSync(sourceDir)) return [];
  const result: Array<{ id: string; layerCount: number; hasMask: boolean; dimensionsHint?: { width: number; height: number } }> = [];

  for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const id = entry.name;
    const segDir = path.join(sourceDir, id);
    const layersDir = path.join(segDir, 'surface_volume');
    const maskPath = path.join(segDir, 'mask.png');

    if (!fs.existsSync(layersDir) || !fs.statSync(layersDir).isDirectory()) continue;

    const layers = fs.readdirSync(layersDir).filter((n) => /\.(tif|tiff|jpg|jpeg)$/i.test(n));
    if (!layers.length) continue;

    let dimensionsHint: { width: number; height: number } | undefined;
    try {
      if (fs.existsSync(maskPath)) {
        const stat = fs.statSync(maskPath);
        if (stat.size > 0) {
          // cheap PNG header sniff — IHDR is at offset 16, width/height are big-endian u32
          const fd = fs.openSync(maskPath, 'r');
          const buf = Buffer.alloc(24);
          fs.readSync(fd, buf, 0, 24, 0);
          fs.closeSync(fd);
          if (buf.slice(12, 16).toString('ascii') === 'IHDR') {
            dimensionsHint = { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
          }
        }
      }
    } catch {
      /* ignore dimension probe errors */
    }

    result.push({
      id,
      layerCount: layers.length,
      hasMask: fs.existsSync(maskPath),
      dimensionsHint,
    });
  }

  result.sort((a, b) => a.id.localeCompare(b.id));
  return result;
}

router.get('/health', (_req: Request, res: Response) => {
  const sourceDir = resolveInkSourceDir();
  const models = Object.entries(MODELS).map(([key, m]) => ({
    key,
    label: m.label,
    description: m.description,
    available: !!resolveCheckpoint(key) && fs.existsSync(resolveCheckpoint(key)!),
  }));
  const defaultModel = readEnv('INK_DEFAULT_MODEL') ?? 'timesformer_64';

  res.json({
    ok: true,
    sourceDir,
    sourceDirExists: sourceDir ? fs.existsSync(sourceDir) : false,
    defaultModel,
    models,
  });
});

router.get('/segments', (_req: Request, res: Response) => {
  const sourceDir = resolveInkSourceDir();
  if (!sourceDir) {
    return res.status(500).json({ error: 'INK_TEST_DIR is not configured in .env' });
  }
  if (!fs.existsSync(sourceDir)) {
    return res.status(404).json({ error: `Ink test directory not found: ${sourceDir}` });
  }
  res.json({ sourceDir, segments: listSegments(sourceDir) });
});

async function runInkJob(jobId: string, segmentIds: string[], model: string): Promise<void> {
  const sourceDir = resolveInkSourceDir();
  const checkpoint = resolveCheckpoint(model);

  if (!sourceDir || !fs.existsSync(sourceDir)) {
    updateJob(jobId, { status: 'failed', stage: 'failed', message: 'INK_TEST_DIR not configured or missing', error: 'source_dir_missing' });
    return;
  }
  if (!checkpoint || !fs.existsSync(checkpoint)) {
    updateJob(jobId, { status: 'failed', stage: 'failed', message: `Checkpoint for ${model} not found`, error: 'checkpoint_missing' });
    return;
  }

  const jobDir = path.join(inkJobsDir, jobId);
  const outputDir = path.join(jobDir, 'outputs');
  fs.mkdirSync(outputDir, { recursive: true });

  const args = [
    '--source-dir', sourceDir,
    '--output-dir', outputDir,
    '--checkpoint', checkpoint,
  ];
  for (const sid of segmentIds) {
    args.push('--segment-id', sid);
  }

  updateJob(jobId, { status: 'running', stage: 'inference', message: `Starting ink inference on [${segmentIds.join(', ')}]` });

  const startedAt = Date.now();
  try {
    await runPython(inkScript, args, `ink:${jobId}`, (line) => appendLog(jobId, line));
  } catch (err: any) {
    updateJob(jobId, {
      status: 'failed',
      stage: 'failed',
      message: err?.message ?? 'Ink inference failed',
      error: err?.message,
      timings: { inference: Date.now() - startedAt },
    });
    return;
  }

  const files = collectImages(outputDir);
  updateJob(jobId, {
    status: 'completed',
    stage: 'completed',
    message: 'Ink detection completed',
    files,
    timings: { inference: Date.now() - startedAt },
  });

  // Leave jobDir on disk so the user can inspect it; retention cleanup drops it later.
}

router.post('/process', (req: Request, res: Response) => {
  const body = req.body ?? {};
  const rawSegments = Array.isArray(body.segments) ? body.segments : [];
  const segments = rawSegments.filter((s: unknown): s is string => typeof s === 'string' && s.trim().length > 0);
  const model = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : (readEnv('INK_DEFAULT_MODEL') ?? 'timesformer_64');

  if (!segments.length) {
    return res.status(400).json({ error: 'Provide at least one segment id in "segments"' });
  }
  if (!MODELS[model]) {
    return res.status(400).json({ error: `Unknown model "${model}". Allowed: ${Object.keys(MODELS).join(', ')}` });
  }

  const sourceDir = resolveInkSourceDir();
  if (!sourceDir) {
    return res.status(500).json({ error: 'INK_TEST_DIR is not configured in .env' });
  }
  const invalid = segments.filter((s: string) => !fs.existsSync(path.join(sourceDir, s, 'surface_volume')));
  if (invalid.length) {
    return res.status(400).json({ error: `Unknown or malformed segment(s): ${invalid.join(', ')}` });
  }

  const jobId = randomUUID();
  const now = Date.now();
  jobs.set(jobId, {
    jobId,
    status: 'queued',
    stage: 'queued',
    message: 'Queued',
    createdAt: now,
    updatedAt: now,
    segments,
    model,
  });

  void runInkJob(jobId, segments, model);

  res.status(202).json({ success: true, jobId, segments, model });
});

router.get('/process/:jobId', (req: Request, res: Response) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found or expired' });
  }
  res.json(job);
});

router.get('/jobs', (_req: Request, res: Response) => {
  const list = Array.from(jobs.values())
    .map((j) => ({
      jobId: j.jobId,
      status: j.status,
      stage: j.stage,
      segments: j.segments,
      model: j.model,
      createdAt: j.createdAt,
      updatedAt: j.updatedAt,
    }))
    .sort((a, b) => b.createdAt - a.createdAt);
  res.json({ jobs: list });
});

export default router;
