import { Router, Request, Response } from 'express';
import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import unzipper from 'unzipper';
import multer from 'multer';
import { randomUUID } from 'crypto';

const router = Router();

const baseDir = path.join(__dirname, '..', '..');
const tempDir = path.join(baseDir, 'temp');
const uploadsDir = path.join(tempDir, 'uploads');
const outputsDir = path.join(tempDir, 'outputs');
const jobsDir = path.join(tempDir, 'jobs');
const envFilePath = path.join(baseDir, '.env');
const fallbackPythonExecutable = path.join(baseDir, '..', '..', '.venv', 'Scripts', 'python.exe');
const visualizationScript = path.join(baseDir, 'src', 'Scripts', 'vesuvius_visualize.py');
const modelInferenceScript = path.join(baseDir, 'src', 'Scripts', 'model_inference.py');
const cvProjectDir = path.join(baseDir, '..', 'CV_project');
const cvCheckpointsDir = path.join(cvProjectDir, 'checkpoints');
const SCRIPT_TIMEOUT_MS = 45 * 60 * 1000;
const JOB_RETENTION_MS = 30 * 60 * 1000;
const MAX_INLINE_HTML_BYTES = Number(process.env.max_inline_html_bytes ?? 4 * 1024 * 1024);

type JobState = 'queued' | 'running' | 'completed' | 'failed';

type OutputArtifact = {
  mimeType: string;
  encoding: 'base64' | 'utf8';
  data: string;
};

type ProcessingJob = {
  jobId: string;
  status: JobState;
  stage: string;
  message: string;
  createdAt: number;
  updatedAt: number;
  files?: Record<string, OutputArtifact>;
  error?: string;
  warnings?: string[];
  timings?: Record<string, number>;
};

const jobs = new Map<string, ProcessingJob>();

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, callback) => {
      fs.mkdirSync(uploadsDir, { recursive: true });
      callback(null, uploadsDir);
    },
    filename: (_req, file, callback) => {
      const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
      callback(null, `${uniqueSuffix}-${file.originalname}`);
    },
  }),
});

fs.mkdirSync(uploadsDir, { recursive: true });
fs.mkdirSync(outputsDir, { recursive: true });
fs.mkdirSync(jobsDir, { recursive: true });

setInterval(() => {
  const now = Date.now();

  for (const [jobId, job] of jobs.entries()) {
    if ((job.status === 'completed' || job.status === 'failed') && now - job.updatedAt > JOB_RETENTION_MS) {
      jobs.delete(jobId);
    }
  }
}, 5 * 60 * 1000).unref();

function parseEnvFile(filePath: string): Record<string, string> {
  if (!fs.existsSync(filePath)) {
    return {};
  }

  const parsedEnv: Record<string, string> = {};
  const fileContents = fs.readFileSync(filePath, 'utf8');

  for (const rawLine of fileContents.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line || line.startsWith('#')) {
      continue;
    }

    const separatorIndex = line.indexOf('=');

    if (separatorIndex === -1) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    let value = line.slice(separatorIndex + 1).trim();

    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }

    if (key) {
      parsedEnv[key] = value;
    }
  }

  return parsedEnv;
}

function resolvePythonExecutable(): string {
  const envFile = parseEnvFile(envFilePath);
  const configuredPython = process.env.Python_ENV ?? envFile.Python_ENV;

  if (configuredPython) {
    return path.isAbsolute(configuredPython) ? configuredPython : path.resolve(baseDir, configuredPython);
  }

  return fallbackPythonExecutable;
}

const pythonExecutable = resolvePythonExecutable();

function resolveConfiguredPath(candidatePath: string | undefined): string | null {
  if (!candidatePath) {
    return null;
  }

  const resolvedPath = path.isAbsolute(candidatePath) ? candidatePath : path.resolve(baseDir, candidatePath);
  return fs.existsSync(resolvedPath) ? resolvedPath : null;
}

function resolveLabelsDir(): string | null {
  const envFile = parseEnvFile(envFilePath);
  const envCandidates = [
    process.env.label_path,
    process.env.labels_dir,
    envFile.label_path,
    envFile.labels_dir,
  ];

  for (const candidate of envCandidates) {
    const resolvedPath = resolveConfiguredPath(candidate);
    if (resolvedPath) {
      return resolvedPath;
    }
  }

  return null;
}

function resolveBooleanEnv(key: string): boolean {
  const envFile = parseEnvFile(envFilePath);
  const rawValue = process.env[key] ?? envFile[key];

  if (!rawValue) {
    return false;
  }

  return ['1', 'true', 'yes', 'on'].includes(rawValue.trim().toLowerCase());
}

function createJob(): ProcessingJob {
  const now = Date.now();
  return {
    jobId: randomUUID(),
    status: 'queued',
    stage: 'queued',
    message: 'Job queued',
    createdAt: now,
    updatedAt: now,
  };
}

function updateJob(jobId: string, updates: Partial<ProcessingJob>): void {
  const existing = jobs.get(jobId);

  if (!existing) {
    return;
  }

  jobs.set(jobId, {
    ...existing,
    ...updates,
    updatedAt: Date.now(),
  });
}

// Helper function to extract ZIP file
async function extractZip(zipPath: string, extractPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    fs.createReadStream(zipPath)
      .pipe(unzipper.Extract({ path: extractPath }))
      .on('close', resolve)
      .on('error', reject);
  });
}

// Helper function to cleanup directories
async function cleanupDirs(dirs: string[]): Promise<void> {
  for (const dir of dirs) {
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
}

function ensureDirectory(directoryPath: string): void {
  if (!fs.existsSync(directoryPath)) {
    fs.mkdirSync(directoryPath, { recursive: true });
  }
}

function collectFilesRecursively(directoryPath: string, extensions: Set<string>): string[] {
  const collectedFiles: string[] = [];

  if (!fs.existsSync(directoryPath)) {
    return collectedFiles;
  }

  const walk = (currentPath: string): void => {
    for (const entry of fs.readdirSync(currentPath, { withFileTypes: true })) {
      const entryPath = path.join(currentPath, entry.name);

      if (entry.isDirectory()) {
        walk(entryPath);
        continue;
      }

      if (extensions.has(path.extname(entry.name).toLowerCase())) {
        collectedFiles.push(entryPath);
      }
    }
  };

  walk(directoryPath);
  return collectedFiles;
}

function inferMimeType(extension: string): string {
  const ext = extension.toLowerCase();

  if (ext === '.png') return 'image/png';
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
  if (ext === '.tif' || ext === '.tiff') return 'image/tiff';
  if (ext === '.html') return 'text/html';
  if (ext === '.csv') return 'text/csv';
  if (ext === '.json') return 'application/json';
  return 'application/octet-stream';
}

function collectOutputArtifacts(directoryPath: string, warnings: string[]): Record<string, OutputArtifact> {
  const outputFiles = collectFilesRecursively(
    directoryPath,
    new Set(['.png', '.jpg', '.jpeg', '.html', '.csv', '.json']),
  );
  const fileContents: Record<string, OutputArtifact> = {};

  for (const filePath of outputFiles) {
    const relativePath = path.relative(directoryPath, filePath).replace(/\\/g, '/');
    const extension = path.extname(filePath).toLowerCase();
    const mimeType = inferMimeType(extension);

    if (extension === '.html') {
      const fileSize = fs.statSync(filePath).size;
      if (fileSize > MAX_INLINE_HTML_BYTES) {
        warnings.push(
          `Skipped large 3D HTML (${relativePath}, ${(fileSize / (1024 * 1024)).toFixed(1)} MB) ` +
          `to prevent memory crashes. Generate with smaller sub-volume size to view in app.`,
        );
        continue;
      }

      fileContents[relativePath] = {
        mimeType,
        encoding: 'utf8',
        data: fs.readFileSync(filePath, 'utf8'),
      };
      continue;
    }

    if (extension === '.csv' || extension === '.json') {
      fileContents[relativePath] = {
        mimeType,
        encoding: 'utf8',
        data: fs.readFileSync(filePath, 'utf8'),
      };
      continue;
    }

    const fileData = fs.readFileSync(filePath);
    fileContents[relativePath] = {
      mimeType,
      encoding: 'base64',
      data: fileData.toString('base64'),
    };
  }

  return fileContents;
}

function runPythonScript(scriptPath: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(
      pythonExecutable,
      [scriptPath, ...args],
      {
        cwd: baseDir,
        maxBuffer: 50 * 1024 * 1024,
        timeout: SCRIPT_TIMEOUT_MS,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
        },
      },
      (error, stdout, stderr) => {
        if (error) {
          const details = [error.message, stderr].filter(Boolean).join('\n');
          reject(new Error(details || 'Python script execution failed'));
          return;
        }

        resolve({ stdout, stderr });
      },
    );
  });
}

function removeTempFile(filePath: string): void {
  if (fs.existsSync(filePath)) {
    fs.rmSync(filePath, { force: true });
  }
}

function copyUploadedFiles(uploadedFiles: Express.Multer.File[], jobUploadsDir: string): void {
  for (const file of uploadedFiles) {
    const fileExt = path.extname(file.originalname).toLowerCase();

    if (fileExt === '.zip') {
      const extractDir = path.join(jobUploadsDir, path.parse(file.originalname).name);
      ensureDirectory(extractDir);
    }
  }
}

async function processJob(jobId: string, uploadedFiles: Express.Multer.File[]): Promise<void> {
  const jobBaseDir = path.join(jobsDir, jobId);
  const jobUploadsDir = path.join(jobBaseDir, 'uploads');
  const jobOutputsDir = path.join(jobBaseDir, 'outputs');
  const visOutputsDir = path.join(jobOutputsDir, 'vis');
  const warnings: string[] = [];
  const labelsDir = resolveLabelsDir();
  const useDenoise = resolveBooleanEnv('vis_denoise');

  try {
    updateJob(jobId, { status: 'running', stage: 'preparing', message: 'Preparing files...' });
    ensureDirectory(jobUploadsDir);
    ensureDirectory(jobOutputsDir);
    ensureDirectory(visOutputsDir);
    copyUploadedFiles(uploadedFiles, jobUploadsDir);

    updateJob(jobId, { stage: 'extracting', message: 'Extracting uploads...' });
    for (const file of uploadedFiles) {
      const fileExt = path.extname(file.originalname).toLowerCase();

      if (fileExt === '.zip') {
        const extractDir = path.join(jobUploadsDir, path.parse(file.originalname).name);
        ensureDirectory(extractDir);
        await extractZip(file.path, extractDir);
      } else {
        const targetPath = path.join(jobUploadsDir, file.originalname);
        fs.copyFileSync(file.path, targetPath);
      }

      removeTempFile(file.path);
    }

    updateJob(jobId, { stage: 'visualization', message: 'Running vesuvius visualization...' });
    const visStart = Date.now();
    try {
      const visualizationArgs = ['--input_dir', jobUploadsDir, '--output_dir', visOutputsDir];

      if (labelsDir) {
        visualizationArgs.push('--labels_dir', labelsDir);
      }

      if (useDenoise) {
        visualizationArgs.push('--denoise');
      }

      await runPythonScript(visualizationScript, visualizationArgs);
    } catch (error: any) {
      warnings.push(`Visualization step failed: ${error?.message || 'Unknown visualization error'}`);
      updateJob(jobId, { stage: 'visualization_warning', message: 'Visualization failed, continuing with model inference...' });
    }
    const visEnd = Date.now();

    // Collect early visualization results and push them to the job
    const earlyVisFiles = collectOutputArtifacts(jobOutputsDir, warnings);
    updateJob(jobId, { 
      stage: 'model_inference', 
      message: 'Running model inference...',
      files: earlyVisFiles, // Publish partial files early to frontend
      timings: { ...(jobs.get(jobId)?.timings || {}), visualization: visEnd - visStart }
    });
    
    const modelStart = Date.now();
    await runPythonScript(modelInferenceScript, [
      '--input-dir', jobUploadsDir,
      '--output-dir', jobOutputsDir,
      '--project-dir', cvProjectDir,
      '--checkpoints-dir', cvCheckpointsDir,
    ]);
    const modelEnd = Date.now();

    updateJob(jobId, { 
      stage: 'collecting', 
      message: 'Collecting output files...',
      timings: { ...(jobs.get(jobId)?.timings || {}), modelInference: modelEnd - modelStart }
    });
    const fileContents = collectOutputArtifacts(jobOutputsDir, warnings);

    await cleanupDirs([jobBaseDir]);

    updateJob(jobId, {
      status: 'completed',
      stage: 'completed',
      message: warnings.length ? 'Processing completed with warnings' : 'Processing completed successfully',
      files: fileContents,
      warnings,
    });
  } catch (error: any) {
    uploadedFiles.forEach((file) => removeTempFile(file.path));
    await cleanupDirs([jobBaseDir]);

    updateJob(jobId, {
      status: 'failed',
      stage: 'failed',
      message: 'Processing failed',
      error: error?.message || 'Unknown processing error',
    });
  }
}

// Process images - handles TIF files or ZIP
router.post('/process', upload.any(), async (req: Request, res: Response) => {
  const uploadedFiles = (req.files as Express.Multer.File[] | undefined) ?? [];

  try {
    if (!uploadedFiles.length) {
      return res.status(400).json({ error: 'Please upload at least one TIF file or one ZIP file' });
    }

    ensureDirectory(uploadsDir);
    ensureDirectory(outputsDir);

    const uploadedExtensions = uploadedFiles.map((file) => path.extname(file.originalname).toLowerCase());
    const hasZip = uploadedExtensions.includes('.zip');
    const hasTif = uploadedExtensions.some((extension) => extension === '.tif' || extension === '.tiff');

    if (hasZip && uploadedFiles.length > 1) {
      uploadedFiles.forEach((file) => removeTempFile(file.path));
      return res.status(400).json({ error: 'Upload either one ZIP file or multiple TIF/TIFF files, not both' });
    }

    if (!hasZip && !hasTif) {
      uploadedFiles.forEach((file) => removeTempFile(file.path));
      return res.status(400).json({ error: 'Only .zip or .tif/.tiff files are supported' });
    }

    const job = createJob();
    jobs.set(job.jobId, job);

    void processJob(job.jobId, uploadedFiles);

    res.status(202).json({
      success: true,
      jobId: job.jobId,
      status: job.status,
      stage: job.stage,
      message: 'Upload accepted. Processing started.',
    });
  } catch (error: any) {
    uploadedFiles.forEach((file) => removeTempFile(file.path));

    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

router.get('/process/:jobId', (req: Request, res: Response) => {
  const job = jobs.get(req.params.jobId);

  if (!job) {
    return res.status(404).json({
      success: false,
      error: 'Job not found or expired',
    });
  }

  return res.json({
    success: true,
    jobId: job.jobId,
    status: job.status,
    stage: job.stage,
    message: job.message,
    files: job.files,
    error: job.error,
    warnings: job.warnings,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    timings: job.timings,
  });
});

// Run a script
router.post('/run', async (req: Request, res: Response) => {
  const { script } = req.body;

  if (!script) {
    return res.status(400).json({ error: 'Script name is required' });
  }

  try {
    const { stdout, stderr } = await runPythonScript(script, []);
    res.json({
      success: true,
      stdout,
      stderr,
    });
  } catch (error: any) {
    res.status(500).json({
      success: false,
      error: error.message,
      stderr: error.stderr,
    });
  }
});

// Get available scripts
router.get('/list', (req: Request, res: Response) => {
  res.json({
    scripts: [
      { name: 'vesuvius_visualize.py', description: 'Generate visualization outputs' },
      { name: 'model_inference.py', description: 'Generate UNet and nnUNet outputs' },
    ],
  });
});

export default router;