import { Router, Request, Response } from 'express';
import { spawn } from 'child_process';
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
const kaggleSegmentationScript = path.join(baseDir, 'src', 'Scripts', 'kaggle_segmentation_inference.py');
const inkDetectionScript = path.join(baseDir, 'src', 'Scripts', 'ink_detection_inference.py');
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
  logLines?: string[];
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

function resolveBooleanEnv(...keys: string[]): boolean {
  const envFile = parseEnvFile(envFilePath);

  let rawValue: string | undefined;
  for (const key of keys) {
    rawValue = process.env[key] ?? envFile[key];
    if (rawValue !== undefined) {
      break;
    }
  }

  if (!rawValue) {
    return false;
  }

  return ['1', 'true', 'yes', 'on'].includes(rawValue.trim().toLowerCase());
}

function resolveNumberEnv(defaultValue: number, ...keys: string[]): number {
  const envFile = parseEnvFile(envFilePath);

  for (const key of keys) {
    const rawValue = process.env[key] ?? envFile[key];
    if (rawValue === undefined) {
      continue;
    }

    const parsed = Number(rawValue);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return defaultValue;
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

function appendJobLog(jobId: string, line: string): void {
  const existing = jobs.get(jobId);
  if (!existing) {
    return;
  }

  const normalized = line.trim();
  if (!normalized) {
    return;
  }

  const nextLines = [...(existing.logLines ?? []), normalized];
  const maxLines = 200;
  const logLines = nextLines.length > maxLines ? nextLines.slice(nextLines.length - maxLines) : nextLines;

  jobs.set(jobId, {
    ...existing,
    logLines,
    message: normalized,
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
  if (ext === '.txt') return 'text/plain';
  return 'application/octet-stream';
}

function collectOutputArtifacts(directoryPath: string, warnings: string[]): Record<string, OutputArtifact> {
  const outputFiles = collectFilesRecursively(
    directoryPath,
    new Set(['.png', '.jpg', '.jpeg', '.html', '.csv', '.json', '.txt']),
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

    if (extension === '.csv' || extension === '.json' || extension === '.txt') {
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

function runPythonScript(
  scriptPath: string,
  args: string[],
  logPrefix: string,
  onLogLine?: (line: string) => void,
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonExecutable,
      [scriptPath, ...args],
      {
        cwd: baseDir,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
          PYTHONUTF8: '1',
          PYTHONUNBUFFERED: '1',
          PYTORCH_CUDA_ALLOC_CONF: process.env.PYTORCH_CUDA_ALLOC_CONF ?? 'expandable_segments:True',
        },
        windowsHide: true,
      },
    );

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    const timeoutHandle = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, SCRIPT_TIMEOUT_MS);

    console.log(`[${logPrefix}] START ${pythonExecutable} ${scriptPath} ${args.join(' ')}`);
    if (onLogLine) {
      onLogLine(`[${logPrefix}] START ${pythonExecutable} ${scriptPath} ${args.join(' ')}`);
    }

    child.stdout.on('data', (chunk: Buffer) => {
      const text = chunk.toString('utf8');
      stdout += text;
      process.stdout.write(`[${logPrefix}] ${text}`);
      if (onLogLine) {
        text.split(/\r?\n/).forEach((line) => {
          if (line.trim()) {
            onLogLine(`[${logPrefix}] ${line}`);
          }
        });
      }
    });

    child.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString('utf8');
      stderr += text;
      process.stderr.write(`[${logPrefix}] ${text}`);
      if (onLogLine) {
        text.split(/\r?\n/).forEach((line) => {
          if (line.trim()) {
            onLogLine(`[${logPrefix}] ${line}`);
          }
        });
      }
    });

    child.on('error', (error) => {
      clearTimeout(timeoutHandle);
      if (onLogLine) {
        onLogLine(`[${logPrefix}] ERROR ${error.message}`);
      }
      reject(new Error(`${logPrefix} launch error: ${error.message}`));
    });

    child.on('close', (code) => {
      clearTimeout(timeoutHandle);

      if (timedOut) {
        if (onLogLine) {
          onLogLine(`[${logPrefix}] TIMEOUT after ${SCRIPT_TIMEOUT_MS} ms`);
        }
        reject(new Error(`${logPrefix} timed out after ${SCRIPT_TIMEOUT_MS} ms`));
        return;
      }

      if (code !== 0) {
        if (onLogLine) {
          onLogLine(`[${logPrefix}] FAILED exit=${code}`);
        }
        const details = [
          `${logPrefix} failed with exit code ${code}`,
          stderr.trim(),
        ].filter(Boolean).join('\n');
        reject(new Error(details || 'Python script execution failed'));
        return;
      }

      console.log(`[${logPrefix}] DONE`);
      if (onLogLine) {
        onLogLine(`[${logPrefix}] DONE`);
      }
      resolve({ stdout, stderr });
    });
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
  const kaggleSegOutputsDir = path.join(jobOutputsDir, 'kaggle_seg');
  const textRecoveryOutputsDir = path.join(jobOutputsDir, 'text_recovery');
  const segOutputInputDir = path.join(jobOutputsDir, 'segmentation_input');
  const warnings: string[] = [];
  const labelsDir = resolveLabelsDir();
  const useDenoise = resolveBooleanEnv('vis_denoise', 'VIS_DENOISE');
  const enableKaggleSeg = resolveBooleanEnv('enable_kaggle_seg', 'ENABLE_KAGGLE_SEG');
  const enableInkDetection = resolveBooleanEnv('enable_ink_detection', 'ENABLE_INK_DETECTION');
  const inferenceFastMode = resolveBooleanEnv('inference_fast_mode', 'INFERENCE_FAST_MODE');
  const finalMaxModels = resolveNumberEnv(0, 'inference_final_max_models', 'INFERENCE_FINAL_MAX_MODELS');
  const nnunetTileStep = resolveNumberEnv(0.5, 'inference_nnunet_tile_step', 'INFERENCE_NNUNET_TILE_STEP');
  const nnunetNoMirroring = resolveBooleanEnv('inference_nnunet_no_mirroring', 'INFERENCE_NNUNET_NO_MIRRORING');
  const skipLegacyNnUnetByEnv = resolveBooleanEnv('inference_skip_legacy_nnunet', 'INFERENCE_SKIP_LEGACY_NNUNET');
  const skipUnetByEnv = resolveBooleanEnv('inference_skip_unet', 'INFERENCE_SKIP_UNET');
  const skipFinalByEnv = resolveBooleanEnv('inference_skip_final_model', 'INFERENCE_SKIP_FINAL_MODEL');

  try {
    updateJob(jobId, { status: 'running', stage: 'preparing', message: 'Preparing files...' });
    ensureDirectory(jobUploadsDir);
    ensureDirectory(jobOutputsDir);
    ensureDirectory(visOutputsDir);
    ensureDirectory(kaggleSegOutputsDir);
    ensureDirectory(textRecoveryOutputsDir);
    ensureDirectory(segOutputInputDir);
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

      await runPythonScript(
        visualizationScript,
        visualizationArgs,
        `job:${jobId}:visualization`,
        (line) => appendJobLog(jobId, line),
      );
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
    const modelInferenceArgs = [
      '--input-dir', jobUploadsDir,
      '--output-dir', jobOutputsDir,
      '--project-dir', cvProjectDir,
      '--checkpoints-dir', cvCheckpointsDir,
    ];

    const legacyNnUnetCkptPaths = [
      path.join(cvProjectDir, 'nnUNet_data', 'nnUNet_results', 'Dataset200_VesuviusSurface', 'nnUNetTrainer__nnUNetPlans__3d_fullres', 'fold_all', 'checkpoint_best.pth'),
      path.join(cvProjectDir, 'nnUNet_data', 'nnUNet_results', 'Dataset200_VesuviusSurface', 'nnUNetTrainer_4000epochs__nnUNetResEncUNetMPlans__3d_fullres', 'fold_all', 'checkpoint_best.pth'),
      path.join(cvProjectDir, 'nnUNet_data', 'nnUNet_results', 'Dataset200_VesuviusSurface', 'nnUNetTrainer_4000epochs__nnUNetResEncUNetLPlans__3d_fullres', 'fold_all', 'checkpoint_best.pth'),
    ];
    const hasLegacyNnUnetCkpt = legacyNnUnetCkptPaths.some((p) => fs.existsSync(p));
    if (skipLegacyNnUnetByEnv || !hasLegacyNnUnetCkpt) {
      modelInferenceArgs.push('--skip-nnunet');
      appendJobLog(
        jobId,
        `[job:${jobId}:model_inference] Legacy nnU-Net skipped (${skipLegacyNnUnetByEnv ? 'env flag' : 'checkpoints not found in nnUNet_data'}).`,
      );
    }

    if (skipUnetByEnv) {
      modelInferenceArgs.push('--skip-unet');
      appendJobLog(jobId, `[job:${jobId}:model_inference] Custom 3D U-Net skipped (env flag).`);
    }

    if (skipFinalByEnv) {
      modelInferenceArgs.push('--skip-final-model');
      appendJobLog(jobId, `[job:${jobId}:model_inference] Final nnU-Net model(s) skipped (env flag).`);
    }

    if (inferenceFastMode) {
      const fastFinalModels = finalMaxModels > 0 ? finalMaxModels : 2;
      modelInferenceArgs.push('--final-max-models', String(fastFinalModels));
      modelInferenceArgs.push('--nnunet-tile-step-size', String(nnunetTileStep > 0 ? nnunetTileStep : 0.7));
      modelInferenceArgs.push('--nnunet-no-mirroring');
    } else {
      if (finalMaxModels > 0) {
        modelInferenceArgs.push('--final-max-models', String(finalMaxModels));
      }
      if (nnunetTileStep > 0) {
        modelInferenceArgs.push('--nnunet-tile-step-size', String(nnunetTileStep));
      }
      if (nnunetNoMirroring) {
        modelInferenceArgs.push('--nnunet-no-mirroring');
      }
    }

    await runPythonScript(modelInferenceScript, modelInferenceArgs, `job:${jobId}:model_inference`, (line) => appendJobLog(jobId, line));
    const modelEnd = Date.now();

    const segmentationMasks = collectFilesRecursively(jobOutputsDir, new Set(['.tif', '.tiff']));
    for (const segmentationMask of segmentationMasks) {
      const outputName = path.basename(segmentationMask);
      const targetPath = path.join(segOutputInputDir, outputName);

      if (!fs.existsSync(targetPath)) {
        fs.copyFileSync(segmentationMask, targetPath);
      }
    }

    if (enableKaggleSeg) {
      updateJob(jobId, {
        stage: 'kaggle_segmentation',
        message: 'Running Kaggle 1st-place segmentation baseline...',
      });

      const kaggleStart = Date.now();
      try {
        await runPythonScript(kaggleSegmentationScript, [
          '--input-dir', jobUploadsDir,
          '--output-dir', kaggleSegOutputsDir,
        ], `job:${jobId}:kaggle_segmentation`, (line) => appendJobLog(jobId, line));
      } catch (error: any) {
        warnings.push(`Kaggle segmentation baseline failed: ${error?.message || 'Unknown error'}`);
      }
      const kaggleEnd = Date.now();

      updateJob(jobId, {
        timings: { ...(jobs.get(jobId)?.timings || {}), kaggleSegmentation: kaggleEnd - kaggleStart },
      });
    }

    if (enableInkDetection) {
      updateJob(jobId, {
        stage: 'ink_detection',
        message: 'Running TimeSformer ink detection...',
      });

      const inkStart = Date.now();
      try {
        await runPythonScript(inkDetectionScript, [
          '--input-dir', jobUploadsDir,
          '--segmentation-dir', segOutputInputDir,
          '--output-dir', textRecoveryOutputsDir,
        ], `job:${jobId}:ink_detection`, (line) => appendJobLog(jobId, line));
      } catch (error: any) {
        warnings.push(`TimeSformer ink detection failed: ${error?.message || 'Unknown error'}`);
      }
      const inkEnd = Date.now();

      updateJob(jobId, {
        timings: { ...(jobs.get(jobId)?.timings || {}), inkDetection: inkEnd - inkStart },
      });
    }

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
    logLines: job.logLines,
    files: job.status === 'completed' ? job.files : undefined,
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
    const { stdout, stderr } = await runPythonScript(script, [], 'manual:run');
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
      { name: 'kaggle_segmentation_inference.py', description: 'Run Kaggle 1st-place segmentation baseline' },
      { name: 'ink_detection_inference.py', description: 'Run TimeSformer ink detection + text enhancement' },
    ],
  });
});

export default router;