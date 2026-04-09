#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const frontendDir = path.join(__dirname);

const npm = spawn('npm', ['start'], {
  cwd: frontendDir,
  stdio: 'inherit',
  shell: true,
});

npm.on('error', (error) => {
  console.error('Failed to start npm:', error);
  process.exit(1);
});

npm.on('exit', (code) => {
  process.exit(code);
});

process.on('SIGINT', () => {
  npm.kill('SIGINT');
});

process.on('SIGTERM', () => {
  npm.kill('SIGTERM');
});
