#!/usr/bin/env node
/**
 * Render a Mermaid .mmd file to PNG/SVG for math-modeling papers.
 *
 * Usage:
 *   node scripts/render_mermaid.js diagrams/problem1_flow.mmd --output images/problem1_flow.png
 *   node scripts/render_mermaid.js diagrams/problem1_flow.mmd -o images/problem1_flow.svg
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

function parseArgs(argv) {
  const out = { input: null, output: null, background: 'white', scale: '2' };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if ((arg === '--output' || arg === '-o') && argv[i + 1]) {
      out.output = argv[++i];
    } else if (arg === '--background' && argv[i + 1]) {
      out.background = argv[++i];
    } else if (arg === '--scale' && argv[i + 1]) {
      out.scale = argv[++i];
    } else if (!arg.startsWith('-') && !out.input) {
      out.input = arg;
    } else {
      throw new Error(`Unknown or incomplete argument: ${arg}`);
    }
  }
  return out;
}

function usageAndExit(message) {
  if (message) console.error(`Error: ${message}`);
  console.error('Usage: node scripts/render_mermaid.js input.mmd --output output.png');
  process.exit(1);
}

function renderMermaid(input, output, opts) {
  const mmdcArgs = [
    '-i', input,
    '-o', output,
    '--backgroundColor', opts.background,
    '--scale', opts.scale,
  ];

  const localMmdc = path.resolve(__dirname, '..', 'node_modules', '.bin', process.platform === 'win32' ? 'mmdc.cmd' : 'mmdc');
  const command = fs.existsSync(localMmdc) ? localMmdc : 'npx';
  const args = fs.existsSync(localMmdc) ? mmdcArgs : ['--yes', 'mmdc', ...mmdcArgs];

  const result = spawnSync(command, args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  if (result.status !== 0) {
    const stderr = (result.stderr || '').trim();
    const stdout = (result.stdout || '').trim();
    throw new Error(`Mermaid render failed.\n${stderr || stdout}`);
  }
}

function main() {
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (err) {
    usageAndExit(err.message);
  }

  if (!args.input) usageAndExit('input .mmd file is required');
  if (!args.output) usageAndExit('output file is required');

  const input = path.resolve(args.input);
  const output = path.resolve(args.output);

  if (!fs.existsSync(input)) usageAndExit(`input not found: ${input}`);
  if (!/\.mmd$/i.test(input)) usageAndExit('input must be a .mmd file');

  const ext = path.extname(output).toLowerCase();
  if (!['.png', '.svg', '.pdf'].includes(ext)) {
    usageAndExit('output extension must be .png, .svg, or .pdf');
  }

  fs.mkdirSync(path.dirname(output), { recursive: true });
  renderMermaid(input, output, args);

  const stat = fs.statSync(output);
  if (stat.size === 0) usageAndExit(`rendered file is empty: ${output}`);

  console.log(`Rendered Mermaid diagram: ${output}`);
}

main();
