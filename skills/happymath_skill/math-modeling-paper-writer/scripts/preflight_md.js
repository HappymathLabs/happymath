#!/usr/bin/env node
/**
 * Preflight checks before Markdown -> DOCX conversion.
 *
 * Checks:
 * - no raw mermaid code blocks remain in the markdown
 * - markdown image references exist
 * - obvious placeholder images are not used
 * - display math blocks are balanced
 * - simple numeric citations have matching reference entries
 */

'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
  console.error('Usage: node scripts/preflight_md.js paper.md');
  process.exit(1);
}

function readPngText(filePath) {
  const buf = fs.readFileSync(filePath);
  return buf.toString('latin1');
}

function isLikelyPlaceholderImage(filePath) {
  const base = path.basename(filePath).toLowerCase();
  if (/placeholder|sample|dummy/.test(base)) return true;

  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.svg') {
    const text = fs.readFileSync(filePath, 'utf8').toLowerCase();
    return text.includes('sample flowchart') || text.includes('placeholder');
  }
  if (ext === '.png') {
    const stat = fs.statSync(filePath);
    const dims = readPngDimensions(filePath);
    if (dims && dims.width === 400 && dims.height === 300 && stat.size < 10000) {
      return true;
    }
    const text = readPngText(filePath).toLowerCase();
    return text.includes('sample flowchart') || text.includes('placeholder');
  }
  return false;
}

function readPngDimensions(filePath) {
  const buf = fs.readFileSync(filePath);
  const signature = '89504e470d0a1a0a';
  if (buf.length < 24 || buf.slice(0, 8).toString('hex') !== signature) return null;
  return {
    width: buf.readUInt32BE(16),
    height: buf.readUInt32BE(20),
  };
}

function stripCodeBlocks(md) {
  return md.replace(/```[\s\S]*?```/g, '');
}

function detectLanguage(md) {
  const chineseChars = (md.match(/[\u4e00-\u9fa5]/g) || []).length;
  const total = md.length || 1;
  return (chineseChars / total) > 0.05 ? 'cn' : 'en';
}

function isHeading(line) {
  return /^#{1,6}\s+/.test(line.trim());
}

function stripHeadingText(line) {
  return line.trim().replace(/^#{1,6}\s+/, '').trim();
}

function stripHeadingNumber(text) {
  return text
    .trim()
    .replace(/^第[一二三四五六七八九十百]+[章节部分]\s*[：:、.．-]?\s*/, '')
    .replace(/^[一二三四五六七八九十]+[、.．]\s*/, '')
    .replace(/^\(?\d+(?:\.\d+)*\)?[、.．:：\s-]+/, '')
    .trim();
}

function collectSection(md, titles) {
  const lines = md.split(/\r?\n/);
  const sections = [];
  let current = null;
  let currentLevel = 0;
  for (const line of lines) {
    const match = line.match(/^(#{1,6})\s+(.+)$/);
    if (match) {
      const level = match[1].length;
      const text = stripHeadingNumber(match[2]);
      if (current && level <= currentLevel) {
        sections.push(current);
        current = null;
      }
      if (!current && titles.some(t => text.toLowerCase() === t.toLowerCase())) {
        current = { title: text, level, lines: [] };
        currentLevel = level;
        continue;
      }
    }
    if (current) current.lines.push(line);
  }
  if (current) sections.push(current);
  return sections;
}

function hasSummarySheetTable(md) {
  const firstLines = md.split(/\r?\n/).slice(0, 30).join('\n');
  return /Problem\s+Chosen/i.test(firstLines) &&
    /Team\s+Control\s+Number/i.test(firstLines) &&
    /MCM\/ICM/i.test(firstLines) &&
    /\|/.test(firstLines);
}

function checkAbstractRules(md, errors) {
  const abstractSections = collectSection(md, ['摘要', '摘 要', 'Abstract', 'Summary']);
  for (const section of abstractSections) {
    const text = section.lines.join('\n');
    if (/(^|\n)\s*\$\$\s*(\n|$)/.test(text) || /\$[^$\n]+\$/.test(text)) {
      errors.push(`${section.title} section contains formulas. Abstract/Summary must use words and data only; bold emphasis is allowed.`);
    }
    if (/!\[[^\]]*\]\([^)]+\)/.test(text)) {
      errors.push(`${section.title} section contains an image. Abstract/Summary must not contain figures.`);
    }
    const tableLines = section.lines.filter(line => /\|/.test(line) && !/^\s*\|?\s*[-:| ]+\s*\|?\s*$/.test(line));
    if (tableLines.length) {
      errors.push(`${section.title} section contains a table. Abstract/Summary must not contain tables.`);
    }
    if (/\[\d+\]/.test(stripCodeBlocks(text))) {
      errors.push(`${section.title} section contains citations. Abstract/Summary must not contain references.`);
    }
  }
}

function isProblemAnalysisHeading(text) {
  const stripped = stripHeadingNumber(text);
  return /^(问题[一二三四五六七八九十\d]+的分析|问题[一二三四五六七八九十\d]+分析)$/.test(stripped);
}

function checkCnProblemAnalysis(md, errors) {
  const sections = collectSection(md, ['问题分析']);
  for (const section of sections) {
    for (const line of section.lines) {
      if (!isHeading(line)) continue;
      const text = stripHeadingText(line);
      if (!isProblemAnalysisHeading(text)) {
        errors.push(`问题分析章节只能包含具体问题的小标题，例如“问题一的分析”。发现不合规标题：${text}`);
      }
    }
  }
}

function collectReferenceNumbers(md) {
  const lines = md.split(/\r?\n/);
  const refs = new Set();
  let inRefs = false;
  for (const line of lines) {
    if (/^##\s*(参考文献|References)\s*$/i.test(line.trim())) {
      inRefs = true;
      continue;
    }
    if (inRefs && /^##\s+/.test(line.trim())) break;
    if (inRefs) {
      const m = line.trim().match(/^\[(\d+)\]\s+/);
      if (m) refs.add(m[1]);
    }
  }
  return refs;
}

function main() {
  const mdArg = process.argv[2];
  if (!mdArg) usage();

  const mdPath = path.resolve(mdArg);
  if (!fs.existsSync(mdPath)) {
    console.error(`Markdown file not found: ${mdPath}`);
    process.exit(1);
  }

  const md = fs.readFileSync(mdPath, 'utf8');
  const baseDir = path.dirname(mdPath);
  const errors = [];
  const warnings = [];
  const lang = detectLanguage(md);

  if (/```mermaid\b/i.test(md)) {
    errors.push('Raw Mermaid code block found. Render it to an image first and reference the image in markdown.');
  }

  checkAbstractRules(md, errors);

  if (lang === 'cn') {
    checkCnProblemAnalysis(md, errors);
  } else if (hasSummarySheetTable(md)) {
    errors.push('MCM/ICM Summary Sheet table is written in markdown. Remove it from the source md and pass --mcm-problem/--mcm-year/--mcm-team to convert_md_to_docx.js.');
  }

  const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
  let m;
  while ((m = imagePattern.exec(md)) !== null) {
    const imageRef = m[2].trim().replace(/^<|>$/g, '');
    if (/^https?:\/\//i.test(imageRef)) {
      warnings.push(`Remote image reference should be downloaded locally before conversion: ${imageRef}`);
      continue;
    }
    const imagePath = path.resolve(baseDir, imageRef);
    if (!fs.existsSync(imagePath)) {
      errors.push(`Image not found: ${imageRef}`);
      continue;
    }
    if (fs.statSync(imagePath).size === 0) {
      errors.push(`Image file is empty: ${imageRef}`);
      continue;
    }
    if (isLikelyPlaceholderImage(imagePath)) {
      errors.push(`Image appears to be a placeholder, not a rendered diagram: ${imageRef}`);
    }
  }

  const displayMathDelimiters = (md.match(/^\s*\$\$\s*$/gm) || []).length;
  if (displayMathDelimiters % 2 !== 0) {
    errors.push('Unbalanced display math delimiters: standalone $$ count is odd.');
  }

  const codeFree = stripCodeBlocks(md);
  const refNumbers = collectReferenceNumbers(md);
  const citeNumbers = new Set();
  const citePattern = /\[(\d+)\]/g;
  while ((m = citePattern.exec(codeFree)) !== null) {
    citeNumbers.add(m[1]);
  }

  for (const n of citeNumbers) {
    if (!refNumbers.has(n)) {
      warnings.push(`Citation [${n}] has no matching reference entry under 参考文献/References.`);
    }
  }

  if (errors.length || warnings.length) {
    for (const warning of warnings) console.warn(`Warning: ${warning}`);
    for (const error of errors) console.error(`Error: ${error}`);
  }

  if (errors.length) {
    console.error(`Preflight failed with ${errors.length} error(s).`);
    process.exit(1);
  }

  console.log(`Preflight passed: ${mdPath}`);
  if (warnings.length) {
    console.log(`Warnings: ${warnings.length}`);
  }
}

main();
