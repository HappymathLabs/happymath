/**
 * convert_md_to_docx.js — Convert math modeling paper markdown to .docx
 *
 * Supports Chinese (CUMCM / 国赛) and English (MCM/ICM / 美赛) formatting.
 *
 * Usage:
 *   node scripts/convert_md_to_docx.js input.md --lang cn --output paper.docx
 *   node scripts/convert_md_to_docx.js input.md --lang en --output paper.docx
 *   node scripts/convert_md_to_docx.js input.md --lang en \
 *       --mcm-problem C --mcm-year 2026 --mcm-team 1111111 --output paper.docx
 *
 * If --lang is omitted, language is auto-detected from the markdown content.
 *
 * Dependencies:
 *   npm install docx temml fast-xml-parser
 */

'use strict';

const fs   = require('fs');
const path = require('path');

const {
  Document, Packer,
  Paragraph, TextRun, Math: DocxMath, MathRun,
  Table, TableRow, TableCell,
  Header, Footer,
  PageNumber, AlignmentType, LineRuleType, HeadingLevel,
  LevelFormat, LevelSuffix, BorderStyle, WidthType, ShadingType, VerticalAlign,
  TableOfContents, PageBreak, ImageRun, SequentialIdentifier, HeightRule, TableLayoutType,
} = require('docx');

const { mathmlToDocxChildren } = require('./mathml-to-docx');
const temml = require('temml');

// ─────────────────────────────────────────────────────────────────────────────
// Config presets
// ─────────────────────────────────────────────────────────────────────────────

const PRESETS = {
  cn: {
    pageW: 11906,      // A4 width DXA
    pageH: 16838,      // A4 height DXA
    margin: 1418,      // 2.5 cm
    bodyFont: { ascii: 'Cambria Math', hAnsi: 'Cambria Math', eastAsia: 'SimSun' },
    headingFont: { ascii: 'Cambria Math', hAnsi: 'Cambria Math', eastAsia: 'SimHei' },
    captionFont: { ascii: 'Cambria Math', hAnsi: 'Cambria Math', eastAsia: 'SimSun' },
    titleSize: 36,
    h1Size: 32,        // 三号 16pt
    h2Size: 28,        // 四号 14pt
    h3Size: 24,        // 小四 12pt
    bodySize: 24,      // 小四 12pt
    captionSize: 22,   // 五号 11pt
    headingBold: false,
    lineSpacing: 240,  // single
    firstLineIndent: 480,
    bodySpacing: { before: 0, after: 0, line: 240 },
    h1Spacing: { before: 120, after: 120, line: 288 },
    h2Spacing: { before: 120, after: 160, line: 276 },
    h3Spacing: { before: 120, after: 120, line: 264 },
    titleSpacing: { before: 0, after: 240, line: 288 },
    abstractHeadingSpacing: { before: 240, after: 120, line: 288 },
    keywordsSpacing: { before: 0, after: 240, line: 240 },
    captionSpacing: { before: 120, after: 60, line: 240 },
    figureSpacing: { before: 120, after: 60, line: 240 },
    tableBodySpacing: { before: 0, after: 120, line: 240 },
    formulaSpacing: { before: 0, after: 0, line: 240 },
    codeSpacing: { before: 120, after: 120, line: 240 },
    listSpacing: { before: 0, after: 0, line: 240 },
    tocTitle: '目录',
    figurePrefix: '图',
    tablePrefix: '表',
    keywordsLabel: '关键词：',
    referencesTitle: '参考文献',
  },
  en: {
    pageW: 12240,      // US Letter width DXA
    pageH: 15840,      // US Letter height DXA
    margin: 1440,      // 1 inch
    bodyFont: { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: 'SimSun' },
    headingFont: { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: 'SimHei' },
    captionFont: { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: 'SimSun' },
    titleSize: 36,
    h1Size: 32,
    h2Size: 28,
    h3Size: 24,
    bodySize: 24,
    captionSize: 22,
    headingBold: true,
    lineSpacing: 276,  // 1.15x
    firstLineIndent: 380,
    bodySpacing: { before: 0, after: 0, line: 276 },
    h1Spacing: { before: 120, after: 120, line: 288 },
    h2Spacing: { before: 120, after: 160, line: 276 },
    h3Spacing: { before: 120, after: 120, line: 264 },
    titleSpacing: { before: 0, after: 240, line: 288 },
    abstractHeadingSpacing: { before: 240, after: 160, line: 288 },
    keywordsSpacing: { before: 0, after: 260, line: 240 },
    captionSpacing: { before: 120, after: 60, line: 240 },
    figureSpacing: { before: 120, after: 60, line: 276 },
    tableBodySpacing: { before: 0, after: 120, line: 276 },
    formulaSpacing: { before: 0, after: 0, line: 276 },
    codeSpacing: { before: 120, after: 120, line: 276 },
    listSpacing: { before: 0, after: 0, line: 276 },
    tocTitle: 'Table of Contents',
    figurePrefix: 'Figure',
    tablePrefix: 'Table',
    keywordsLabel: 'Keywords: ',
    referencesTitle: 'References',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

let CFG = null;
let CONTENT_W = 0;
let _chapter = 0;
let _formulaNumber = 0;

const THICK = { style: BorderStyle.SINGLE, size: 12, color: '000000' };
const THIN  = { style: BorderStyle.SINGLE, size: 6,  color: '000000' };
const NONE  = { style: BorderStyle.NONE,   size: 0,  color: 'FFFFFF' };

// ─────────────────────────────────────────────────────────────────────────────
// CLI arguments
// ─────────────────────────────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const result = { lang: null, output: null, input: null, mcm: {} };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--lang' || a === '-l') result.lang = args[++i];
    else if (a === '--output' || a === '-o') result.output = args[++i];
    else if (a === '--mcm-problem') result.mcm.problem = args[++i];
    else if (a === '--mcm-year') result.mcm.year = args[++i];
    else if (a === '--mcm-team') result.mcm.team = args[++i];
    else if (!a.startsWith('-') && !result.input) result.input = a;
  }
  return result;
}

function detectLanguage(text) {
  // Simple heuristic: if more than 10% of characters are Chinese, use cn
  const chineseChars = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const total = text.length || 1;
  return (chineseChars / total) > 0.05 ? 'cn' : 'en';
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline math & citation helpers (reused from docx-editor-cn)
// ─────────────────────────────────────────────────────────────────────────────

const UNICODE_TO_LATEX = {
  'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\varepsilon',
  'ζ': '\\zeta', 'η': '\\eta', 'θ': '\\theta', 'ι': '\\iota', 'κ': '\\kappa',
  'λ': '\\lambda', 'μ': '\\mu', 'ν': '\\nu', 'ξ': '\\xi', 'π': '\\pi',
  'ρ': '\\rho', 'σ': '\\sigma', 'τ': '\\tau', 'υ': '\\upsilon', 'φ': '\\phi',
  'χ': '\\chi', 'ψ': '\\psi', 'ω': '\\omega',
  'Γ': '\\Gamma', 'Δ': '\\Delta', 'Θ': '\\Theta', 'Λ': '\\Lambda', 'Ξ': '\\Xi',
  'Π': '\\Pi', 'Σ': '\\Sigma', 'Φ': '\\Phi', 'Ψ': '\\Psi', 'Ω': '\\Omega',
  '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
  '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
  'ₐ': '_a', 'ₑ': '_e', 'ₕ': '_h', 'ᵢ': '_i', 'ⱼ': '_j', 'ₖ': '_k',
  'ₗ': '_l', 'ₘ': '_m', 'ₙ': '_n', 'ₒ': '_o', 'ₚ': '_p',
  'ᵣ': '_r', 'ₛ': '_s', 'ₜ': '_t', 'ᵤ': '_u', 'ᵥ': '_v',
  'ₓ': '_x', 'ᵧ': '_y',
  '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
  '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
  'ⁿ': '^n', 'ⁱ': '^i',
  '∞': '\\infty', '∑': '\\sum', '∏': '\\prod', '∫': '\\int',
  '≤': '\\leq', '≥': '\\geq', '≠': '\\neq', '≈': '\\approx',
  '→': '\\to', '←': '\\leftarrow', '↔': '\\leftrightarrow',
  '∈': '\\in', '∉': '\\notin', '⊂': '\\subset', '⊃': '\\supset',
  '∀': '\\forall', '∃': '\\exists', '∧': '\\land', '∨': '\\lor',
  '×': '\\times', '÷': '\\div', '±': '\\pm', '∓': '\\mp',
  '·': '\\cdot', '…': '\\ldots', '⋯': '\\cdots',
  '′': "'", '″': "''",
  '⟨': '\\langle ', '⟩': ' \\rangle',
};

const GREEK_CHARS = 'αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ';
const SUB_SUP_CHARS = '₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵧ⁰¹²³⁴⁵⁶⁷⁸⁹ⁿⁱ';
const MATH_OPS = '∞∑∏∫≤≥≠≈→←↔∈∉⊂⊃∀∃∧∨×÷±∓·…⋯′″⟨⟩';

function unicodeToLatex(text) {
  let result = text.replace(/([A-Za-z])̄/g, '\\bar{$1}');
  result = result.replace(/(\^|_)(?!\{)([A-Za-z0-9]+)/g, '$1{$2}');
  for (const [unicode, latex] of Object.entries(UNICODE_TO_LATEX)) {
    result = result.split(unicode).join(latex);
  }
  return result;
}

function containsMath(text) {
  if (new RegExp(`[${GREEK_CHARS}]`).test(text)) return true;
  if (new RegExp(`[${SUB_SUP_CHARS}]`).test(text)) return true;
  if (new RegExp(`[${MATH_OPS}]`).test(text)) return true;
  if (/[A-Z]\*/.test(text)) return true;
  if (/\$[^$]+\$/.test(text)) return true;
  if (/[_^]\{[^}]+\}/.test(text)) return true;
  if (/[A-Za-z]\d*[_^][a-zA-Z0-9]/.test(text)) return true;
  if (/\\[a-zA-Z]/.test(text)) return true;
  if (/[A-Za-z]̄/.test(text)) return true;
  return false;
}

function containsCitation(text) {
  return /\[\d+\]/.test(text);
}

function parseInlineMath(text) {
  const mathPattern = new RegExp(
    `\\$([^$]+)\\$` +
    `|(\\\\[a-zA-Z]+\\{[^}]*\\}[ \\t]*\\{[^}]*\\})` +
    `|(\\\\[a-zA-Z]+\\{[^}]*\\})` +
    `|(\\\\[a-zA-Z]+)` +
    `|([A-Za-z${GREEK_CHARS}]+[_^]\\{[^}]+\\})` +
    `|([A-Za-z]+\\d*[_^][a-zA-Z0-9]+\\s*\\([^)]+\\))` +
    `|([A-Za-z]+\\d*[_^][a-zA-Z0-9]+)` +
    `|([A-Z][${SUB_SUP_CHARS}]+\\*?\\s*\\([^)]+\\))` +
    `|([A-Z]\\*\\s*\\([^)]+\\))` +
    `|([A-Z]\\s*\\([^)]*[${GREEK_CHARS}${SUB_SUP_CHARS}][^)]*\\))` +
    `|([${GREEK_CHARS}][${SUB_SUP_CHARS}]*\\*?)` +
    `|([A-Za-z]+[${SUB_SUP_CHARS}]+\\*?)` +
    `|([A-Z]\\*)` +
    `|([${MATH_OPS}])`,
    'g'
  );

  const children = [];
  let lastIndex = 0;
  let m;
  while ((m = mathPattern.exec(text)) !== null) {
    if (m.index > lastIndex) children.push(new TextRun(text.slice(lastIndex, m.index)));
    const mc = m[1] || m[2] || m[3] || m[4] || m[5] || m[6] || m[7] || m[8] || m[9] || m[10] || m[11] || m[12] || m[13] || m[14];
    if (mc) {
      const latex = unicodeToLatex(mc);
      try {
        const mathml = temml.renderToString(latex, { displayMode: false, throwOnError: false });
        const mathChildren = mathmlToDocxChildren(mathml);
        if (mathChildren && mathChildren.length) {
          children.push(new DocxMath({ children: mathChildren }));
        } else {
          children.push(new DocxMath({ children: [new MathRun(mc)] }));
        }
      } catch (e) {
        children.push(new DocxMath({ children: [new MathRun(mc)] }));
      }
    }
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < text.length) children.push(new TextRun(text.slice(lastIndex)));
  if (children.length === 0) children.push(new TextRun(text));
  return children;
}

function parseInlineContent(text) {
  const children = [];
  let lastIndex = 0;
  // Match citations [n] and inline math $...$
  const pattern = /(\[\d+\])|\$([^$]+)\$/g;
  let m;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > lastIndex) {
      const plain = text.slice(lastIndex, m.index);
      if (containsMath(plain)) {
        children.push(...parseInlineMath(plain));
      } else {
        children.push(new TextRun(plain));
      }
    }
    if (m[1]) {
      children.push(new TextRun({ text: m[1], superScript: true }));
    } else if (m[2]) {
      const latex = m[2];
      try {
        const mathml = temml.renderToString(latex, { displayMode: false, throwOnError: false });
        const mathChildren = mathmlToDocxChildren(mathml);
        children.push(new DocxMath({ children: mathChildren && mathChildren.length ? mathChildren : [new MathRun(latex)] }));
      } catch (e) {
        children.push(new DocxMath({ children: [new MathRun(latex)] }));
      }
    }
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < text.length) {
    const plain = text.slice(lastIndex);
    if (containsMath(plain)) {
      children.push(...parseInlineMath(plain));
    } else {
      children.push(new TextRun(plain));
    }
  }
  if (children.length === 0) children.push(new TextRun(text));
  return children;
}

// ─────────────────────────────────────────────────────────────────────────────
// Style helpers
// ─────────────────────────────────────────────────────────────────────────────

function font(fontObj) {
  return { ...fontObj };
}

function paragraphStyle(id) {
  return id;
}

function buildStyles() {
  return {
    default: {
      document: {
        run: { font: font(CFG.bodyFont), size: CFG.bodySize },
        paragraph: {
          spacing: { ...CFG.bodySpacing, lineRule: LineRuleType.AUTO },
          indent: { firstLine: CFG.firstLineIndent },
        },
      },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: font(CFG.headingFont), size: CFG.h1Size, bold: CFG.headingBold },
        paragraph: {
          alignment: AlignmentType.CENTER,
          indent: { firstLine: 0 },
          spacing: { ...CFG.h1Spacing, lineRule: LineRuleType.AUTO },
          outlineLevel: 0,
        },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: font(CFG.headingFont), size: CFG.h2Size, bold: CFG.headingBold },
        paragraph: {
          alignment: AlignmentType.LEFT,
          indent: { firstLine: 0 },
          spacing: { ...CFG.h2Spacing, lineRule: LineRuleType.AUTO },
          outlineLevel: 1,
        },
      },
      {
        id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: font(CFG.headingFont), size: CFG.h3Size, bold: CFG.headingBold },
        paragraph: {
          alignment: AlignmentType.LEFT,
          indent: { firstLine: 0 },
          spacing: { ...CFG.h3Spacing, lineRule: LineRuleType.AUTO },
          outlineLevel: 2,
        },
      },
      {
        id: 'FigureCaption', name: 'Figure Caption', basedOn: 'Normal',
        run: { font: font(CFG.captionFont), size: CFG.captionSize, bold: true },
        paragraph: {
          alignment: AlignmentType.CENTER,
          indent: { firstLine: 0 },
          spacing: { ...CFG.captionSpacing, lineRule: LineRuleType.AUTO },
        },
      },
      {
        id: 'TableCaption', name: 'Table Caption', basedOn: 'Normal',
        run: { font: font(CFG.captionFont), size: CFG.captionSize, bold: true },
        paragraph: {
          alignment: AlignmentType.CENTER,
          indent: { firstLine: 0 },
          spacing: { ...CFG.captionSpacing, lineRule: LineRuleType.AUTO },
        },
      },
      {
        id: 'Reference', name: 'Reference', basedOn: 'Normal',
        run: { font: font(CFG.bodyFont), size: CFG.bodySize },
        paragraph: {
          spacing: { ...CFG.bodySpacing, lineRule: LineRuleType.AUTO },
          indent: { left: 480, hanging: 480, firstLine: 0 },
        },
      },
    ],
  };
}

function buildNumberingConfig(chapterCount) {
  const configs = [
    {
      reference: 'references',
      levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '[%1]',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 480, hanging: 480 } } } },
      ],
    },
    {
      reference: 'bullets',
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: '•',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      ],
    },
    {
      reference: 'numbers',
      levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.DECIMAL, text: '%2.',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        { level: 2, format: LevelFormat.DECIMAL, text: '%3.',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
      ],
    },
  ];

  for (let c = 1; c <= chapterCount; c++) {
    configs.push({
      reference: `sections_c${c}`,
      levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: `${c}.%1`,
          suffix: LevelSuffix.SPACE, alignment: AlignmentType.LEFT },
        { level: 1, format: LevelFormat.DECIMAL, text: `${c}.%1.%2`,
          suffix: LevelSuffix.SPACE, alignment: AlignmentType.LEFT },
      ],
    });
  }

  return { config: configs };
}

function addOrderedListConfigs(numbering, listCount) {
  for (let n = 1; n <= listCount; n++) {
    numbering.config.push({
      reference: `numbers_l${n}`,
      levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.DECIMAL, text: '%2.',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 360 } } } },
        { level: 2, format: LevelFormat.DECIMAL, text: '%3.',
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
      ],
    });
  }
  return numbering;
}

// ─────────────────────────────────────────────────────────────────────────────
// Content helpers
// ─────────────────────────────────────────────────────────────────────────────

function pageBreakPara() {
  return new Paragraph({ children: [new PageBreak()] });
}

function spacingPara(spacing) {
  return new Paragraph({
    indent: { firstLine: 0 },
    spacing: { ...spacing, lineRule: LineRuleType.AUTO },
    children: [],
  });
}

function h1(text) {
  _chapter++;
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    indent: { firstLine: 0 },
    spacing: { ...CFG.h1Spacing, lineRule: LineRuleType.AUTO },
    children: [new TextRun(text)],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    numbering: { reference: `sections_c${_chapter}`, level: 0 },
    indent: { firstLine: 0 },
    spacing: { ...CFG.h2Spacing, lineRule: LineRuleType.AUTO },
    children: [new TextRun(text)],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    numbering: { reference: `sections_c${_chapter}`, level: 1 },
    indent: { firstLine: 0 },
    spacing: { ...CFG.h3Spacing, lineRule: LineRuleType.AUTO },
    children: [new TextRun(text)],
  });
}

function stripStructuralNumber(text) {
  let t = text.trim();
  t = t.replace(/^第[一二三四五六七八九十百]+[章节部分]\s*[：:、.．-]?\s*/, '');
  t = t.replace(/^[一二三四五六七八九十]+[、.．]\s*/, '');
  t = t.replace(/^\(?\d+(?:\.\d+)*\)?[、.．:：\s-]+/, '');
  return t || text.trim();
}

function stripHeadingNumber(text, level) {
  // Level-2 headings are not auto-numbered by this converter, so keep their
  // source numbering. Level-3+ headings are auto-numbered and must be cleaned.
  if (level <= 2) return text.trim();
  return stripStructuralNumber(text);
}

function figCaption(text) {
  return new Paragraph({
    style: 'FigureCaption',
    indent: { firstLine: 0 },
    spacing: { ...CFG.captionSpacing, lineRule: LineRuleType.AUTO },
    children: [
      new TextRun(`${CFG.figurePrefix} ${_chapter}-`),
      new SequentialIdentifier(`figure_c${_chapter}`),
      new TextRun(` ${text}`),
    ],
  });
}

function tableCaption(text) {
  return new Paragraph({
    style: 'TableCaption',
    indent: { firstLine: 0 },
    spacing: { ...CFG.captionSpacing, lineRule: LineRuleType.AUTO },
    children: [
      new TextRun(`${CFG.tablePrefix} ${_chapter}-`),
      new SequentialIdentifier(`table_c${_chapter}`),
      new TextRun(` ${text}`),
    ],
  });
}

function readImageSize(filePath) {
  const buf = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.png' && buf.length >= 24 && buf.slice(0, 8).toString('hex') === '89504e470d0a1a0a') {
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  }
  if ((ext === '.jpg' || ext === '.jpeg') && buf.length >= 4 && buf[0] === 0xff && buf[1] === 0xd8) {
    let offset = 2;
    while (offset + 9 < buf.length) {
      if (buf[offset] !== 0xff) break;
      const marker = buf[offset + 1];
      const length = buf.readUInt16BE(offset + 2);
      if (length < 2) break;
      if ((marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) ||
          (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)) {
        return { width: buf.readUInt16BE(offset + 7), height: buf.readUInt16BE(offset + 5) };
      }
      offset += 2 + length;
    }
  }
  if (ext === '.gif' && buf.length >= 10 && ['GIF87a', 'GIF89a'].includes(buf.slice(0, 6).toString('ascii'))) {
    return { width: buf.readUInt16LE(6), height: buf.readUInt16LE(8) };
  }
  if (ext === '.bmp' && buf.length >= 26 && buf.slice(0, 2).toString('ascii') === 'BM') {
    return { width: Math.abs(buf.readInt32LE(18)), height: Math.abs(buf.readInt32LE(22)) };
  }
  if (ext === '.svg') {
    const text = buf.toString('utf8', 0, Math.min(buf.length, 4096));
    const widthMatch = text.match(/\bwidth=["']?([\d.]+)(?:px)?["']?/i);
    const heightMatch = text.match(/\bheight=["']?([\d.]+)(?:px)?["']?/i);
    if (widthMatch && heightMatch) {
      return { width: Number(widthMatch[1]), height: Number(heightMatch[1]) };
    }
    const viewBoxMatch = text.match(/\bviewBox=["']\s*[-\d.]+\s+[-\d.]+\s+([\d.]+)\s+([\d.]+)\s*["']/i);
    if (viewBoxMatch) {
      return { width: Number(viewBoxMatch[1]), height: Number(viewBoxMatch[2]) };
    }
  }
  return null;
}

function fitImageSize(size) {
  if (!size || !size.width || !size.height || size.width <= 0 || size.height <= 0) {
    return { width: 450, height: 300 };
  }
  const maxWidth = Math.min(560, Math.floor(CONTENT_W / 15));
  const maxHeight = 620;
  const scale = Math.min(1, maxWidth / size.width, maxHeight / size.height);
  return {
    width: Math.max(1, Math.round(size.width * scale)),
    height: Math.max(1, Math.round(size.height * scale)),
  };
}

function blockFormula(latex) {
  _formulaNumber++;
  const noBorders = { top: NONE, bottom: NONE, left: NONE, right: NONE };
  const mathObj = latexToMath(latex);

  const leftCell = new TableCell({
    width: { size: 567, type: WidthType.DXA },
    borders: noBorders,
    shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      indent: { firstLine: 0 },
      spacing: { ...CFG.formulaSpacing, lineRule: LineRuleType.AUTO },
      children: [],
    })],
  });

  const formulaCell = new TableCell({
    width: { size: CONTENT_W - 1134, type: WidthType.DXA },
    borders: noBorders,
    shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      indent: { firstLine: 0 },
      spacing: { ...CFG.formulaSpacing, lineRule: LineRuleType.AUTO },
      children: [mathObj],
    })],
  });

  const numberCell = new TableCell({
    width: { size: 567, type: WidthType.DXA },
    borders: noBorders,
    shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      indent: { firstLine: 0 },
      spacing: { ...CFG.formulaSpacing, lineRule: LineRuleType.AUTO },
      children: [new TextRun(`(${_formulaNumber})`)],
    })],
  });

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [567, CONTENT_W - 1134, 567],
    borders: {
      top: NONE, bottom: NONE, left: NONE, right: NONE,
      insideHorizontal: NONE, insideVertical: NONE,
    },
    rows: [new TableRow({ children: [leftCell, formulaCell, numberCell] })],
  });
}

function latexToMath(latex) {
  try {
    const mathml = temml.renderToString(latex, { displayMode: true, throwOnError: false });
    const children = mathmlToDocxChildren(mathml);
    if (children && children.length) return new DocxMath({ children });
  } catch (e) {
    console.warn(`[formula] LaTeX parse error: ${latex}`, e.message);
  }
  return new DocxMath({ children: [new MathRun(latex)] });
}

function threeLineTable(headers, rows, captionText) {
  const n = headers.length;
  const colWidths = (function () {
    const w = Math.floor(CONTENT_W / n);
    const arr = Array(n).fill(w);
    arr[n - 1] = CONTENT_W - w * (n - 1);
    return arr;
  })();

  const cellOf = (text, w, borders, bold = false) => {
    let cellChildren;
    if (containsMath(text)) {
      cellChildren = parseInlineContent(text);
      if (bold) {
        cellChildren = cellChildren.map(child => {
          if (child instanceof TextRun) {
            return new TextRun({ text: child.text || '', bold: true, font: child.font });
          }
          return child;
        });
      }
    } else {
      cellChildren = [new TextRun({ text, bold, font: font(CFG.captionFont) })];
    }

    return new TableCell({
      width: { size: w, type: WidthType.DXA },
      borders,
      shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        indent: { firstLine: 0 },
        spacing: { ...CFG.tableBodySpacing, lineRule: LineRuleType.AUTO },
        children: cellChildren,
      })],
    });
  };

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cellOf(h, colWidths[i], { top: THICK, bottom: THIN, left: NONE, right: NONE }, true)),
  });

  const bodyRows = rows.map((row, ri) => {
    const isLast = ri === rows.length - 1;
    return new TableRow({
      children: row.map((cell, i) => cellOf(String(cell), colWidths[i], {
        top: NONE,
        bottom: isLast ? THICK : NONE,
        left: NONE,
        right: NONE,
      })),
    });
  });

  const elements = [];
  if (captionText) elements.push(tableCaption(captionText));
  elements.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    layout: TableLayoutType.FIXED,
    rows: [headerRow, ...bodyRows],
  }));
  return elements;
}

// ─────────────────────────────────────────────────────────────────────────────
// Markdown parsing
// ─────────────────────────────────────────────────────────────────────────────

function stripMarkdownBold(text) {
  return text.replace(/\*\*(.+?)\*\*/g, '$1');
}

function parseMarkdownLine(text) {
  // Parse inline bold and italic
  const parts = [];
  const regex = /(\*\*.+?\*\*)|(\*.+?\*)|(`[^`]+`)/g;
  let lastIndex = 0;
  let m;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, m.index) });
    }
    if (m[1]) {
      parts.push({ type: 'bold', value: m[1].slice(2, -2) });
    } else if (m[2]) {
      parts.push({ type: 'italic', value: m[2].slice(1, -1) });
    } else if (m[3]) {
      parts.push({ type: 'code', value: m[3].slice(1, -1) });
    }
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < text.length) parts.push({ type: 'text', value: text.slice(lastIndex) });
  if (parts.length === 0) parts.push({ type: 'text', value: text });
  return parts;
}

function parseInlineRich(text) {
  const parts = parseMarkdownLine(text);
  const runs = [];
  for (const p of parts) {
    if (p.type === 'text') {
      runs.push(...parseInlineContent(p.value));
    } else if (p.type === 'bold') {
      runs.push(new TextRun({ text: p.value, bold: true }));
    } else if (p.type === 'italic') {
      runs.push(new TextRun({ text: p.value, italics: true }));
    } else if (p.type === 'code') {
      runs.push(new TextRun({ text: p.value, font: 'Courier New' }));
    }
  }
  return runs;
}

function parseReferenceLine(text) {
  const match = text.match(/^(\[\d+\])\s*(.*)$/);
  if (!match) return parseInlineRich(text);
  return [
    new TextRun({ text: match[1] }),
    new TextRun(' '),
    ...parseInlineRich(match[2]),
  ];
}

function markdownToDocxElements(md, baseDir) {
  const lines = md.split(/\r?\n/);
  const elements = [];
  let i = 0;
  let inCodeBlock = false;
  let codeBuffer = [];
  let codeLang = '';
  let pendingTableCaption = null;
  let orderedListCount = 0;
  let currentOrderedListRef = null;
  let previousWasOrderedList = false;
  let inReferences = false;

  // First pass: count chapters for numbering config (level-2 headings, excluding abstract)
  let chapterCount = 0;
  for (const line of lines) {
    const hMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (hMatch && hMatch[1].length === 2) {
      const txt = stripStructuralNumber(hMatch[2].trim());
      if (txt !== '摘要' && txt !== 'Abstract') chapterCount++;
    }
  }
  // Ensure at least a few chapter slots
  chapterCount = Math.max(chapterCount, 3);

  while (i < lines.length) {
    const line = lines[i];

    // Code blocks
    if (line.trim().startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLang = line.trim().slice(3).trim();
        codeBuffer = [];
      } else {
        inCodeBlock = false;
        const codeText = codeBuffer.join('\n');
        elements.push(new Paragraph({
          indent: { firstLine: 0, left: 240 },
          spacing: { ...CFG.codeSpacing, lineRule: LineRuleType.AUTO },
          children: [new TextRun({ text: codeText, font: 'Courier New', size: 20 })],
        }));
        codeBuffer = [];
      }
      i++;
      continue;
    }

    if (inCodeBlock) {
      codeBuffer.push(line);
      i++;
      continue;
    }

    // Empty line
    if (line.trim() === '') {
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      i++;
      continue;
    }

    // Mermaid diagram blocks — skip raw mermaid code; images should already be saved
    if (line.trim().startsWith('```mermaid')) {
      while (i < lines.length && !lines[i].trim().startsWith('```')) i++;
      i++;
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      let text = headingMatch[2].trim();
      previousWasOrderedList = false;
      currentOrderedListRef = null;

      // Title is the only level-1 heading treated as document title
      if (level === 1) {
        elements.push(new Paragraph({
          alignment: AlignmentType.CENTER,
          indent: { firstLine: 0 },
          spacing: { ...CFG.titleSpacing, lineRule: LineRuleType.AUTO },
          children: [new TextRun({
            text,
            bold: CFG.headingBold,
            size: CFG.titleSize,
            font: font(CFG.headingFont),
          })],
        }));
        i++;
        continue;
      }

      const structuralText = stripStructuralNumber(text);
      text = stripHeadingNumber(text, level);

      // Abstract heading (level 2): centered bold, not a chapter
      if (level === 2 && (structuralText === '摘要' || structuralText === 'Abstract')) {
        elements.push(new Paragraph({
          alignment: AlignmentType.CENTER,
          indent: { firstLine: 0 },
          spacing: { ...CFG.abstractHeadingSpacing, lineRule: LineRuleType.AUTO },
          children: [new TextRun({ text, bold: CFG.headingBold, size: CFG.h1Size, font: font(CFG.headingFont) })],
        }));
        i++;
        continue;
      }

      // References section starts on a new page
      if (structuralText === CFG.referencesTitle) {
        elements.push(pageBreakPara());
        inReferences = true;
      } else if (level === 2) {
        inReferences = false;
      }

      // Appendices also start on a new page
      if (structuralText === '附录' || structuralText === 'Appendices') {
        elements.push(pageBreakPara());
      }

      if (level === 2) {
        // Chapter heading
        elements.push(h1(text));
      } else if (level === 3) {
        elements.push(h2(text));
      } else if (level >= 4) {
        elements.push(h3(text));
      }
      i++;
      continue;
    }

    // Keywords line
    const keywordsMatchCN = line.match(/^关键词[：:]\s*(.+)$/);
    const keywordsMatchEN = line.match(/^Keywords:\s*(.+)$/i);
    const keywordsMatch = keywordsMatchCN || keywordsMatchEN;
    if (keywordsMatch) {
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      const label = keywordsMatchCN ? CFG.keywordsLabel : 'Keywords: ';
      elements.push(new Paragraph({
        indent: { firstLine: 0 },
        spacing: { ...CFG.keywordsSpacing, lineRule: LineRuleType.AUTO },
        children: [
          new TextRun({ text: label, bold: true, font: font(CFG.bodyFont) }),
          new TextRun({ text: keywordsMatch[1], font: font(CFG.bodyFont) }),
        ],
      }));
      elements.push(pageBreakPara());
      i++;
      continue;
    }

    // Table caption line (above table)
    const tableCaptionMatch = line.match(/^(?:表|Table)\s*\d+[-–—]\d+\s+(.+)$/);
    if (tableCaptionMatch) {
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      pendingTableCaption = tableCaptionMatch[1];
      i++;
      continue;
    }

    // Tables
    if (line.includes('|')) {
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      const tableLines = [];
      while (i < lines.length && lines[i].includes('|')) {
        tableLines.push(lines[i]);
        i++;
      }
      // Remove separator line
      const contentLines = tableLines.filter(l => !/^\s*\|[-:\s|]*\|\s*$/.test(l));
      if (contentLines.length >= 1) {
        const cells = contentLines.map(l => l.split('|').map(c => c.trim()).filter(c => c !== ''));
        if (cells.length >= 1) {
          const headers = cells[0];
          const rows = cells.slice(1);
          elements.push(...threeLineTable(headers, rows, pendingTableCaption));
          pendingTableCaption = null;
        }
      }
      continue;
    }

    // Images
    const imageMatch = line.match(/^!\[(.*?)\]\((.+?)\)$/);
    if (imageMatch) {
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      let caption = imageMatch[1].trim();
      // Strip existing figure prefix like "图 1-1" or "Figure 1-1"
      caption = caption.replace(/^(?:图|Figure)\s*\d+[-–—]\d+\s*/, '');
      const imgPath = path.resolve(baseDir, imageMatch[2]);
      if (fs.existsSync(imgPath)) {
        const ext = path.extname(imgPath).toLowerCase().replace('.', '');
        const validTypes = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg'];
        const type = validTypes.includes(ext) ? ext : 'png';
        const imgData = fs.readFileSync(imgPath);
        const transformation = fitImageSize(readImageSize(imgPath));
        elements.push(new Paragraph({
          alignment: AlignmentType.CENTER,
          indent: { firstLine: 0 },
          spacing: { ...CFG.figureSpacing, lineRule: LineRuleType.AUTO },
          children: [new ImageRun({
            type,
            data: imgData,
            transformation,
            altText: { title: caption, description: caption, name: path.basename(imgPath) },
          })],
        }));
        if (caption) elements.push(figCaption(caption));
      } else {
        console.warn(`[warn] Image not found: ${imgPath}`);
        elements.push(new Paragraph({ children: [new TextRun(`[Image: ${imageMatch[2]}]`)] }));
      }
      i++;
      continue;
    }

    // Display math $$
    if (line.trim() === '$$') {
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      const mathLines = [];
      i++;
      while (i < lines.length && lines[i].trim() !== '$$') {
        mathLines.push(lines[i]);
        i++;
      }
      i++;
      const latex = mathLines.join('\n');
      if (latex.trim()) {
        elements.push(spacingPara({ before: 80, after: 0, line: CFG.lineSpacing }));
        elements.push(blockFormula(latex));
        elements.push(spacingPara({ before: 0, after: 80, line: CFG.lineSpacing }));
      }
      continue;
    }

    // Lists
    const bulletMatch = line.match(/^(\s*)[-*+]\s+(.+)$/);
    const numberMatch = line.match(/^(\s*)\d+\.\s+(.+)$/);
    if (bulletMatch || numberMatch) {
      const isBullet = !!bulletMatch;
      const match = bulletMatch || numberMatch;
      const indentLevel = Math.floor(match[1].length / 2);
      const text = match[2];
      if (!isBullet && !previousWasOrderedList) {
        orderedListCount++;
        currentOrderedListRef = `numbers_l${orderedListCount}`;
      }
      elements.push(new Paragraph({
        numbering: { reference: isBullet ? 'bullets' : currentOrderedListRef, level: Math.min(indentLevel, 2) },
        indent: { firstLine: 0 },
        spacing: { ...CFG.listSpacing, lineRule: LineRuleType.AUTO },
        children: parseInlineRich(text),
      }));
      previousWasOrderedList = !isBullet;
      if (isBullet) currentOrderedListRef = null;
      i++;
      continue;
    }

    // References
    if (inReferences && /^\[\d+\]\s+/.test(line.trim())) {
      elements.push(new Paragraph({
        style: 'Reference',
        spacing: { ...CFG.bodySpacing, lineRule: LineRuleType.AUTO },
        indent: { left: 480, hanging: 480, firstLine: 0 },
        children: parseReferenceLine(line.trim()),
      }));
      previousWasOrderedList = false;
      currentOrderedListRef = null;
      i++;
      continue;
    }

    // Normal paragraph
    elements.push(new Paragraph({
      spacing: { ...CFG.bodySpacing, lineRule: LineRuleType.AUTO },
      indent: { firstLine: CFG.firstLineIndent },
      children: parseInlineRich(line),
    }));
    previousWasOrderedList = false;
    currentOrderedListRef = null;
    i++;
  }

  return { elements, chapterCount, orderedListCount };
}

// ─────────────────────────────────────────────────────────────────────────────
// MCM summary sheet header
// ─────────────────────────────────────────────────────────────────────────────

function mcmSummarySheet(mcm) {
  if (!mcm || !mcm.problem || !mcm.year || !mcm.team) return [];

  const headerFont = { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: 'SimSun' };
  const whiteRule = { style: BorderStyle.SINGLE, size: 1, color: 'FFFFFF', space: 0 };
  const sheetCellBorders = { top: whiteRule, bottom: whiteRule, left: whiteRule, right: whiteRule };
  const blackPixelPng = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg==',
    'base64'
  );
  const placeholder = (value) => 'x'.repeat(String(value).length);
  const valueRun = (text) => new TextRun({
    text,
    bold: false,
    color: 'FF0000',
    size: 36,
    font: headerFont,
  });
  const labelRun = (text) => new TextRun({
    text,
    bold: true,
    size: 24,
    font: headerFont,
  });
  const widthAnchorPara = () => new Paragraph({
    alignment: AlignmentType.CENTER,
    indent: { firstLine: 0 },
    spacing: { before: 0, after: 0, line: 20, lineRule: LineRuleType.EXACT },
    children: [new TextRun({
      text: 'W'.repeat(160),
      color: 'FFFFFF',
      size: 2,
      font: 'Times New Roman',
    })],
  });
  const para = (children, spacing = {}) => new Paragraph({
    alignment: AlignmentType.CENTER,
    indent: { firstLine: 0 },
    spacing: { before: 0, after: 0, line: 240, lineRule: LineRuleType.AUTO, ...spacing },
    children,
  });
  const cell = (children) => new TableCell({
    width: { size: Math.floor(CONTENT_W / 3), type: WidthType.DXA },
    borders: sheetCellBorders,
    shading: { fill: 'FFFFFF', type: ShadingType.CLEAR },
    margins: { top: 0, bottom: 0, left: 0, right: 0 },
    verticalAlign: VerticalAlign.TOP,
    children,
  });

  const rowHeight = 1010;
  const cellWidth = Math.floor(CONTENT_W / 3);
  const contentTable = new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [cellWidth, cellWidth, CONTENT_W - cellWidth * 2],
    layout: TableLayoutType.FIXED,
    alignment: AlignmentType.CENTER,
    borders: {
      top: whiteRule, bottom: whiteRule, left: whiteRule, right: whiteRule,
      insideHorizontal: whiteRule, insideVertical: whiteRule,
    },
    rows: [
      new TableRow({
        height: { value: rowHeight, rule: HeightRule.EXACT },
        children: [
          cell([
            widthAnchorPara(),
            para([labelRun('Problem Chosen')]),
            para([valueRun(placeholder(mcm.problem))], { before: 24, after: 40 }),
          ]),
          cell([
            widthAnchorPara(),
            para([labelRun(placeholder(mcm.year || new Date().getFullYear()))]),
            para([labelRun('MCM/ICM')]),
            para([labelRun('Summary Sheet')], { after: 40 }),
          ]),
          cell([
            widthAnchorPara(),
            para([labelRun('Team Control Number')]),
            para([valueRun(placeholder(mcm.team))], { before: 24, after: 40 }),
          ]),
        ],
      }),
    ],
  });

  const thinRule = new Paragraph({
    alignment: AlignmentType.CENTER,
    indent: { firstLine: 0 },
    spacing: { before: 0, after: 0, line: 24, lineRule: LineRuleType.EXACT },
    children: [new ImageRun({
      data: blackPixelPng,
      transformation: { width: Math.floor(CONTENT_W / 15), height: 2 },
      altText: { title: 'MCM summary sheet rule', description: 'Thin horizontal rule' },
    })],
  });

  return [contentTable, thinRule, new Paragraph({ indent: { firstLine: 0 }, spacing: { before: 80, after: 0 }, children: [] })];
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs();
  if (!args.input) {
    console.error('Usage: node convert_md_to_docx.js input.md [--lang cn|en] [--output output.docx] [--mcm-problem C] [--mcm-year 2026] [--mcm-team 1111111]');
    process.exit(1);
  }

  const mdPath = path.resolve(args.input);
  if (!fs.existsSync(mdPath)) {
    console.error(`File not found: ${mdPath}`);
    process.exit(1);
  }

  const md = fs.readFileSync(mdPath, 'utf-8');
  const lang = args.lang || detectLanguage(md);
  CFG = PRESETS[lang];
  CONTENT_W = CFG.pageW - 2 * CFG.margin;

  const baseDir = path.dirname(mdPath);
  const { elements, chapterCount, orderedListCount } = markdownToDocxElements(md, baseDir);

  const STYLES = buildStyles();
  const NUMBERING = addOrderedListConfigs(buildNumberingConfig(chapterCount), orderedListCount);

  const children = [];
  children.push(...mcmSummarySheet(args.mcm));
  children.push(...elements);

  const doc = new Document({
    styles: STYLES,
    numbering: NUMBERING,
    sections: [{
      properties: {
        page: {
          size: { width: CFG.pageW, height: CFG.pageH },
          margin: { top: CFG.margin, right: CFG.margin, bottom: CFG.margin, left: CFG.margin },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              indent: { firstLine: 0 },
              children: [new TextRun({ children: [PageNumber.CURRENT], font: font(CFG.bodyFont) })],
            }),
          ],
        }),
      },
      children,
    }],
  });

  const output = args.output || mdPath.replace(/\.md$/i, '.docx');
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(output, buf);
  console.log(`✓ Written: ${output} (${lang.toUpperCase()} format)`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
