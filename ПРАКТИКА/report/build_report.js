// Сборка Отчёт_по_практике.docx под реальный формат кафедры (см.
// "ПРИМЕР ОФОРМЛЕНИЯ ПРАКТИКИ/Отчет, АлтунянАВ.docx" — НИТУ МИСИС,
// Институт информационных технологий и компьютерных наук, Кафедра
// инженерной кибернетики): Times New Roman 12pt, интервал 1.15, поля
// 3/1.5/1.75/1.5 см, красная строка ~0.8 см, заголовки обычным регистром.
// Фактические цифры (404 285 строк и т.д.) — из report_assets/ и трёх
// реальных SQLite-хранилищ, посчитаны отдельно перед сборкой, не изобретены.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Header, Footer, PageNumber, TableOfContents, ImageRun, Table, TableRow,
  TableCell, WidthType, BorderStyle, VerticalAlign, Bookmark,
  LevelFormat, convertInchesToTwip, PageBreak, ExternalHyperlink,
} = require("docx");

// report_assets/ и feature_assets/ — на уровень выше: их пишут python-скрипты
// (analyze.py/visualize.py/analyze_features.py) относительно СВОЕГО
// расположения (корень репозитория), а не относительно этого JS-тулинга.
const ASSETS = path.join(__dirname, "..", "report_assets");
const FEATURE_ASSETS = path.join(__dirname, "..", "feature_assets");

const FONT = "Times New Roman";
const SIZE = 24; // 12pt в half-points — как в реальном примере (docDefaults sz=24)
const SIZE_SMALL = 20; // 10pt — подписи к рисункам/таблицам
const HEADING1_SIZE = 28; // 14pt bold
const HEADING2_SIZE = 26; // 13pt bold
const LINE_SPACING = { line: 276, lineRule: "auto" }; // 1.15 — как в примере
const FIRST_LINE_INDENT = 454; // twips (~0.8 см) — как в примере

const CM = 566.929;
const MARGINS = {
  top: Math.round(1.75 * CM),
  bottom: Math.round(1.5 * CM),
  left: Math.round(3 * CM),
  right: Math.round(1.5 * CM),
};

function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LINE_SPACING, after: 120 },
    indent: opts.noIndent ? undefined : { firstLine: FIRST_LINE_INDENT },
    children: Array.isArray(text)
      ? text
      : [new TextRun({ text, font: FONT, size: SIZE, bold: opts.bold, italics: opts.italics })],
  });
}

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: opts.size || SIZE, bold: opts.bold, italics: opts.italics });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160, ...LINE_SPACING },
    children: [new TextRun({ text, font: FONT, size: HEADING1_SIZE, bold: true })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140, ...LINE_SPACING },
    children: [new TextRun({ text, font: FONT, size: HEADING2_SIZE, bold: true })],
  });
}

function centered(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE_SPACING, after: opts.after ?? 120 },
    children: [new TextRun({ text, font: FONT, size: opts.size || SIZE, bold: opts.bold, italics: opts.italics })],
  });
}

function bullet(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LINE_SPACING, after: 80 },
    bullet: { level: 0 },
    children: [new TextRun({ text, font: FONT, size: SIZE })],
  });
}

function figureImage(fileName, widthPx, heightPx, maxWidthCm = 15, dir = ASSETS) {
  const buf = fs.readFileSync(path.join(dir, fileName));
  const scale = (maxWidthCm * 37.795) / widthPx; // px @ 96dpi -> cm
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 60 },
    children: [
      new ImageRun({
        type: "png",
        data: buf,
        transformation: { width: Math.round(widthPx * scale), height: Math.round(heightPx * scale) },
      }),
    ],
  });
}

function figureCaption(num, text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text: `Рисунок ${num} — ${text}`, font: FONT, size: SIZE_SMALL })],
  });
}

module.exports = {
  FONT, SIZE, SIZE_SMALL, HEADING1_SIZE, HEADING2_SIZE, LINE_SPACING, MARGINS, FEATURE_ASSETS,
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Header, Footer, PageNumber, TableOfContents, ImageRun, Table, TableRow,
  TableCell, WidthType, BorderStyle, VerticalAlign, LevelFormat, PageBreak, ExternalHyperlink,
  body, run, h1, h2, centered, bullet, figureImage, figureCaption,
};
