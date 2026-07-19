// Главная сборка: объединяет все части, задаёт форматирование страницы
// (поля, нумерация страниц) под реальный формат кафедры, пакует
// Отчёт_по_практике.docx в корень репозитория. Запуск (из корня репозитория
// или из этой папки — путь вывода не зависит от текущей директории):
//   node report/assemble_report.js

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Footer, PageNumber,
  AlignmentType, FONT, SIZE, SIZE_SMALL, MARGINS,
} = require("./build_report.js");

const { titlePage, toc, vvedenie } = require("./generate_report.js");
const { mainChapter } = require("./generate_report_main_chapter.js");
const { zaklyuchenie, spisokLiteratury } = require("./generate_report_final.js");
const { prilozhenie1, prilozhenie2 } = require("./generate_report_appendices.js");

const footer = new Footer({
  children: [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SIZE_SMALL }),
      ],
    }),
  ],
});

// Титульный лист считается страницей 1, но номер на ней не печатается —
// пустой footer только для первой страницы раздела (как в примере кафедры).
const firstPageFooter = new Footer({ children: [new Paragraph({ children: [] })] });

const doc = new Document({
  creator: "P2P-drops research pipeline",
  title: "Отчёт о производственной практике — аналитика и сбор из открытых источников реквизитов платёжного мошенничества",
  styles: {
    default: {
      document: { run: { font: FONT, size: SIZE } },
    },
  },
  sections: [
    {
      properties: {
        page: {
          margin: MARGINS,
          pageNumbers: { start: 1 },
        },
        titlePage: true,
      },
      footers: { default: footer, first: firstPageFooter },
      children: [
        ...titlePage,
        ...toc,
        ...vvedenie,
        ...mainChapter,
        ...zaklyuchenie,
        ...spisokLiteratury,
        ...prilozhenie1,
        ...prilozhenie2,
      ],
    },
  ],
});

const OUT_PATH = path.join(__dirname, "..", "Отчёт_по_практике.docx");

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT_PATH, buffer);
  console.log("Готово:", OUT_PATH);
});
