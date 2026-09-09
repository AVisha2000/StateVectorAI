import test from "node:test";
import assert from "node:assert/strict";
import { workingNote, noteLines, workingNoteLayout, NOTE_CANVAS } from "./workingNote.js";

test("reader document retains the full source and explicit illustrative provenance", () => {
  const studio = { title: "Compare boundary conditions", description: "Keep the negative result. ".repeat(100) };
  const note = workingNote(studio);
  assert.equal(note.title, studio.title);
  assert.equal(note.passages[0].text, studio.description);
  assert.deepEqual(note.passages.map(p=>p.title), ["The question","The assumption","The next small test"]);
  assert.match(note.disclosure, /not a published paper or a research result/);
  assert.match(note.eyebrow, /DEMONSTRATION/);
  assert.equal(studio.description, note.passages[0].text);
});

test("physical preview wraps text, marks truncation, and preserves complete short inputs", () => {
  const measure = text => Array.from(text).length;
  assert.deepEqual(noteLines("One small test", 9, measure, 4), ["One small","test"]);
  assert.deepEqual(noteLines("One small test", 9, measure, 1), ["One smal…"]);
  assert.deepEqual(noteLines("", 9, measure, 2), []);
  for (const text of ["word ".repeat(300), "x".repeat(300), "🔬".repeat(30)]) {
    const lines = noteLines(text, 9, measure, 3);
    assert.equal(lines.length,3);
    assert.ok(lines.every(line=>measure(line)<=9));
    assert.ok(lines.at(-1).endsWith("…"));
    assert.ok(lines.every(line=>!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/.test(line)));
  }
  let calls=0;
  noteLines("word ".repeat(10000),9,value=>{calls++;return value.length;},3);
  assert.ok(calls<50,"do not measure discarded paragraphs after preview fills");
});

test("printed rows stay inside authored page and separate from neighboring sections", () => {
  const measure = (text,font) => Array.from(text).length * Number(font.match(/(\d+)px/)[1]) * .6;
  for (const title of ["Compare boundaries", "Very long title ".repeat(70), "λ".repeat(600)]) {
    const note = workingNote({title, description:"A long description with a negative finding. ".repeat(70)});
    const rows = workingNoteLayout(note,measure);
    for (const row of rows) {
      assert.ok(Number.isFinite(row.y));
      assert.ok(row.x>=0 && row.x+measure(row.text,row.font)<=NOTE_CANVAS.width-64+1e-9);
      assert.ok(row.y>30 && row.y<NOTE_CANVAS.height-30);
    }
    assert.ok(rows.filter(row=>row.y<292).every(row=>row.y<=234));
    assert.ok(rows.filter(row=>row.y>=336&&row.y<470).every(row=>row.y<=406));
    assert.ok(rows.filter(row=>row.y>=514&&row.y<682).every(row=>row.y<=619));
    assert.ok(rows.filter(row=>row.y>=726&&row.y<=796).every(row=>row.x+measure(row.text,row.font)<=604));
    assert.equal(note.title,title,"layout must not shorten the reader's source");
    assert.ok(rows.some(row=>row.text.includes("Illustrative working note")));
  }
});
