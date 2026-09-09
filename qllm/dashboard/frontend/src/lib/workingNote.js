// One document model for the held-paper preview and the full, quotable reader.
// These are illustrative studio notes, never invented publication metadata.
export function workingNote(studio) {
  return {
    title: studio.title,
    eyebrow: "STUDIO DEMONSTRATION / NOTE 001",
    disclosure: "Illustrative working note. This is not a published paper or a research result.",
    passages: [
      { title: "The question", text: studio.description },
      {
        title: "The assumption",
        text: "Write down the conditions under which the idea should work. Which variable changes? Which control stays fixed? What evidence would distinguish the explanations?",
      },
      {
        title: "The next small test",
        text: "Start with a bounded comparison, record both expected and unexpected outcomes, and return to the question before expanding the work.",
      },
    ],
  };
}

// The physical sheet is a preview. Overflow is visible as an ellipsis, while
// the reader retains the complete source. Long tokens split at code points.
export function noteLines(text, width, measure, limit) {
  const lines = [];
  const shortened = () => {
    const kept = lines.slice(0, limit);
    let last = Array.from(kept.at(-1));
    while (last.length && measure(last.join("") + "…") > width) last.pop();
    kept[kept.length - 1] = last.join("") + "…";
    return kept;
  };
  let line = "";
  for (const word of String(text ?? "").trim().split(/\s+/)) {
    const joined = line ? `${line} ${word}` : word;
    if (measure(joined) <= width) { line = joined; continue; }
    if (line) {
      lines.push(line); line = "";
      if (lines.length === limit) return shortened();
    }
    for (const character of word) {
      if (line && measure(line + character) > width) {
        lines.push(line); line = "";
        if (lines.length === limit) return shortened();
      }
      line += character;
    }
  }
  if (line) lines.push(line);
  return lines;
}

export const NOTE_CANVAS = Object.freeze({ width: 768, height: 1008 });

// Authored print layout, in canvas pixels. Text measurement is injected so the
// same bounded layout is tested without a browser or platform-specific fonts.
export function workingNoteLayout(note, measure) {
  const rows = [];
  const block = (text, y, font, lineHeight, limit, color = "#29443a", width = 640) => {
    const lines = noteLines(text, width, value => measure(value, font), limit);
    lines.forEach((value, index) => rows.push({ text: value, x: 64, y: y + index * lineHeight, font, color }));
  };
  block(note.eyebrow, 52, "20px Arial", 26, 1, "#596d60");
  block(note.title, 130, "48px Georgia", 52, 3);
  for (const [i, passage] of note.passages.entries()) {
    const y = [292, 470, 682][i];
    block(`0${i + 1} / ${passage.title}`, y, "bold 25px Arial", 30, 1);
    // The lower-right margin is the physical thumb grip, not a text region.
    block(passage.text, y + 44, "27px Georgia", 35, [3, 4, 3][i], "#29443a", i === 2 ? 540 : 640);
  }
  block(note.disclosure, 915, "22px Arial", 28, 3, "#596d60");
  return rows;
}
