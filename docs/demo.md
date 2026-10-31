# Demo walkthrough

This takes about five minutes. The three examples come from the test half of this dataset. The two changed examples are copies that were changed on purpose, because none of the real Gemini outputs was labelled as changed.

## Before you start

1. Open the Ollama app. Check it with `.venv\Scripts\driftcheck ping`.
2. In PowerShell, in the project folder, run `.venv\Scripts\driftcheck demo`.
3. The browser opens http://localhost:7860.
4. Point at the note at the top of the page. It names the vision model, qwen2.5vl:7b on local Ollama, and says that colour and pattern are skipped on scene photos.

The first run takes about a minute while the models load. After that the examples are quick.

## 1. A clean pass

1. Under Try an example from the test set, click Clean pass: white top, Gemini studio shot.
2. Press Run DriftCheck.
3. Point at the verdict. It says Looks fine and no check fired.
4. Point at the check table. Colour, pattern, shape and product present say fine, each with a reason. Text and logo say not applicable, because the original top has no text or logo.
5. Point at the heatmap. There are no red patches on the output, so the pattern matches.

What to say: Gemini kept this product, and DriftCheck agrees.

## 2. A colour change

1. Click Colour changed: NEYMAR jersey, colour shifted on purpose.
2. Press Run DriftCheck.
3. Point at the verdict. It says Changed and names Colour as the check that fired.
4. Point at the colour row in the check table. It says how far the colours moved and which colour became which.
5. Point at the two text boxes. Both show the same words read from the jersey, the player name, RESPECT and unicef, so the text check says fine. Only the colour changed, and only the colour check fired.
6. Point at the colour strip under the heatmap. Each original colour sits next to the colour it became.

What to say: each check looks at one thing, so the report says what changed, not just that something changed.

## 3. A pattern change

1. Click Pattern changed: striped tee, pattern shuffled on purpose.
2. Press Run DriftCheck.
3. Point at the verdict. It says Changed and names Pattern.
4. Point at the heatmap. The red patches on the output show where the print no longer matches the original.
5. Point at the colour row. It says fine, because the same colours are still there, just moved around.

What to say: colour alone would miss this. The pattern check catches it.

## If someone asks about weak spots

1. Scene photos: change the image type dropdown to scene and run an example again. Colour and pattern now say not applicable, and the verdict rests on the checks that ran.
2. The text check passes garbled words at its current threshold. README.md explains this and the planned fix.
3. The full numbers, including the weak parts, are in README.md and reports\eval_report.md.
