# Demo walkthrough

This takes about five minutes. The four examples use one photo of a shirt. Each second image was edited on purpose to show how DriftCheck responds. They are not real AI outputs and not Scalio outputs. Say this when you start.

## Before you start

1. Open the Ollama app. Check it with `.venv\Scripts\driftcheck ping`.
2. In PowerShell, in the project folder, run `.venv\Scripts\driftcheck demo`.
3. The browser opens http://localhost:7860.
4. Point at the note at the top of the page. It names the vision model, qwen2.5vl:7b on local Ollama, and says that colour and pattern are skipped on scene photos.

The first run takes about a minute while the models load. After that the examples are quick.

## 1. An unchanged copy

1. Under Try an example, click Unchanged copy of the shirt.
2. Press Run DriftCheck.
3. Point at the verdict. It says Looks fine and no check fired.
4. Point at the check table. Product present, colour, pattern and shape say fine. Text and logo say not applicable, because this shirt has no text or logo to compare.

What to say: nothing changed, and DriftCheck agrees.

## 2. A colour change

1. Click Colour edited on purpose: black shirt made blue.
2. Press Run DriftCheck.
3. Point at the verdict. It says Changed and names Colour.
4. Point at the colour row. It says how far the colours moved and which colour became which.
5. Point at the colour strip under the heatmap. Each original colour sits next to the blue it became.

What to say: each check looks at one thing, so the report says what changed, not just that something changed.

## 3. Added text, a miss

1. Click Text edited on purpose: the word SALE added.
2. Press Run DriftCheck.
3. Point at the verdict. It says Looks fine. This is a miss.
4. Point at the text row. It says not applicable, because the original shirt has no text. The text check only looks for text that was on the original, so it never looks at new text.

What to say: this is a known gap, and checking for added text is the next fix.

## 4. Added stripes, a miss

1. Click Pattern edited on purpose: stripes added.
2. Press Run DriftCheck.
3. Point at the heatmap. The red patches show where the pattern check saw a difference.
4. Point at the pattern row. It says 22 percent of the shirt differs, but the check only flags above 90 percent at its current setting, so the verdict is Looks fine. This is a miss.

What to say: the check sees the change, but its setting is too loose. It was tuned that way because worn and scene photos made it flag good images. Now that those photos skip the pattern check, the setting can be tightened.

## If someone asks for more

1. Scene photos: change the image type dropdown to scene and run an example again. Colour and pattern now say not applicable, and the verdict rests on the checks that ran.
2. The full numbers, including the weak parts, are in README.md and reports\eval_report.md.
