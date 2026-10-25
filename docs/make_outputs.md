# How to make the AI outputs

DriftCheck needs AI outputs of the core products. There is no API image generation. The outputs are made by hand in the free Gemini app.

## Steps

```
.venv\Scripts\driftcheck jobs
```

This writes data\jobs\gemini_app\manual_checklist.md with one line per image. Each line has the photo to upload, the prompt to paste and the file name to save. There are two images per product, so 80 for the 40 core products.

For each line:
1. Start a new chat in the Gemini app, so earlier images do not leak in.
2. Upload the photo.
3. Paste the prompt.
4. Download the image and save it with the exact name. A .png with the same name is fine.

Then:

```
.venv\Scripts\driftcheck register
```

register writes a json next to each image with the prompt, generator gemini_app and the file date. It lists any file that breaks the naming rule. Rename it and run register again.

## Naming rule

```
data\products\<product_id>\outputs\<style>__<generator>.jpg
```

The styles are model_shot and studio for clothes and accessories, lifestyle and studio for other goods, and festival_poster and instagram_post for shop fronts. driftcheck jobs --all-styles adds a third style for each.

## Kaggle route

notebooks\kaggle_qwen_image_edit.ipynb runs Qwen Image Edit with driftcheck jobs --generator qwen_edit. It ran out of GPU memory on free Kaggle T4s, see docs\limitations.md.
