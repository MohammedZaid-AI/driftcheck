# How to make the AI outputs

DriftCheck needs 3 AI outputs per product: model_shot, studio and lifestyle. There is no API image generation. Use one of two routes. Both end with driftcheck register.

## Naming rule

Every output is saved as:

```
data\products\<product_id>\outputs\<style>__<generator>.jpg
```

Style is model_shot, studio or lifestyle. Generator is qwen_edit for the Kaggle route and gemini_app for the manual route. Example: data\products\p003\outputs\studio__gemini_app.jpg

PNG and WEBP are fine. register turns them into JPG.

## Route 1: Kaggle with Qwen Image Edit

```
.venv\Scripts\driftcheck jobs --generator qwen_edit
```

This writes data\jobs\qwen_edit\kaggle_input.zip. Then follow the steps at the top of notebooks\kaggle_qwen_image_edit.ipynb. When you have downloaded qwen_edit_outputs.zip from Kaggle:

```
Expand-Archive qwen_edit_outputs.zip -DestinationPath data\jobs\qwen_edit\result
.venv\Scripts\driftcheck register --from data\jobs\qwen_edit\result
```

The notebook writes its own json next to each image with the model name, steps, seed and time.

## Route 2: by hand in the free Gemini app

```
.venv\Scripts\driftcheck jobs --generator gemini_app
```

Open data\jobs\gemini_app\manual_checklist.md. For each item:
1. Start a new chat in the Gemini app so earlier images do not leak in.
2. Upload the original shown in the checklist.
3. Paste the prompt.
4. Download the image and save it with the exact Save as path.

Then:

```
.venv\Scripts\driftcheck register
```

register writes the json with the prompt, generator gemini_app, model Gemini app, free tier, and the file date. To record a more exact model name, add --model.

## Check

register lists any file that breaks the naming rule. Rename it and run register again.
