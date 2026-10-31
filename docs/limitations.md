# Limitations

This file lists where DriftCheck is known to fail. It grows as the evaluation finds more.

## Making outputs

### Qwen Image Edit did not run on free Kaggle GPUs

The Kaggle notebook loads the 4 bit build of Qwen Image Edit 2511 across both T4 GPUs. It used 11.8 GB on GPU 0 and 5.1 GB on GPU 1. Every generation then ran out of memory on GPU 0, even at 768 by 768 pixels. No outputs were made this way. The notebook is kept as it is in notebooks/kaggle_qwen_image_edit.ipynb.

All outputs for the evaluation come from the free Gemini app instead, made by hand. So the results describe one generator only.

## False alarms on real AI images

On 40 Gemini test images that were labelled as unchanged, the first version of DriftCheck flagged 27 as drifted. A big cause was the cut out: in model shots it kept the whole person, so colour and pattern compared the garment with the model's face, skin and other clothes.

A clothes parsing model now keeps only the garment in worn photos, and the thresholds were retuned so that at most 10 percent of the real calibration images are flagged per check. After both changes, 20 of the 40 real test images are still flagged. The same garment worn on a body still looks different from the flat photo to the pattern check, and the logo and shape checks, which use the vision model, flag some good images too. The changes also lowered how many synthetic changes are caught, from 80.8 to 60.6 percent.

Colour and pattern are now skipped on scene images, such as model shots, lifestyle scenes and posters, because the same product looks different on a body or in a scene. With that, and a fix to how the evaluation decides text so it matches the live check, 11 of the 40 real test images are still flagged. The rest of the false alarms come mostly from the logo and shape checks.

The text check now passes letter swaps. Its tuned threshold is a match score of 40, so BROOKLYN read as UROKKLYH still counts as the same text. In the test split it caught none of the synthetic text replacements on its own. A colour or pattern change in a scene image is now only caught if another check notices it.

All of these changes were made after the first test results were seen.

## Speed and memory

On a laptop with 12 GB of RAM and a 6 GB GPU, one image takes a median of 51 seconds with qwen2.5vl:7b on Ollama, and up to 146 seconds for the slowest 10 percent. With the 7b model loaded, easyocr crashed the Python process 11 times in 406 images with a Windows access violation. The evaluation restarted itself after each crash.

## Text

### Large stylised titles can be misread

On p045, the Rasoi roast masala box, the big Bengali title রসুই is not read correctly. The normal pass reads it as রস্ুই at 0.3 confidence. A second pass on the title area at twice the size reads বসুই at 0.43 confidence. That is one letter wrong. The smaller Bengali words on the same box are read well, at 0.80 to 0.94 confidence.

Because the original and the output go through the same OCR, the same misreading on both sides still matches. A real change to the title can be missed if OCR misreads the output in a way that happens to match.

### Script detection is a first guess

The scripts on each product are detected once from the original and saved as text_scripts in meta.json. On the 40 core products the detection was wrong for 5 of them, all with Hindi or other Devanagari text: p017, p043, p049, p057 and p068. easyocr reads real Hindi on these photos at only 0.36 to 0.57 confidence, under the 0.7 needed to count. On p049 the Bengali reader also read the Hindi as made up Bengali at 0.74 confidence. These 5 were set by hand in config/selection.json.

New products need the same check by eye before their text results can be trusted.
