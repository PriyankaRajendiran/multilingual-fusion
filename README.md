# Multilingual Semantic Fusion Framework — Stage 1 (Dataset)

This is the first delivery per your Section 28 priority: folder structure,
`requirements.txt`, `config.py`, `prepare_dataset.py`, and a minimal test —
nothing else has been built yet (no CLIP/BLIP-2/XLM-R/IndicBERT code, no
FAISS, no fusion modules). That comes next, once you confirm this stage
works on your machine.

## What was actually verified (not just written)

I ran all three scripts for real against the **official** multi30k/dataset
GitHub repository (not a mock):

```
python prepare_dataset.py    # -> downloads real captions, builds dataset/captions.csv
python translate_captions.py # -> builds dataset/multilingual_captions.csv
python test_dataset.py       # -> sanity-checks both CSVs
```

`prepare_dataset.py` genuinely fetched the first 100 English captions and
their matching image filenames from `multi30k/dataset` (task1, train split)
and wrote `dataset/captions.csv`. It correctly reported that 0/100 image
*files* exist locally — because Multi30K/Flickr30K images are not
redistributable, so you must add them yourself (see "Getting the images"
below). This is the honest behavior your spec asked for: no fabrication,
no silent substitution.

`translate_captions.py` failed to reach `translate.google.com` in the
environment I built this in (its network is restricted to package
registries and GitHub only). It correctly reported the failures and left
every Tamil/Hindi cell blank rather than inventing text — again, per your
"do not fabricate" requirement. **On your own machine, with normal
internet access, this will translate successfully.** Re-run it after
placing your images; it resumes from wherever it left off.

## 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the 100 images

Multi30K's captions/splits are open (hosted on GitHub, downloaded
automatically by `prepare_dataset.py`), but the underlying **photographs**
belong to Flickr30K, which requires you to request access directly:
- Flickr30K request form: https://forms.illinois.edu/sec/229675
- Multi30K repo (captions/splits only, no images): https://github.com/multi30k/dataset

Once you have the Flickr30K images, run `prepare_dataset.py` first — it
will print the exact 100 filenames it selected (e.g. `1000092795.jpg`) —
then copy just those 100 files into `dataset/images/`.

## 3. Run

```bash
python prepare_dataset.py      # builds dataset/captions.csv (100 English rows)
python translate_captions.py   # fuses in Tamil + Hindi -> dataset/multilingual_captions.csv
python test_dataset.py         # verifies both files are structurally correct
```

Re-running any script is safe: `prepare_dataset.py` caches the small
downloaded text files in `dataset/_raw_cache/`, and `translate_captions.py`
checkpoints every 10 rows and only re-translates blanks.

## 4. Scaling later

Change one line in `config.py`:

```python
NUM_IMAGES = 500   # was 100
```

Nothing else changes — `prepare_dataset.py` will select 500 pairs instead
(as long as `NUM_IMAGES <= 29000`, the size of the train split), and every
downstream script keys off `config.NUM_IMAGES` / the resulting CSVs.

## 5. Next stage

Once `test_dataset.py` passes on your machine with real images in place,
the next delivery is: `extract_image_features.py` (CLIP), `generate_captions.py`
(BLIP-2, optional/MODE 1 by default), `extract_text_features.py` (XLM-R +
IndicBERT), `fusion.py`, `build_faiss_index.py`, then the two retrieval
scripts — each independently runnable, as you specified.
