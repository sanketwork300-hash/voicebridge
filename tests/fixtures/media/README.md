# tests/fixtures/media

No audio or video is committed to the repository (licensing and size).
Build the evaluation media locally:

    python -m voicebridge.benchmark.datasets fleurs --source ja --target en -n 20 --out benchmark/datasets/ja_en.json
    python scripts/fetch_nonspeech.py        # laughter, crying, applause, singing (Wikimedia Commons, attributed)
    python scripts/make_test_media.py --lang ja_jp --count 5   # speech + music + noise episode (wav/mp3/mp4)

Coverage and measured results: docs/evaluation.md. Categories not yet covered
by real data: overlapping speech, background dialogue, real anime/drama audio.
