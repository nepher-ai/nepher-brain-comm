# GR00T N1.7 brain image

Built once by Nepher and reused by every tournament whose model is GR00T N1.7. The image contains the runtime and the architecture check. It does not contain model weights. Participants mount a submission at `/submission`.

Before publishing, seal `reference_signature.json` from the base checkpoint:

```
python -c "from gr00t_n17_check import seal_from_directory; from pathlib import Path; seal_from_directory(Path('weights'), Path('reference_signature.json'))"
```

`sealed` must be true. An unsealed reference rejects every submission.

```
bash images/gr00t-n17/build_and_push.sh registry.example/nepher-brain-gr00t-n17:phase
```

The script prints `repo@sha256:...`. That digest is the phase config's `brain_image`.
