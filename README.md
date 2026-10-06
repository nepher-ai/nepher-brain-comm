# nepher-brain

Lockstep protocol between a tournament submission and the simulator.

The library is model-agnostic. It defines the messages (`hello`, `begin_episode`, `act`, `end_episode`, `close`, `error`), the `Brain` class, and the image commands `nepher-brain serve`, `nepher-brain check`, and `nepher-brain smoke`.

A submission is a directory:

```
agent.yaml
brain/          # pure Python, importable from this directory
weights/        # files whose sha256 is listed in agent.yaml
train/          # stored for review, never executed
```

`agent.yaml` names the entry class (`module:Class`), the action horizon, the action dimension, and the weight hashes. The brain image may set `NEPHER_BRAIN_CHECK=module:function` to add a model check. That function returns an error string, or `None` when the submission is acceptable.

One image is built per model family, in the `nepher-brain-images` repository. Validators pull that image by digest and mount the submission read-only at `/submission`. They do not build an image per submission.

```
nepher-brain serve --submission ./examples/zero_brain --socket-dir /tmp/brain --replicas 1
nepher-brain check --submission ./examples/zero_brain
nepher-brain smoke --socket /tmp/brain/brain-0.sock
```
