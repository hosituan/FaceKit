"""Convert the FaceNet frozen graph (TF1) to Core ML and verify it against TensorFlow.

Usage: python convert_model.py <modelFacenet.pb> <out_dir> [image ...]

Produces FaceNet.mlpackage (fp16) and FaceNetFP32.mlpackage. The Core ML model takes a
160x160 RGB image and applies the same normalisation as the original app,
(x - 128) / 128, so callers only need to hand it an aligned face crop.
"""
import os
import sys

import coremltools as ct
import numpy as np
import tensorflow as tf
from PIL import Image

SIZE = 160


def load_graph(path):
    graph_def = tf.compat.v1.GraphDef()
    with open(path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def


def tf_embed(graph_def, batch):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    with tf.compat.v1.Session(graph=graph) as sess:
        return sess.run("embeddings:0", {"input:0": (batch - 128.0) / 128.0})


def convert(graph_def, precision):
    model = ct.convert(
        graph_def,
        source="tensorflow",
        inputs=[ct.ImageType(name="input", shape=(1, SIZE, SIZE, 3),
                             scale=1 / 128.0, bias=[-1.0, -1.0, -1.0],
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embeddings")],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=precision,
        convert_to="mlprogram",
    )
    model.short_description = "FaceNet (InceptionResNetV1) face embedding, 128-d, L2-normalised."
    model.input_description["input"] = "Aligned RGB face crop, 160x160."
    model.output_description["embeddings"] = "128-d L2-normalised embedding."
    model.author = "FaceNet: David Sandberg (MIT). Converted for FaceKit."
    return model


def sample_images(paths, rng):
    images = [np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
              for p in paths]
    images += [rng.uniform(0, 255, (SIZE, SIZE, 3)).astype(np.float32) for _ in range(4)]
    return images


def main():
    pb, out_dir, image_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    graph_def = load_graph(pb)
    images = sample_images(image_paths, np.random.default_rng(0))
    reference = tf_embed(graph_def, np.stack(images))

    for name, precision in [("FaceNetFP32", ct.precision.FLOAT32), ("FaceNet", ct.precision.FLOAT16)]:
        model = convert(graph_def, precision)
        path = os.path.join(out_dir, f"{name}.mlpackage")
        model.save(path)
        cosines = []
        for img, ref in zip(images, reference):
            out = model.predict({"input": Image.fromarray(img.astype(np.uint8))})["embeddings"].reshape(-1)
            cosines.append(float(np.dot(out, ref) / (np.linalg.norm(out) * np.linalg.norm(ref))))
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        print(f"{name}: {size / 1e6:.1f} MB, cosine vs TF min={min(cosines):.5f} mean={np.mean(cosines):.5f}")

    # Pairwise distances must be preserved too, not just per-sample direction.
    fp16 = ct.models.MLModel(os.path.join(out_dir, "FaceNet.mlpackage"))
    outs = np.stack([fp16.predict({"input": Image.fromarray(i.astype(np.uint8))})["embeddings"].reshape(-1)
                     for i in images])
    d_tf = np.linalg.norm(reference[:, None] - reference[None], axis=-1)
    d_ml = np.linalg.norm(outs[:, None] - outs[None], axis=-1)
    print(f"max |pairwise L2 distance diff| fp16 vs TF: {np.abs(d_tf - d_ml).max():.5f}")


if __name__ == "__main__":
    main()
