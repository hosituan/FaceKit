# FaceKit

On-device face enrollment and identification for iOS 15+ / macOS 12+, as a Swift package.
Extracted from the `clockon-clockoff-face-recognition` attendance app and rewritten.
The story behind it: [Five years later: rebuilding my iOS FaceNet app as an open-source Swift package](https://hosituan.medium.com/five-years-later-rebuilding-my-ios-facenet-app-as-an-open-source-swift-package-447a99848d1e).

- **Detection & alignment**: Vision face landmarks, crop levelled on the eyes.
- **Embedding**: FaceNet (InceptionResNetV1, 128-d) converted to Core ML, fp16 weights (44 MB).
  Output matches the original TensorFlow graph (cosine > 0.9999).
- **Matching**: nearest neighbour against enrolled templates with two distance thresholds.
- **Enrollment**: augmentation (mirror, ±9° rotation) and spherical k-means compression.
- **Storage**: `FaceStore` protocol; in-memory and AES-GCM encrypted file stores included.
- No third-party dependencies; nothing leaves the device.

## Installation

```swift
.package(url: "https://github.com/hosituan/FaceKit.git", from: "0.1.0")
```

Products: `FaceKit` (everything) or `FaceKitCore` (matching/clustering/storage only, no model).

## Usage

```swift
import FaceKit

let key = SymmetricKey(size: .bits256)          // keep it in the Keychain
let recognizer = try await FaceRecognizer(
    store: FileFaceStore(url: storeURL, key: key))

// Enroll: several photos or video frames of one person.
try await recognizer.enroll(id: "emp_042", images: frames)

// Identify every face in a camera frame.
var consensus = FrameConsensus(requiredFrames: 5, windowSize: 8)
for r in try await recognizer.identify(in: pixelBuffer, orientation: .leftMirrored) {
    switch r.match.decision {
    case .confident, .candidate:
        if let id = consensus.observe(r.match.identityID) { /* confirmed: log attendance */ }
    case .unknown:
        _ = consensus.observe(nil)
    }
}
```

`MatchResult` exposes the real L2 distance (`0...2`, lower is closer) and the runner-up
distance; there is no synthetic "confidence %".

## Example app

`Examples/FaceKitDemo` is a SwiftUI app using the package from this working copy: live
front-camera recognition with a box and name per face (green = confident, yellow = candidate,
red = unknown), enrollment by name from ~3 s of camera frames, and a list to delete people.

Open `Examples/FaceKitDemo/FaceKitDemo.xcodeproj`, choose your team under Signing, and run it on
an iPhone or iPad (the simulator has no camera).

## Accuracy and thresholds

Measured with the full pipeline (Vision → alignment → Core ML) on the LFW verification
protocol, 6,000 pairs, default `FaceAligner` (margin 0.1, eyes levelled):

| Metric | Value |
|---|---|
| 10-fold accuracy | 99.17% ± 0.51 |
| Equal error rate | 0.83% (distance 1.13) |
| TAR at FAR ≤ 1% / 0.1% | 99.23% / 96.90% |
| Distance 0.9 (`confident` default) | FAR 0.00% (0/3000), TAR 88.63% |
| Distance 1.0 (`candidate` default) | FAR 0.03%, TAR 95.97% |
| Throughput (M1 Max, parallel, macOS) | 2.7 ms per crop |

Margin sweep: 0.0 → 98.95%, 0.05 → 99.22%, 0.1 → 99.17%, 0.15 → 99.05%, 0.2 → 98.90%;
without eye levelling (margin 0.1) 98.93%.

Caveats:
- These are 1:1 rates. In 1:N identification a stranger is compared with every enrolled
  person, so the false-accept chance grows roughly with N. Require `FrameConsensus` for
  `.candidate` matches and re-measure on your own users and camera.
- LFW is celebrity photos; the model's training data (see below) may overlap LFW identities,
  so results can be optimistic. A front-facing kiosk camera is a different distribution.
- The original app used 0.4 / 0.7; on LFW those accept only 3% / 53% of genuine pairs.

Reproduce:

```sh
cd Tools/FaceKitEval
swift run -c release FaceKitEval <lfw image dir> <pairs.txt>
```

## Limitations

- **No liveness / anti-spoofing.** A printed photo or a video of an enrolled person will match.
  Do not use this alone for security-sensitive decisions.
- Templates are biometric data, treated as sensitive personal data by laws such as the EU GDPR
  (Article 9) and Vietnam's personal data protection rules; check the rules that apply to you.
  Obtain consent, encrypt at rest (`FileFaceStore(key:)`), and offer deletion (`remove(id:)`).

## Licence

FaceKit's code is released under the [MIT License](LICENSE). The bundled model is covered by
[NOTICE](NOTICE) and the section below.

## Model provenance and licence

The FaceNet code by David Sandberg is MIT-licensed. The bundled weights produce 128-d
embeddings, while the pretrained models FaceNet lists today (2018, trained on CASIA-WebFace and
VGGFace2) produce 512-d, so these come from an earlier FaceNet release. The public face datasets
those models were trained on carry research-only or withdrawn terms. **Check the weights'
licence before commercial use**, or swap in a model you have rights to.

## Development

```sh
swift test                                         # unit + model parity tests
FACEKIT_FACES_DIR=~/faces swift test               # + end-to-end on local photos (one folder per person)
```

Regenerating the model (needs the original `modelFacenet.pb`):

```sh
python3.11 -m venv venv && venv/bin/pip install -r Tools/requirements.txt
venv/bin/python Tools/convert_model.py modelFacenet.pb out/ [sample images...]
xcrun coremlcompiler compile out/FaceNet.mlpackage Sources/FaceKitCoreML/Resources/
```

The model ships precompiled (`.mlmodelc`) so it works with both Xcode and `swift build`.
