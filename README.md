# FaceKit

On-device face enrollment and identification for iOS 15+ / macOS 12+, as a Swift package.
Extracted from the `clockon-clockoff-face-recognition` attendance app and rewritten.

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

## Thresholds

Defaults (`confident: 0.4`, `candidate: 0.7`) come from the original app and are **not
calibrated**. Measure false accept / false reject rates on data representative of your users
and set `MatchThresholds` accordingly.

## Limitations

- **No liveness / anti-spoofing.** A printed photo or a video of an enrolled person will match.
  Do not use this alone for security-sensitive decisions.
- Templates are biometric data (GDPR art. 9, Vietnam Decree 13/2023, BIPA...). Obtain consent,
  encrypt at rest (`FileFaceStore(key:)`), and offer deletion (`remove(id:)`).

## Model provenance and licence

The FaceNet code by David Sandberg is MIT-licensed. The bundled weights produce 128-d
embeddings, which matches the 2017 pretrained FaceNet releases (trained on CASIA-WebFace or
MS-Celeb-1M). Those datasets carry research-only / withdrawn terms. **Check the weights'
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
