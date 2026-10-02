// Evaluates FaceKit's full pipeline (Vision detection -> alignment -> Core ML embedding)
// on the LFW verification protocol and reports the numbers needed to pick MatchThresholds.
//
// Usage: swift run -c release FaceKitEval <lfw image dir> <pairs.txt>
// LFW: http://vis-www.cs.umass.edu/lfw/ (mirrored by scikit-learn on figshare).

import CoreImage
import FaceKit
import Foundation

struct Pair {
    let a: String
    let b: String
    let same: Bool
    let fold: Int
}

struct Config {
    let name: String
    let aligner: FaceAligner
}

func parsePairs(_ url: URL) throws -> [Pair] {
    let rows = try String(contentsOf: url, encoding: .utf8)
        .split(whereSeparator: \.isNewline)
        .map { $0.split(separator: "\t").map(String.init) }
    let folds = Int(rows[0][0])!, perFold = Int(rows[0][1])!
    func path(_ name: String, _ index: String) -> String {
        "\(name)/\(name)_\(String(format: "%04d", Int(index)!)).jpg"
    }
    var pairs: [Pair] = []
    var row = 1
    for fold in 0..<folds {
        for _ in 0..<perFold {
            let r = rows[row]; row += 1
            pairs.append(Pair(a: path(r[0], r[1]), b: path(r[0], r[2]), same: true, fold: fold))
        }
        for _ in 0..<perFold {
            let r = rows[row]; row += 1
            pairs.append(Pair(a: path(r[0], r[1]), b: path(r[2], r[3]), same: false, fold: fold))
        }
    }
    return pairs
}

/// LFW images are centred on the subject, so take the detected face nearest the centre.
/// If Vision finds nothing, fall back to a fixed central box (counted and reported).
func subjectFace(in image: CIImage, detector: FaceDetector) -> (face: DetectedFace, detected: Bool) {
    let faces = (try? detector.detect(in: image)) ?? []
    let centre = CGPoint(x: 0.5, y: 0.5)
    if let face = faces.min(by: { hypot($0.boundingBox.midX - centre.x, $0.boundingBox.midY - centre.y)
                                   < hypot($1.boundingBox.midX - centre.x, $1.boundingBox.midY - centre.y) }) {
        return (face, true)
    }
    return (DetectedFace(boundingBox: CGRect(x: 0.32, y: 0.3, width: 0.36, height: 0.42)), false)
}

func embedAll(_ paths: [String], root: URL, configs: [Config]) throws -> (embeddings: [[String: FaceEmbedding]], misses: Int) {
    let workers = max(ProcessInfo.processInfo.activeProcessorCount - 2, 1)
    var results = Array(repeating: [String: FaceEmbedding](), count: configs.count)
    var misses = 0
    var firstError: Error?
    let lock = NSLock()
    var done = 0

    DispatchQueue.concurrentPerform(iterations: workers) { worker in
        do {
            let detector = FaceDetector()
            let renderer = FaceRenderer()
            let embedder = try FaceEmbedder()
            for index in stride(from: worker, to: paths.count, by: workers) {
                let path = paths[index]
                guard let image = CIImage(contentsOf: root.appendingPathComponent(path)) else {
                    throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: path])
                }
                let (face, detected) = subjectFace(in: image, detector: detector)
                var local: [FaceEmbedding] = []
                for config in configs {
                    guard let crop = renderer.cgImage(from: config.aligner.align(face, in: image)) else {
                        throw FaceKitError.invalidModelOutput
                    }
                    local.append(try embedder.embedding(for: crop))
                }
                lock.lock()
                for (c, e) in local.enumerated() { results[c][path] = e }
                if !detected { misses += 1 }
                done += 1
                if done % 1000 == 0 { print("  embedded \(done)/\(paths.count)") }
                lock.unlock()
            }
        } catch {
            lock.lock(); firstError = firstError ?? error; lock.unlock()
        }
    }
    if let firstError { throw firstError }
    return (results, misses)
}

struct Report {
    let accuracy: (mean: Double, std: Double, threshold: Double)
    let eer: (rate: Double, threshold: Double)
    let atFAR: [(far: Double, threshold: Double, tar: Double)]
    let atThreshold: [(threshold: Double, far: Double, tar: Double)]
    let genuine: [Double]
    let impostor: [Double]
}

func rate(_ values: [Double], below threshold: Double) -> Double {
    Double(values.filter { $0 <= threshold }.count) / Double(values.count)
}

func evaluate(_ pairs: [Pair], distances: [Double]) -> Report {
    let grid = stride(from: 0.0, through: 2.0, by: 0.005).map { $0 }
    let folds = Set(pairs.map(\.fold)).sorted()

    func accuracy(_ indices: [Int], _ t: Double) -> Double {
        Double(indices.filter { (distances[$0] <= t) == pairs[$0].same }.count) / Double(indices.count)
    }
    // Standard LFW protocol: choose the threshold on 9 folds, test on the 10th.
    var accs: [Double] = [], chosen: [Double] = []
    for fold in folds {
        let train = pairs.indices.filter { pairs[$0].fold != fold }
        let test = pairs.indices.filter { pairs[$0].fold == fold }
        let t = grid.max { accuracy(train, $0) < accuracy(train, $1) }!
        chosen.append(t)
        accs.append(accuracy(test, t))
    }
    let mean = accs.reduce(0, +) / Double(accs.count)
    let std = (accs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(accs.count)).squareRoot()

    let genuine = pairs.indices.filter { pairs[$0].same }.map { distances[$0] }.sorted()
    let impostor = pairs.indices.filter { !pairs[$0].same }.map { distances[$0] }.sorted()

    let eerT = grid.min { abs(1 - rate(genuine, below: $0) - rate(impostor, below: $0))
                        < abs(1 - rate(genuine, below: $1) - rate(impostor, below: $1)) }!
    let eer = (rate(impostor, below: eerT) + 1 - rate(genuine, below: eerT)) / 2

    // Largest threshold whose false-accept rate does not exceed the target.
    let atFAR = [0.01, 0.001].map { far -> (Double, Double, Double) in
        let allowed = Int((far * Double(impostor.count)).rounded(.down))
        let t = impostor[allowed].nextDown
        return (far, t, rate(genuine, below: t))
    }
    let atThreshold = [0.4, 0.7, 0.9, 1.0, 1.1].map { t in (t, rate(impostor, below: t), rate(genuine, below: t)) }

    return Report(accuracy: (mean, std, chosen.reduce(0, +) / Double(chosen.count)),
                  eer: (eer, eerT), atFAR: atFAR, atThreshold: atThreshold,
                  genuine: genuine, impostor: impostor)
}

func percentile(_ sorted: [Double], _ p: Double) -> Double {
    sorted[min(Int(p * Double(sorted.count)), sorted.count - 1)]
}

func f(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", x) }

setvbuf(stdout, nil, _IOLBF, 0) // show progress when piped to a file
let args = CommandLine.arguments
guard args.count == 3 else {
    print("usage: FaceKitEval <lfw image dir> <pairs.txt>")
    exit(2)
}
let root = URL(fileURLWithPath: args[1])
let pairs = try parsePairs(URL(fileURLWithPath: args[2]))
let paths = Array(Set(pairs.flatMap { [$0.a, $0.b] })).sorted()

let configs = [
    Config(name: "margin 0.0, levelled", aligner: FaceAligner(margin: 0.0)),
    Config(name: "margin 0.05, levelled", aligner: FaceAligner(margin: 0.05)),
    Config(name: "margin 0.1, levelled", aligner: FaceAligner(margin: 0.1)),
    Config(name: "margin 0.15, levelled", aligner: FaceAligner(margin: 0.15)),
    Config(name: "margin 0.2, levelled", aligner: FaceAligner(margin: 0.2)),
    Config(name: "margin 0.1, not levelled", aligner: FaceAligner(margin: 0.1, correctsRoll: false)),
]

print("LFW: \(pairs.count) pairs, \(paths.count) images, configs: \(configs.count)")
let start = Date()
let (embeddings, misses) = try embedAll(paths, root: root, configs: configs)
let seconds = Date().timeIntervalSince(start)
print("Embedded in \(f(seconds, 1)) s (\(f(seconds / Double(paths.count * configs.count) * 1000, 1)) ms per crop, parallel); "
      + "no face detected in \(misses) images (central-box fallback)\n")

for (c, config) in configs.enumerated() {
    let distances = pairs.map { Double(embeddings[c][$0.a]!.distance(to: embeddings[c][$0.b]!)) }
    let r = evaluate(pairs, distances: distances)
    print("== \(config.name)")
    print("  10-fold accuracy \(f(r.accuracy.mean * 100, 2))% ± \(f(r.accuracy.std * 100, 2)), best threshold ≈ \(f(r.accuracy.threshold))")
    print("  EER \(f(r.eer.rate * 100, 2))% at \(f(r.eer.threshold))")
    for p in r.atFAR {
        print("  FAR ≤ \(f(p.far * 100, 1))%: threshold \(f(p.threshold)), TAR \(f(p.tar * 100, 2))%")
    }
    for p in r.atThreshold {
        print("  threshold \(f(p.threshold, 1)): FAR \(f(p.far * 100, 2))%, TAR \(f(p.tar * 100, 2))%")
    }
    print("  genuine p50/p95/p99 \(f(percentile(r.genuine, 0.5)))/\(f(percentile(r.genuine, 0.95)))/\(f(percentile(r.genuine, 0.99)))"
          + "  impostor p1/p5/p50 \(f(percentile(r.impostor, 0.01)))/\(f(percentile(r.impostor, 0.05)))/\(f(percentile(r.impostor, 0.5)))\n")
}
