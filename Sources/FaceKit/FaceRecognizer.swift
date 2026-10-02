@_exported import FaceKitCore
@_exported import FaceKitCoreML
@_exported import FaceKitVision

import CoreImage
import CoreML
import CoreVideo
import ImageIO

/// One face found in an image and what it matched.
public struct Recognition: Hashable, Sendable {
    public let face: DetectedFace
    public let match: MatchResult
}

public struct EnrollmentOptions: Sendable {
    public enum Mode: Sendable {
        /// Discard any templates previously enrolled under the same id.
        case replace
        /// Merge with the existing templates before compressing.
        case append
    }

    public var mode: Mode
    /// Add mirrored and slightly rotated copies of every face.
    public var augment: Bool
    /// Compress embeddings into at most this many templates with k-means; `nil` keeps them all.
    public var templatesPerIdentity: Int?

    public init(mode: Mode = .replace, augment: Bool = true, templatesPerIdentity: Int? = 3) {
        self.mode = mode
        self.augment = augment
        self.templatesPerIdentity = templatesPerIdentity
    }
}

/// On-device face enrollment and identification.
///
/// ```swift
/// let recognizer = try await FaceRecognizer(store: FileFaceStore(url: url, key: key))
/// try await recognizer.enroll(id: "emp_042", images: photos)
/// let results = try await recognizer.identify(in: frame, orientation: .right)
/// ```
public actor FaceRecognizer {
    public var thresholds: MatchThresholds { matcher.thresholds }

    private let store: any FaceStore
    private let detector = FaceDetector()
    private let aligner: FaceAligner
    private let renderer = FaceRenderer()
    private let embedder: FaceEmbedder
    private var matcher: Matcher
    private var gallery: [Identity]

    public init(store: any FaceStore = InMemoryFaceStore(),
                thresholds: MatchThresholds = MatchThresholds(),
                aligner: FaceAligner = FaceAligner(),
                modelConfiguration: MLModelConfiguration = MLModelConfiguration()) async throws {
        self.store = store
        self.aligner = aligner
        self.matcher = Matcher(thresholds: thresholds)
        self.embedder = try FaceEmbedder(configuration: modelConfiguration)
        self.gallery = try await store.loadAll()
    }

    public var identities: [Identity] { gallery }

    public func setThresholds(_ thresholds: MatchThresholds) {
        matcher.thresholds = thresholds
    }

    // MARK: Identification

    public func identify(in image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> [Recognition] {
        try identify(upright: CIImage(cgImage: image).oriented(orientation))
    }

    public func identify(in pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation = .up) throws -> [Recognition] {
        try identify(upright: CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation))
    }

    /// Embeddings for every face in the image, without matching.
    public func embeddings(in image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> [(face: DetectedFace, embedding: FaceEmbedding)] {
        try embeddings(upright: CIImage(cgImage: image).oriented(orientation))
    }

    private func identify(upright image: CIImage) throws -> [Recognition] {
        try embeddings(upright: image).map {
            Recognition(face: $0.face, match: matcher.match($0.embedding, against: gallery))
        }
    }

    private func embeddings(upright image: CIImage) throws -> [(face: DetectedFace, embedding: FaceEmbedding)] {
        try detector.detect(in: image).compactMap { face in
            guard let crop = renderer.cgImage(from: aligner.align(face, in: image)) else { return nil }
            return (face, try embedder.embedding(for: crop))
        }
    }

    // MARK: Enrollment

    /// Enrolls the largest face of each image under `id`. Images without a face are skipped.
    @discardableResult
    public func enroll(id: String, images: [CGImage], orientation: CGImagePropertyOrientation = .up,
                       options: EnrollmentOptions = EnrollmentOptions()) async throws -> Identity {
        var collected: [FaceEmbedding] = []
        for cgImage in images {
            let image = CIImage(cgImage: cgImage).oriented(orientation)
            let faces = try detector.detect(in: image)
            guard let face = faces.max(by: { $0.boundingBox.area < $1.boundingBox.area }) else { continue }
            let aligned = aligner.align(face, in: image)
            let variants = options.augment ? [aligned] + aligner.augmentations(of: aligned) : [aligned]
            for variant in variants {
                guard let crop = renderer.cgImage(from: variant) else { continue }
                collected.append(try embedder.embedding(for: crop))
            }
        }
        guard !collected.isEmpty else { throw FaceKitError.noUsableFaces }

        if options.mode == .append, let existing = gallery.first(where: { $0.id == id }) {
            collected += existing.templates
        }
        let templates = options.templatesPerIdentity.map { KMeans(k: $0).centroids(of: collected) } ?? collected
        let identity = Identity(id: id, templates: templates)
        try await store.save(identity)
        gallery.removeAll { $0.id == id }
        gallery.append(identity)
        return identity
    }

    public func remove(id: String) async throws {
        try await store.delete(id: id)
        gallery.removeAll { $0.id == id }
    }

    public func removeAll() async throws {
        try await store.deleteAll()
        gallery.removeAll()
    }
}

private extension CGRect {
    var area: CGFloat { width * height }
}
