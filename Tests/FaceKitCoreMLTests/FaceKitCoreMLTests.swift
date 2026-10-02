import CoreImage
import ImageIO
import XCTest
@testable import FaceKit
@testable import FaceKitVision

private func fixtureURL(_ name: String) -> URL {
    Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)!
}

private func loadCGImage(_ url: URL) -> CGImage {
    let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(source, 0, nil)!
}

final class FaceEmbedderTests: XCTestCase {
    /// The Core ML model and its preprocessing must reproduce the original TensorFlow graph.
    func testMatchesTensorFlowReference() throws {
        let image = loadCGImage(fixtureURL("gradient160.png"))
        let reference = try JSONDecoder().decode([Float].self, from: Data(contentsOf: fixtureURL("gradient160_tf_embedding.json")))

        let embedding = try FaceEmbedder().embedding(for: image)

        XCTAssertEqual(embedding.dimension, 128)
        XCTAssertLessThan(embedding.distance(to: FaceEmbedding(reference)), 0.01)
    }

    func testIsDeterministic() throws {
        let embedder = try FaceEmbedder()
        let image = loadCGImage(fixtureURL("gradient160.png"))
        XCTAssertEqual(try embedder.embedding(for: image).distance(to: try embedder.embedding(for: image)), 0, accuracy: 1e-5)
    }
}

final class FaceAlignerTests: XCTestCase {
    /// A red "face" box in a blue image must fill the crop when there is no margin.
    func testCropsTheFaceBox() throws {
        let blue = CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(to: CGRect(x: 0, y: 0, width: 400, height: 300))
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: CGRect(x: 100, y: 50, width: 120, height: 120))
        let image = red.composited(over: blue)
        let face = DetectedFace(boundingBox: CGRect(x: 0.25, y: 50.0 / 300, width: 0.3, height: 0.4))

        let aligned = FaceAligner(margin: 0).align(face, in: image)
        XCTAssertEqual(aligned.extent, CGRect(x: 0, y: 0, width: 160, height: 160))

        let pixels = try XCTUnwrap(FaceRenderer().cgImage(from: aligned))
        let data = try XCTUnwrap(pixels.dataProvider?.data) as Data
        let bytesPerRow = pixels.bytesPerRow
        for (x, y) in [(5, 5), (80, 80), (154, 154)] {
            let offset = y * bytesPerRow + x * 4
            XCTAssertGreaterThan(data[offset], 200, "red at (\(x), \(y))")
            XCTAssertLessThan(data[offset + 2], 50, "no blue at (\(x), \(y))")
        }
        XCTAssertEqual(FaceAligner().augmentations(of: aligned).map(\.extent), Array(repeating: aligned.extent, count: 3))
    }

    func testLevelsTiltedEyes() {
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 200))
        let level = DetectedFace(boundingBox: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
                                 leftEye: CGPoint(x: 0.4, y: 0.6), rightEye: CGPoint(x: 0.6, y: 0.6))
        let tilted = DetectedFace(boundingBox: level.boundingBox,
                                  leftEye: CGPoint(x: 0.4, y: 0.55), rightEye: CGPoint(x: 0.6, y: 0.65))
        let aligner = FaceAligner()
        // Mapping the tilted eyes through the alignment must level them.
        let a = aligner.alignmentTransform(for: tilted, extent: image.extent)
        let left = CGPoint(x: 80, y: 110).applying(a)
        let right = CGPoint(x: 120, y: 130).applying(a)
        XCTAssertEqual(left.y, right.y, accuracy: 1e-6)
        XCTAssertEqual(aligner.alignmentTransform(for: level, extent: image.extent).b, 0, accuracy: 1e-9)
    }
}

final class FaceRecognizerTests: XCTestCase {
    func testSaveAndReplaceIdentitiesPersistToStore() async throws {
        let store = InMemoryFaceStore([Identity(id: "old", templates: [FaceEmbedding([1, 0])])])
        let recognizer = try await FaceRecognizer(store: store)
        let loaded = await recognizer.identities
        XCTAssertEqual(loaded.map(\.id), ["old"])

        try await recognizer.save(Identity(id: "a", templates: [FaceEmbedding([0, 1])]))
        let afterSave = await recognizer.identities
        XCTAssertEqual(Set(afterSave.map(\.id)), ["old", "a"])

        try await recognizer.replaceAll(with: [Identity(id: "b", templates: [FaceEmbedding([1, 1])])])
        let afterReplace = await recognizer.identities
        let stored = await store.loadAll()
        XCTAssertEqual(afterReplace.map(\.id), ["b"])
        XCTAssertEqual(stored.map(\.id), ["b"])
    }

    func testImageWithoutFaces() async throws {
        let recognizer = try await FaceRecognizer()
        let image = loadCGImage(fixtureURL("gradient160.png"))

        let results = try await recognizer.identify(in: image)
        XCTAssertTrue(results.isEmpty)
        do {
            try await recognizer.enroll(id: "nobody", images: [image])
            XCTFail("enrolling an image without a face must fail")
        } catch let error as FaceKitError {
            XCTAssertEqual(error, .noUsableFaces)
        }
    }

    /// End-to-end check with real photos. Set FACEKIT_FACES_DIR to a folder with one
    /// sub-folder per person (at least two photos each); kept out of the repo for privacy.
    func testRecognisesPeopleFromLocalPhotos() async throws {
        guard let path = ProcessInfo.processInfo.environment["FACEKIT_FACES_DIR"] else {
            throw XCTSkip("FACEKIT_FACES_DIR not set")
        }
        let root = URL(fileURLWithPath: path)
        let people = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .filter(\.hasDirectoryPath)
        let recognizer = try await FaceRecognizer()

        var probes: [(id: String, image: CGImage)] = []
        for person in people {
            let photos = try FileManager.default.contentsOfDirectory(at: person, includingPropertiesForKeys: nil)
                .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
                .sorted { $0.path < $1.path }
                .map(loadCGImage)
            guard photos.count >= 2 else { continue }
            try await recognizer.enroll(id: person.lastPathComponent, images: Array(photos.dropLast()))
            probes.append((person.lastPathComponent, photos.last!))
        }
        for probe in probes {
            let match = try await recognizer.identify(in: probe.image).first?.match
            XCTAssertEqual(match?.nearestID, probe.id, "distance \(match?.distance ?? .infinity)")
        }
    }
}
