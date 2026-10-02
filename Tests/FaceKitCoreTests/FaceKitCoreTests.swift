import CryptoKit
import XCTest
@testable import FaceKitCore

/// Deterministic RNG so clustering tests are reproducible.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Points scattered around `center`, all on the unit sphere.
func cluster(around center: [Float], count: Int, spread: Float, rng: inout SplitMix64) -> [FaceEmbedding] {
    (0..<count).map { _ in FaceEmbedding(center.map { $0 + Float.random(in: -spread...spread, using: &rng) }) }
}

final class FaceEmbeddingTests: XCTestCase {
    func testNormalisesAndMeasuresDistance() {
        let a = FaceEmbedding([3, 4])
        XCTAssertEqual(a.values[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(a.values[1], 0.8, accuracy: 1e-6)
        XCTAssertEqual(FaceEmbedding([1, 0]).distance(to: FaceEmbedding([-1, 0])), 2, accuracy: 1e-6)
        XCTAssertEqual(a.distance(to: a), 0, accuracy: 1e-6)
    }
}

final class MatcherTests: XCTestCase {
    let gallery = [
        Identity(id: "a", templates: [FaceEmbedding([1, 0, 0]), FaceEmbedding([0.9, 0.1, 0])]),
        Identity(id: "b", templates: [FaceEmbedding([0, 1, 0])]),
    ]

    func testConfidentCandidateAndUnknown() {
        let matcher = Matcher(thresholds: MatchThresholds(confident: 0.4, candidate: 0.7))

        let confident = matcher.match(FaceEmbedding([1, 0.05, 0]), against: gallery)
        XCTAssertEqual(confident.decision, .confident)
        XCTAssertEqual(confident.identityID, "a")
        XCTAssertEqual(confident.runnerUpDistance!, FaceEmbedding([1, 0.05, 0]).distance(to: FaceEmbedding([0, 1, 0])), accuracy: 1e-6)

        // Distance ~0.50 to "a": a probable match, not an unambiguous one.
        let candidate = matcher.match(FaceEmbedding([1, 0, 0.55]), against: gallery)
        XCTAssertEqual(candidate.decision, .candidate)
        XCTAssertEqual(candidate.identityID, "a")

        let unknown = matcher.match(FaceEmbedding([0, 0, 1]), against: gallery)
        XCTAssertEqual(unknown.decision, .unknown)
        XCTAssertNil(unknown.identityID)
        XCTAssertNotNil(unknown.nearestID, "nearest id is still reported for diagnostics")
    }

    func testEmptyGallery() {
        let result = Matcher().match(FaceEmbedding([1, 0]), against: [])
        XCTAssertEqual(result.decision, .unknown)
        XCTAssertNil(result.nearestID)
        XCTAssertEqual(result.distance, .infinity)
    }

    func testRunnerUpIgnoresTemplatesOfTheSameIdentity() {
        let result = Matcher().match(FaceEmbedding([1, 0, 0]), against: gallery)
        XCTAssertEqual(result.nearestID, "a")
        // Second-best template belongs to "a" as well; the runner-up must be "b".
        XCTAssertEqual(result.runnerUpDistance!, Float(2).squareRoot(), accuracy: 1e-6)
    }
}

final class KMeansTests: XCTestCase {
    func testFindsWellSeparatedClusters() {
        var rng = SplitMix64(state: 42)
        let centers: [[Float]] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        let points = centers.flatMap { cluster(around: $0, count: 20, spread: 0.1, rng: &rng) }.shuffled(using: &rng)

        let centroids = KMeans(k: 3).centroids(of: points, using: &rng)

        XCTAssertEqual(centroids.count, 3)
        for center in centers {
            let nearest = centroids.map { $0.distance(to: FaceEmbedding(center)) }.min()!
            XCTAssertLessThan(nearest, 0.1, "every true cluster has a centroid close to it")
        }
    }

    func testReturnsPointsWhenThereAreNoMoreThanK() {
        let points = [FaceEmbedding([1, 0]), FaceEmbedding([0, 1])]
        XCTAssertEqual(KMeans(k: 3).centroids(of: points), points)
    }

    func testIdenticalPointsDoNotProduceNaN() {
        var rng = SplitMix64(state: 7)
        let points = Array(repeating: FaceEmbedding([0.6, 0.8]), count: 6)
        let centroids = KMeans(k: 3).centroids(of: points, using: &rng)
        XCTAssertEqual(centroids.count, 3)
        XCTAssertFalse(centroids.contains { $0.values.contains(where: \.isNaN) })
    }
}

final class FrameConsensusTests: XCTestCase {
    func testConfirmsAfterEnoughFramesAndThenResets() {
        var consensus = FrameConsensus(requiredFrames: 3, windowSize: 4)
        XCTAssertNil(consensus.observe("a"))
        XCTAssertNil(consensus.observe(nil))
        XCTAssertNil(consensus.observe("a"))
        XCTAssertEqual(consensus.observe("a"), "a")
        XCTAssertNil(consensus.observe("a"), "window is cleared after a confirmation")
    }

    func testOldObservationsFallOutOfTheWindow() {
        var consensus = FrameConsensus(requiredFrames: 2, windowSize: 2)
        XCTAssertNil(consensus.observe("a"))
        XCTAssertNil(consensus.observe("b"))
        XCTAssertNil(consensus.observe("a"))
        XCTAssertEqual(consensus.observe("a"), "a")
    }
}

final class FaceStoreTests: XCTestCase {
    func testFileStoreRoundTripsEncrypted() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathComponent("faces.bin")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let key = SymmetricKey(size: .bits256)
        let store = FileFaceStore(url: url, key: key)

        try await store.save(Identity(id: "a", templates: [FaceEmbedding([1, 0])]))
        try await store.save(Identity(id: "b", templates: [FaceEmbedding([0, 1])]))
        try await store.save(Identity(id: "a", templates: [FaceEmbedding([0.6, 0.8])]))
        try await store.delete(id: "b")

        let loaded = try await FileFaceStore(url: url, key: key).loadAll()
        XCTAssertEqual(loaded, [Identity(id: "a", templates: [FaceEmbedding([0.6, 0.8])])])

        let raw = try Data(contentsOf: url)
        XCTAssertNil(try? JSONDecoder().decode([Identity].self, from: raw), "file must not be plain JSON")
        do {
            _ = try await FileFaceStore(url: url, key: SymmetricKey(size: .bits256)).loadAll()
            XCTFail("a wrong key must not decrypt the store")
        } catch {}

        try await store.deleteAll()
        let empty = try await store.loadAll()
        XCTAssertTrue(empty.isEmpty)
    }
}
