import Foundation

/// Spherical k-means used to compress many enrollment embeddings into a few templates.
///
/// Fixes over the original app: points go to the truly nearest centroid, fewer points than
/// `k` is handled, empty clusters are re-seeded, and centroids stay on the unit sphere.
public struct KMeans: Sendable {
    public var k: Int
    public var maxIterations: Int
    public var restarts: Int
    /// Stop when no centroid moves more than this distance.
    public var tolerance: Float

    public init(k: Int, maxIterations: Int = 50, restarts: Int = 5, tolerance: Float = 1e-4) {
        precondition(k > 0, "k must be positive")
        self.k = k
        self.maxIterations = maxIterations
        self.restarts = restarts
        self.tolerance = tolerance
    }

    /// Returns at most `k` centroids. With `k` or fewer points the points themselves are returned.
    public func centroids<G: RandomNumberGenerator>(of points: [FaceEmbedding], using rng: inout G) -> [FaceEmbedding] {
        guard points.count > k else { return points }
        var best: (centroids: [FaceEmbedding], cost: Float)?
        for _ in 0..<max(restarts, 1) {
            let run = cluster(points, using: &rng)
            if best.map({ run.cost < $0.cost }) ?? true { best = run }
        }
        return best!.centroids
    }

    public func centroids(of points: [FaceEmbedding]) -> [FaceEmbedding] {
        var rng = SystemRandomNumberGenerator()
        return centroids(of: points, using: &rng)
    }

    private func cluster<G: RandomNumberGenerator>(_ points: [FaceEmbedding], using rng: inout G) -> (centroids: [FaceEmbedding], cost: Float) {
        var centroids = seed(points, using: &rng)
        var assignment = [Int](repeating: 0, count: points.count)

        for _ in 0..<maxIterations {
            for (i, p) in points.enumerated() {
                assignment[i] = Self.nearest(p, in: centroids).index
            }
            var moved: Float = 0
            for c in centroids.indices {
                let members = points.indices.filter { assignment[$0] == c }.map { points[$0] }
                // Re-seed an empty cluster with the point farthest from its centroid.
                let updated = FaceEmbedding.mean(of: members) ?? farthestPoint(points, from: centroids)
                moved = max(moved, updated.distance(to: centroids[c]))
                centroids[c] = updated
            }
            if moved <= tolerance { break }
        }

        let cost = points.reduce(Float(0)) { total, p in
            let d = Self.nearest(p, in: centroids).distance
            return total + d * d
        }
        return (centroids, cost)
    }

    /// k-means++ seeding.
    private func seed<G: RandomNumberGenerator>(_ points: [FaceEmbedding], using rng: inout G) -> [FaceEmbedding] {
        var centroids = [points[Int.random(in: points.indices, using: &rng)]]
        while centroids.count < k {
            let weights = points.map { p -> Float in
                let d = Self.nearest(p, in: centroids).distance
                return d * d
            }
            let total = weights.reduce(0, +)
            guard total > 0 else {
                // All remaining points coincide with centroids; any point will do.
                centroids.append(points[Int.random(in: points.indices, using: &rng)])
                continue
            }
            var target = Float.random(in: 0..<total, using: &rng)
            var chosen = points.count - 1
            for (i, w) in weights.enumerated() {
                target -= w
                if target < 0 { chosen = i; break }
            }
            centroids.append(points[chosen])
        }
        return centroids
    }

    private func farthestPoint(_ points: [FaceEmbedding], from centroids: [FaceEmbedding]) -> FaceEmbedding {
        points.max { Self.nearest($0, in: centroids).distance < Self.nearest($1, in: centroids).distance }!
    }

    static func nearest(_ point: FaceEmbedding, in centroids: [FaceEmbedding]) -> (index: Int, distance: Float) {
        var best = (index: 0, distance: Float.infinity)
        for (i, c) in centroids.enumerated() {
            let d = point.distance(to: c)
            if d < best.distance { best = (i, d) }
        }
        return best
    }
}
