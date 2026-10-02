import Foundation

/// A face descriptor produced by an embedding model. Values are L2-normalised on creation,
/// so the Euclidean distance between two embeddings lies in `0...2`.
public struct FaceEmbedding: Codable, Hashable, Sendable {
    public let values: [Float]

    public init(_ values: [Float]) {
        let norm = values.reduce(0) { $0 + $1 * $1 }.squareRoot()
        self.values = norm > 0 ? values.map { $0 / norm } : values
    }

    public var dimension: Int { values.count }

    public func distance(to other: FaceEmbedding) -> Float {
        precondition(dimension == other.dimension, "Embedding dimensions differ")
        var sum: Float = 0
        for i in values.indices {
            let d = values[i] - other.values[i]
            sum += d * d
        }
        return sum.squareRoot()
    }

    /// Normalised mean of several embeddings.
    public static func mean(of embeddings: [FaceEmbedding]) -> FaceEmbedding? {
        guard let first = embeddings.first else { return nil }
        var sum = [Float](repeating: 0, count: first.dimension)
        for e in embeddings {
            for i in sum.indices { sum[i] += e.values[i] }
        }
        return FaceEmbedding(sum)
    }
}

/// An enrolled person: an app-defined identifier plus the templates used for matching.
public struct Identity: Codable, Hashable, Sendable {
    public let id: String
    public var templates: [FaceEmbedding]

    public init(id: String, templates: [FaceEmbedding]) {
        self.id = id
        self.templates = templates
    }
}
