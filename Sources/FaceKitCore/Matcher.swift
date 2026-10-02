import Foundation

/// Distance thresholds used to classify a match.
///
/// The defaults are the values the original app used with this FaceNet model. They have
/// not been calibrated on a benchmark yet; measure FAR/FRR on your own data before relying on them.
public struct MatchThresholds: Codable, Hashable, Sendable {
    /// Distances at or below this are an unambiguous match.
    public var confident: Float
    /// Distances at or below this are a probable match that should be confirmed
    /// (for example over several frames with `FrameConsensus`).
    public var candidate: Float

    public init(confident: Float = 0.4, candidate: Float = 0.7) {
        precondition(confident <= candidate, "confident must not exceed candidate")
        self.confident = confident
        self.candidate = candidate
    }
}

public struct MatchResult: Hashable, Sendable {
    public enum Decision: Hashable, Sendable {
        case confident
        case candidate
        case unknown
    }

    /// Nearest enrolled identity, if any identity is enrolled.
    public let nearestID: String?
    /// Distance to the nearest template of `nearestID` (`.infinity` when the gallery is empty).
    public let distance: Float
    /// Distance to the nearest template of a *different* identity, if there is one.
    public let runnerUpDistance: Float?
    public let decision: Decision

    /// The matched identity, or `nil` when the face is unknown.
    public var identityID: String? { decision == .unknown ? nil : nearestID }
}

/// Nearest-neighbour matching of an embedding against enrolled identities.
public struct Matcher: Sendable {
    public var thresholds: MatchThresholds

    public init(thresholds: MatchThresholds = MatchThresholds()) {
        self.thresholds = thresholds
    }

    public func match(_ embedding: FaceEmbedding, against gallery: [Identity]) -> MatchResult {
        var best: (id: String, distance: Float)?
        var runnerUp: Float?

        for identity in gallery {
            guard let d = identity.templates.map({ embedding.distance(to: $0) }).min() else { continue }
            if let current = best {
                if d < current.distance {
                    runnerUp = current.distance
                    best = (identity.id, d)
                } else if runnerUp.map({ d < $0 }) ?? true {
                    runnerUp = d
                }
            } else {
                best = (identity.id, d)
            }
        }

        guard let best else {
            return MatchResult(nearestID: nil, distance: .infinity, runnerUpDistance: nil, decision: .unknown)
        }
        let decision: MatchResult.Decision
        if best.distance <= thresholds.confident {
            decision = .confident
        } else if best.distance <= thresholds.candidate {
            decision = .candidate
        } else {
            decision = .unknown
        }
        return MatchResult(nearestID: best.id, distance: best.distance, runnerUpDistance: runnerUp, decision: decision)
    }
}
