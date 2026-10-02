import Foundation

/// Distance thresholds used to classify a match.
///
/// Defaults are calibrated on LFW with the bundled FaceNet model and default `FaceAligner`
/// (see Tools/FaceKitEval): `confident` 0.9 accepted no impostor pair (0/3000) at 88.6% TAR,
/// `candidate` 1.0 gave 0.03% FAR at 96.0% TAR. These are 1:1 rates; with N enrolled people
/// a stranger's chance of matching someone grows roughly N-fold, so re-measure on your own data.
public struct MatchThresholds: Codable, Hashable, Sendable {
    /// Distances at or below this are an unambiguous match.
    public var confident: Float
    /// Distances at or below this are a probable match that should be confirmed
    /// (for example over several frames with `FrameConsensus`).
    public var candidate: Float

    public init(confident: Float = 0.9, candidate: Float = 1.0) {
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
