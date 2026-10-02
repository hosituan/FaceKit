import Foundation

/// Confirms an identity only after it has been seen in enough recent frames.
///
/// Feed one observation per frame (`nil` for unknown or no face). When an identity reaches
/// `requiredFrames` occurrences within the last `windowSize` observations it is returned
/// once and the window is cleared, so the next confirmation needs a fresh run of frames.
/// Cool-down rules (for example "log at most once per minute") belong to the app.
public struct FrameConsensus: Sendable {
    public let requiredFrames: Int
    public let windowSize: Int
    private var window: [String?] = []

    public init(requiredFrames: Int = 5, windowSize: Int = 8) {
        precondition(requiredFrames > 0 && windowSize >= requiredFrames,
                     "windowSize must be at least requiredFrames")
        self.requiredFrames = requiredFrames
        self.windowSize = windowSize
    }

    public mutating func observe(_ id: String?) -> String? {
        window.append(id)
        if window.count > windowSize { window.removeFirst(window.count - windowSize) }
        guard let id, window.filter({ $0 == id }).count >= requiredFrames else { return nil }
        window.removeAll()
        return id
    }

    public mutating func reset() {
        window.removeAll()
    }
}
