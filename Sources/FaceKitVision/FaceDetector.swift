import CoreImage
import Vision

/// A face found in an upright image.
public struct DetectedFace: Hashable, Sendable {
    /// Normalised bounding box in Vision coordinates (origin bottom-left, `0...1`).
    public let boundingBox: CGRect
    /// Normalised eye centres (same coordinate space), when landmarks were found.
    public let leftEye: CGPoint?
    public let rightEye: CGPoint?
    /// Head roll in radians as reported by Vision, if available.
    public let roll: Float?
    public let confidence: Float

    public init(boundingBox: CGRect, leftEye: CGPoint? = nil, rightEye: CGPoint? = nil,
                roll: Float? = nil, confidence: Float = 1) {
        self.boundingBox = boundingBox
        self.leftEye = leftEye
        self.rightEye = rightEye
        self.roll = roll
        self.confidence = confidence
    }

    /// Bounding box with a top-left origin, convenient for UIKit/SwiftUI overlays.
    public var boundingBoxTopLeft: CGRect {
        CGRect(x: boundingBox.minX, y: 1 - boundingBox.maxY, width: boundingBox.width, height: boundingBox.height)
    }
}

/// Detects faces and eye landmarks with Vision.
public final class FaceDetector: @unchecked Sendable {
    public init() {}

    /// - Parameter image: an upright image (apply EXIF orientation before calling).
    public func detect(in image: CIImage) throws -> [DetectedFace] {
        let request = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(ciImage: image, orientation: .up).perform([request])
        let size = image.extent.size
        return (request.results ?? []).map { observation in
            let landmarks = observation.landmarks
            return DetectedFace(
                boundingBox: observation.boundingBox,
                leftEye: landmarks?.leftEye.flatMap { Self.normalizedCenter(of: $0, imageSize: size) },
                rightEye: landmarks?.rightEye.flatMap { Self.normalizedCenter(of: $0, imageSize: size) },
                roll: observation.roll?.floatValue,
                confidence: observation.confidence
            )
        }
    }

    private static func normalizedCenter(of region: VNFaceLandmarkRegion2D, imageSize: CGSize) -> CGPoint? {
        let points = region.pointsInImage(imageSize: imageSize)
        guard !points.isEmpty, imageSize.width > 0, imageSize.height > 0 else { return nil }
        let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        let n = CGFloat(points.count)
        return CGPoint(x: sum.x / n / imageSize.width, y: sum.y / n / imageSize.height)
    }
}
